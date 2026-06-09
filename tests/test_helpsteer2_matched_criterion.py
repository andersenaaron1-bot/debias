import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aisafety.mech.judge_reasoning import (
    comparison_prompt_content,
    parse_final_verdict,
)
from aisafety.scripts.analyze_helpsteer2_matched_criterion import analyze
from aisafety.scripts.build_helpsteer2_matched_criterion_suite import (
    ATTRIBUTE_NAMES,
    build_comparisons,
    classify_pair,
    criterion_target,
    materialize,
)
from aisafety.scripts.run_judge_reasoning_budget_sweep import _verdict_row


def _source_row(
    prompt: str,
    response: str,
    attributes: tuple[float, ...],
) -> dict:
    return {
        "prompt": prompt,
        "response": response,
        **dict(zip(ATTRIBUTE_NAMES, attributes, strict=True)),
    }


def _synthetic_source_rows() -> list[dict]:
    return [
        _source_row("dominance", "dominant", (4, 4, 4, 3, 3)),
        _source_row("dominance", "dominated", (2, 2, 2, 1, 1)),
        _source_row("single", "correct", (3, 4, 3, 2, 2)),
        _source_row("single", "incorrect", (3, 2, 3, 2, 2)),
        _source_row("tradeoff", "helpful", (4, 2, 3, 2, 2)),
        _source_row("tradeoff", "correct", (2, 4, 3, 2, 2)),
        _source_row("tie", "tie one", (3, 3, 3, 2, 2)),
        _source_row("tie", "tie two", (3, 3, 3, 2, 2)),
    ]


class HelpSteer2MatchedSuiteTests(unittest.TestCase):
    def test_pair_strata_are_distinct(self) -> None:
        self.assertEqual(
            classify_pair(
                (4, 4, 4, 3, 3),
                (2, 2, 2, 1, 1),
                weighted_tie_epsilon=0.05,
            )[0],
            "dominance",
        )
        self.assertEqual(
            classify_pair(
                (3, 4, 3, 2, 2),
                (3, 2, 3, 2, 2),
                weighted_tie_epsilon=0.05,
            )[0],
            "single_attribute",
        )
        self.assertEqual(
            classify_pair(
                (4, 2, 3, 2, 2),
                (2, 4, 3, 2, 2),
                weighted_tie_epsilon=0.05,
            )[0],
            "tradeoff",
        )
        self.assertEqual(
            classify_pair(
                (3, 3, 3, 2, 2),
                (3, 3, 3, 2, 2),
                weighted_tie_epsilon=0.05,
            )[0],
            "near_tie",
        )

    def test_criterion_targets_include_ties(self) -> None:
        left = (4, 2, 3, 2, 2)
        right = (2, 4, 3, 2, 2)
        self.assertEqual(
            criterion_target(
                stratum="tradeoff",
                criterion_id="overall",
                option_a_attributes=left,
                option_b_attributes=right,
                weighted_tie_epsilon=0.05,
            ),
            "C",
        )
        self.assertEqual(
            criterion_target(
                stratum="tradeoff",
                criterion_id="helpfulness",
                option_a_attributes=left,
                option_b_attributes=right,
                weighted_tie_epsilon=0.05,
            ),
            "A",
        )
        self.assertEqual(
            criterion_target(
                stratum="tradeoff",
                criterion_id="correctness",
                option_a_attributes=left,
                option_b_attributes=right,
                weighted_tie_epsilon=0.05,
            ),
            "B",
        )

    def test_materializer_keeps_pair_conditions_on_one_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = materialize(
                rows=_synthetic_source_rows(),
                out_dir=Path(temporary),
                max_pairs_per_stratum=1,
                min_pairs_per_stratum=1,
                num_shards=2,
                weighted_tie_epsilon=0.05,
                seed=1234,
                source_description="synthetic",
            )
            self.assertEqual(manifest["n_pairs"], 4)
            self.assertEqual(manifest["n_comparisons"], 40)
            pair_shards: dict[str, set[int]] = {}
            for shard_index in range(2):
                path = Path(temporary) / f"comparisons_shard_{shard_index}.jsonl"
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    pair_shards.setdefault(row["pair_id"], set()).add(shard_index)
            self.assertTrue(all(len(shards) == 1 for shards in pair_shards.values()))

    def test_prompt_and_parser_support_tie_verdict(self) -> None:
        prompt = comparison_prompt_content(
            {
                "prompt": "Question",
                "option_a_text": "A",
                "option_b_text": "B",
                "criterion_text": "Use the explicit criterion.",
                "allow_tie": True,
            },
            reasoning_mode="thinking",
        )
        self.assertIn("FINAL: C", prompt)
        self.assertEqual(
            parse_final_verdict("Analysis\nFINAL: C", labels=["A", "B", "C"]),
            "C",
        )


class HelpSteer2MatchedAnalysisTests(unittest.TestCase):
    def test_three_way_verdict_and_matched_analysis(self) -> None:
        pair = {
            "pair_id": "pair",
            "origin_pair_id": "pair",
            "prompt": "Explain.",
            "option_a_text": "helpful",
            "option_b_text": "correct",
            "option_a_attributes": dict(
                zip(ATTRIBUTE_NAMES, (4, 2, 3, 2, 2), strict=True)
            ),
            "option_b_attributes": dict(
                zip(ATTRIBUTE_NAMES, (2, 4, 3, 2, 2), strict=True)
            ),
            "pair_stratum": "tradeoff",
            "analysis_split": "fit",
        }
        comparisons = build_comparisons(
            [pair],
            weighted_tie_epsilon=0.05,
        )
        labels = ["A", "B", "C"]
        rows = []
        for comparison in comparisons:
            target_index = labels.index(comparison["target_option"])
            logits = np.full(3, -2.0)
            logits[target_index] = 2.0
            for mode, branch, budget in (
                ("direct", -1, 0),
                ("thinking", 0, 1024),
            ):
                rows.append(
                    _verdict_row(
                        comparison=comparison,
                        run_label="test",
                        model_id="test",
                        prompt_style="chat_template",
                        mode=mode,
                        branch_index=branch,
                        branch_seed=1234,
                        budget_tokens=budget,
                        available_prefix_tokens=budget,
                        natural_choice=comparison["target_option"],
                        full_natural_choice=comparison["target_option"],
                        full_generated_tokens=16,
                        max_budget_saturated=False,
                        logits=logits,
                        labels=labels,
                        trace_id=f"{comparison['comparison_id']}:{mode}",
                    )
                )
        outputs = analyze(rows, bootstrap=20, seed=1234)
        summary = outputs["matched_summary"]
        self.assertTrue(summary["robust_criterion_accuracy"].eq(1.0).all())
        switches = outputs["criterion_switch_summary"]
        self.assertTrue(switches["switch_compliance_rate"].eq(1.0).all())
        tie_row = _verdict_row(
            comparison=next(
                row
                for row in comparisons
                if row["criterion_id"] == "overall"
                and row["presentation_order"] == "original"
            ),
            run_label="test",
            model_id="test",
            prompt_style="chat_template",
            mode="direct",
            branch_index=-1,
            branch_seed=1234,
            budget_tokens=0,
            available_prefix_tokens=0,
            natural_choice="C",
            full_natural_choice="C",
            full_generated_tokens=1,
            max_budget_saturated=False,
            logits=np.asarray([-2.0, -2.0, 2.0]),
            labels=labels,
            trace_id="tie",
        )
        self.assertEqual(tie_row["forced_choice"], "C")
        self.assertTrue(tie_row["forced_target_selected"])


if __name__ == "__main__":
    unittest.main()
