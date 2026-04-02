# LRZ Cluster Runbook

This is the minimal path from the local repo to containerized Slurm execution on the LRZ AI Cluster.

It assumes:

- the code lives in a GitHub repository
- the runtime image is published to GHCR
- the final training JSONLs are staged on LRZ storage outside the image
- Git on LRZ is done via HTTPS + PAT
- GPU jobs are submitted with an explicit node specification placeholder in the examples below

Repository and image naming in this runbook:

- GitHub repository name: `debias`
- GHCR image name: `ghcr.io/<YOUR_GITHUB_USER>/debias`
- Python package name inside the container: `aisafety`

Recommended LRZ host layout:

- code checkout: `$HOME/debias`
- DSS root for heavy I/O: `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/debias`
- Hugging Face cache on DSS
- training data on DSS
- artifacts and logs on DSS

Important LRZ note:

- `enroot` is not available on login nodes
- use Slurm with direct Pyxis image pulls via `--container-image=ghcr.io#...`
- if the GHCR package is private, direct pulls will require registry auth; the simplest path is to keep the GHCR image public

## 1. Create and push the GitHub repo

If you already created the empty repository on GitHub, from the local repo root run:

```bash
git branch -M main
git remote add origin git@github.com:<YOUR_GITHUB_USER>/debias.git
git add .
git commit -m "Add LRZ container and Slurm training path"
git push -u origin main
```

If you use GitHub CLI, you can create and push in one step:

```bash
gh repo create <YOUR_GITHUB_USER>/debias --private --source . --remote origin --push
```

## 2. Build and publish the container

The repo includes:

- [Dockerfile](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/Dockerfile)
- [ghcr.yml](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/.github/workflows/ghcr.yml)

Recommended path:

- push to `main`
- trigger the `Build And Publish Container` workflow from GitHub Actions

If you want to publish a tagged image:

```bash
git tag v0.1.0
git push origin v0.1.0
```

That publishes an image like:

```text
ghcr.io/<YOUR_GITHUB_USER>/debias:v0.1.0
```

Manual local build and push is also possible:

```bash
docker login ghcr.io -u <YOUR_GITHUB_USER>
docker buildx build \
  --platform linux/amd64 \
  -t ghcr.io/<YOUR_GITHUB_USER>/debias:manual \
  --push .
```

## 3. Clone the repo on LRZ

On LRZ:

If the repo is private and SSH keys do not work on LRZ, clone via HTTPS + PAT:

```bash
read -p "GitHub user: " GH_USER; read -s -p "GitHub PAT: " GH_PAT; echo; git clone https://${GH_USER}:${GH_PAT}@github.com/${GH_USER}/debias.git $HOME/debias; unset GH_PAT; cd $HOME/debias
```

If the repo is public:

```bash
git clone https://github.com/<YOUR_GITHUB_USER>/debias.git $HOME/debias; cd $HOME/debias
```

## 4. Stage the required data on LRZ

The image does not contain data.

Recommended DSS setup:

```bash
export DSS_ROOT=/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/debias; mkdir -p $DSS_ROOT/data $DSS_ROOT/artifacts $DSS_ROOT/slurm $DSS_ROOT/.cache/huggingface
```

Stage the final training files into the same relative locations under the repo checkout:

```text
data/derived/pref_pairs_shp2/pref_pairs_train.jsonl
data/derived/pref_pairs_shp2/pref_pairs_val.jsonl
data/derived/helpsteer2_anchor/anchor_train.jsonl
data/derived/helpsteer2_anchor/anchor_val.jsonl
data/derived/style_groups/m2_publishable_v1/style_groups_train.jsonl
data/derived/style_groups/m2_publishable_v1/style_groups_val.jsonl
data/derived/style_groups/m3_publishable_v1/style_groups_train.jsonl
data/derived/style_groups/m3_publishable_v1/style_groups_val.jsonl
data/derived/cue_discovery_v2/balanced_splits/corpus_scored_balanced_train.jsonl
data/derived/cue_discovery_v2/balanced_splits/corpus_scored_balanced_val.jsonl
```

If you have shell access from your local machine, one simple option is:

```bash
export DSS_ROOT=/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/debias; ssh <LRZ_USER>@<LRZ_LOGIN_HOST> "mkdir -p $DSS_ROOT/data/derived" && rsync -av data/derived/pref_pairs_shp2 data/derived/helpsteer2_anchor data/derived/style_groups/m2_publishable_v1 data/derived/style_groups/m3_publishable_v1 data/derived/cue_discovery_v2 <LRZ_USER>@<LRZ_LOGIN_HOST>:$DSS_ROOT/data/derived/
```

## 5. Submit M2-full

Use the generic Slurm script [train_from_config.sbatch](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/cluster/lrz/train_from_config.sbatch).

Example:

```bash
export DSS_ROOT=/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/debias; sbatch -p <LRZ_PARTITION> --nodelist=<GPU_NODE_SPEC> --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --output=$DSS_ROOT/slurm/slurm-%x-%j.out --error=$DSS_ROOT/slurm/slurm-%x-%j.err --container-image=ghcr.io#<YOUR_GITHUB_USER>/debias:v0.1.0 --container-mounts=$HOME/debias:/workspace,$DSS_ROOT/data:/workspace/data,$DSS_ROOT/artifacts:/workspace/artifacts,$DSS_ROOT/.cache/huggingface:/workspace/.cache/huggingface $HOME/debias/cluster/lrz/train_from_config.sbatch /workspace/configs/experiments/m2_full_v1.json
```

## 6. Submit M3

```bash
export DSS_ROOT=/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/debias; sbatch -p <LRZ_PARTITION> --nodelist=<GPU_NODE_SPEC> --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --output=$DSS_ROOT/slurm/slurm-%x-%j.out --error=$DSS_ROOT/slurm/slurm-%x-%j.err --container-image=ghcr.io#<YOUR_GITHUB_USER>/debias:v0.1.0 --container-mounts=$HOME/debias:/workspace,$DSS_ROOT/data:/workspace/data,$DSS_ROOT/artifacts:/workspace/artifacts,$DSS_ROOT/.cache/huggingface:/workspace/.cache/huggingface $HOME/debias/cluster/lrz/train_from_config.sbatch /workspace/configs/experiments/m3_full_v1.json
```

## 7. Run evaluation

Once a run is complete, evaluate it with:

```bash
export DSS_ROOT=/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/debias; sbatch -p <LRZ_PARTITION> --nodelist=<GPU_NODE_SPEC> --time=08:00:00 --gres=gpu:1 --cpus-per-task=8 --output=$DSS_ROOT/slurm/slurm-%x-%j.out --error=$DSS_ROOT/slurm/slurm-%x-%j.err --container-image=ghcr.io#<YOUR_GITHUB_USER>/debias:v0.1.0 --container-mounts=$HOME/debias:/workspace,$DSS_ROOT/data:/workspace/data,$DSS_ROOT/artifacts:/workspace/artifacts,$DSS_ROOT/.cache/huggingface:/workspace/.cache/huggingface $HOME/debias/cluster/lrz/eval_reward_suite.sbatch /workspace/artifacts/reward/m2_full_v1
```

## 8. Interactive smoke test

Before the first full job, do one short interactive test:

```bash
export DSS_ROOT=/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/debias; salloc -p <LRZ_PARTITION> --nodelist=<GPU_NODE_SPEC> --gres=gpu:1 --time=01:00:00 --cpus-per-task=8
```

```bash
export DSS_ROOT=/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/debias; srun --pty --container-image=ghcr.io#<YOUR_GITHUB_USER>/debias:v0.1.0 --container-mounts=$HOME/debias:/workspace,$DSS_ROOT/data:/workspace/data,$DSS_ROOT/artifacts:/workspace/artifacts,$DSS_ROOT/.cache/huggingface:/workspace/.cache/huggingface bash -lc 'cd /workspace && python -m aisafety.scripts.run_experiment_config --config /workspace/configs/experiments/m2_full_v1.json --print-only'
```

If that prints a correct command and the mounted files are visible, the batch jobs are ready.
