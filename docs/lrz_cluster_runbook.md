# LRZ Cluster Runbook

This is the minimal path from the local repo to containerized Slurm execution on the LRZ AI Cluster.

It assumes:

- the code lives in a GitHub repository
- the runtime image is published to GHCR
- the final training JSONLs are staged on LRZ storage outside the image

Repository and image naming in this runbook:

- GitHub repository name: `debias`
- GHCR image name: `ghcr.io/<YOUR_GITHUB_USER>/debias`
- Python package name inside the container: `aisafety`

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

```bash
mkdir -p $HOME/projects
cd $HOME/projects
git clone git@github.com:<YOUR_GITHUB_USER>/debias.git
cd debias
```

## 4. Stage the required data on LRZ

The image does not contain data.

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
rsync -av --mkpath \
  data/derived/pref_pairs_shp2/pref_pairs_train.jsonl \
  data/derived/pref_pairs_shp2/pref_pairs_val.jsonl \
  data/derived/helpsteer2_anchor/anchor_train.jsonl \
  data/derived/helpsteer2_anchor/anchor_val.jsonl \
  data/derived/style_groups/m2_publishable_v1/style_groups_train.jsonl \
  data/derived/style_groups/m2_publishable_v1/style_groups_val.jsonl \
  data/derived/style_groups/m3_publishable_v1/style_groups_train.jsonl \
  data/derived/style_groups/m3_publishable_v1/style_groups_val.jsonl \
  data/derived/cue_discovery_v2/balanced_splits/corpus_scored_balanced_train.jsonl \
  data/derived/cue_discovery_v2/balanced_splits/corpus_scored_balanced_val.jsonl \
  <LRZ_USER>@<LRZ_LOGIN_HOST>:$HOME/projects/debias/
```

## 5. Import the container once on LRZ

Create a reusable local Enroot image:

```bash
mkdir -p $HOME/containers
enroot import -o $HOME/containers/debias-v0.1.0.sqsh \
  docker://ghcr.io#<YOUR_GITHUB_USER>/debias:v0.1.0
```

## 6. Submit M2-full

Use the generic Slurm script [train_from_config.sbatch](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/cluster/lrz/train_from_config.sbatch).

Example:

```bash
mkdir -p $HOME/projects/debias/.cache/huggingface
mkdir -p $HOME/projects/debias/outputs

sbatch \
  -p <LRZ_PARTITION> \
  --time=24:00:00 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --container-image=$HOME/containers/debias-v0.1.0.sqsh \
  --container-mounts=$HOME/projects/debias:/workspace,$HOME/projects/debias/.cache/huggingface:/workspace/.cache/huggingface \
  $HOME/projects/debias/cluster/lrz/train_from_config.sbatch \
  /workspace/configs/experiments/m2_full_v1.json
```

## 7. Submit M3

```bash
sbatch \
  -p <LRZ_PARTITION> \
  --time=24:00:00 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --container-image=$HOME/containers/debias-v0.1.0.sqsh \
  --container-mounts=$HOME/projects/debias:/workspace,$HOME/projects/debias/.cache/huggingface:/workspace/.cache/huggingface \
  $HOME/projects/debias/cluster/lrz/train_from_config.sbatch \
  /workspace/configs/experiments/m3_full_v1.json
```

## 8. Run evaluation

Once a run is complete, evaluate it with:

```bash
sbatch \
  -p <LRZ_PARTITION> \
  --time=08:00:00 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --container-image=$HOME/containers/debias-v0.1.0.sqsh \
  --container-mounts=$HOME/projects/debias:/workspace,$HOME/projects/debias/.cache/huggingface:/workspace/.cache/huggingface \
  $HOME/projects/debias/cluster/lrz/eval_reward_suite.sbatch \
  /workspace/artifacts/reward/m2_full_v1
```

## 9. Interactive smoke test

Before the first full job, do one short interactive test:

```bash
salloc -p <LRZ_PARTITION> --gres=gpu:1 --time=01:00:00 --cpus-per-task=8

srun --pty \
  --container-image=$HOME/containers/debias-v0.1.0.sqsh \
  --container-mounts=$HOME/projects/debias:/workspace,$HOME/projects/debias/.cache/huggingface:/workspace/.cache/huggingface \
  bash

cd /workspace
python -m aisafety.scripts.run_experiment_config \
  --config /workspace/configs/experiments/m2_full_v1.json \
  --print-only
```

If that prints a correct command and the mounted files are visible, the batch jobs are ready.
