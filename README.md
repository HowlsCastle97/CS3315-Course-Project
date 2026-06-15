# Bird Species Classification Using Deep Learning

**CS 3315: Big Data — Final Project**
LT David Escalera, USN

A data science project that trains a deep convolutional neural network to identify 200 bird
species from photographs, then evaluates how reliable that classifier remains when someone
deliberately tries to fool it. The motivation is real-world: citizen-science apps like Merlin
and iNaturalist feed automated predictions into conservation databases, so a model that can be
manipulated quietly corrupts the data those databases depend on.

## Dataset

[CUB-200-2011 (Caltech-UCSD Birds 200)](https://www.vision.caltech.edu/datasets/cub_200_2011/) —
11,788 images across 200 North American species (5,994 train / 5,794 test), roughly balanced at
~30 images per class. This is a *fine-grained* classification task: many species differ only in
subtle plumage, beak shape, or eye markings.

## Approach

| Model | Training | Clean Accuracy | Macro F1 |
|-------|----------|---------------:|---------:|
| Baseline ResNet-50 | ImageNet transfer learning, 20 epochs | 81.46% | 0.8148 |
| Adversarially Trained ResNet-50 | 70/30 clean–adversarial loss mix (PGD) | 16.57% | 0.1783 |

Robustness was then probed with three attacks: **PGD** and **Square Attack** (evasion, at test
time) and a **Feature-Collision** clean-label **poisoning** attack (at training time).

## Key Finding

The baseline model looks excellent on a standard evaluation (81% accuracy) but collapses under
attack — down to **4.19%** under Square Attack at eps=8/255. The poisoning attack flipped a
single target image from a correct 98.78% prediction to a wrong class while *overall accuracy
actually rose* to 81.12%, meaning a standard evaluation would never have caught it. Strong clean
accuracy is not the same thing as a trustworthy model.

## Repository Contents

```
.
├── bird_classification_cs3315.ipynb  
├── bird_classification_cs3315.html   
├── requirements.txt                  
└── README.md
```

> **Not included:** the CUB-200-2011 dataset (~1.1 GB) is too large for git and must be
> downloaded separately — see step 2 below. Trained model weights are also not committed;
> they are regenerated when you run the notebook.

## Reproducing (First-Time Setup)

These steps assume nothing about your environment beyond Python and (for the full pipeline) a
CUDA GPU.

**1. Install dependencies.**

```bash
python -m venv birds && source birds/bin/activate    # or use conda
pip install -r requirements.txt
```

The one pin that matters most is `torchvision>=0.13` — older versions lack the pretrained-weights
API the notebook uses and will error on the first model build.

**2. Download the dataset.** CUB-200-2011 is not in this repo. Get it from the
[Caltech-UCSD page](https://www.vision.caltech.edu/datasets/cub_200_2011/) and extract it; the
folder must contain `images/`, `images.txt`, `image_class_labels.txt`, `train_test_split.txt`,
and `classes.txt`.

**3. Tell the notebook where the data is.** In the config cell, the data path is read from an
environment variable so you don't have to edit code:

```python
DATA_DIR = os.environ.get('CUB_DATA_DIR', os.path.expanduser('~/CUB_200_2011'))

_required = ['images.txt', 'image_class_labels.txt', 'train_test_split.txt', 'classes.txt', 'images']
_missing = [f for f in _required if not os.path.exists(os.path.join(DATA_DIR, f))]
if _missing:
    raise FileNotFoundError(
        f"CUB-200-2011 not found at {DATA_DIR} (missing: {_missing}). "
        f"Download it from https://www.vision.caltech.edu/datasets/cub_200_2011/ "
        f"and set the CUB_DATA_DIR environment variable to its location."
    )
```

Then just `export CUB_DATA_DIR=/path/to/CUB_200_2011` before running — no code edit needed.

**4. Run the notebook cells in order**, or convert and run as a script (see the Hamming section
below for the batch-job version).

**Core libraries:** PyTorch, torchvision, [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox),
scikit-learn, NumPy, SciPy, Matplotlib, Pillow, tqdm. The full pipeline was run on an NVIDIA A40
GPU; it will run on CPU but the attacks (especially Square Attack, ~90 min on the A40) become
impractically slow.

**A note on exact reproducibility:** the notebook uses random data augmentation and does not fix a
random seed, and GPU operations are not fully deterministic, so a fresh run will produce results
that are *close to* but not bit-identical to the reported numbers (e.g. ~81%, not exactly 81.46%).
To get closer, add a seed near the top of the notebook:

```python
import random, numpy as np, torch
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
```

Full bitwise determinism on GPU additionally requires `torch.use_deterministic_algorithms(True)`
and cuDNN settings, which can slow training and is generally not worth it here.

## Running on Hamming (NPS HPC Cluster)

Hamming is a SLURM-managed cluster, so the project is run by submitting a batch job rather
than executing the notebook interactively on the login node. The steps below assume the
Jupyter notebook has been exported to a Python script; the full pipeline (training both
models, all attacks) is GPU-bound and long-running, so it should never be run on the login
node.

> **Note:** Partition name, account/allocation, and exact module names vary per user and
> change over time. Confirm yours with `module avail`, `sinfo`, and your Hamming welcome
> email, then substitute them into the script below. The GPU request (`--gres=gpu:a40:1`)
> matches the NVIDIA A40 used for the results in this report.

**1. Log in and set up the environment** (one time):

```bash
ssh <your_username>@hamming.nps.edu

module load python        # or the anaconda/miniconda module available on Hamming
conda create -n birds python=3.10 -y
conda activate birds
pip install -r requirements.txt
```

**2. Stage the dataset.** Get `CUB_200_2011.tgz` onto the cluster and extract it. If Hamming's
nodes block outbound internet (common on HPC systems), download the archive on your own machine
and copy it up rather than fetching it on the cluster:

```bash
# On your laptop:
scp CUB_200_2011.tgz <your_username>@hamming.nps.edu:~

# On Hamming:
tar -xzf ~/CUB_200_2011.tgz -C $HOME     # extracts to $HOME/CUB_200_2011
```

The `export CUB_DATA_DIR=...` line in the batch script (step 4) points the code at this location.

**3. Convert the notebook to a script** (run once, on the login node — this is lightweight):

```bash
jupyter nbconvert --to script bird_classification_cs3315.ipynb
# produces bird_classification_cs3315.py
```

**4. Create a SLURM batch script** named `run_birds.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=bird_cls
#SBATCH --partition=<gpu_partition>     # e.g. from `sinfo`; confirm the GPU partition name
#SBATCH --account=<your_account>        # your allocation, if Hamming requires one
#SBATCH --gres=gpu:a40:1                # 1 NVIDIA A40 (matches the report's results)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00                 # full pipeline incl. attacks is long; pad generously
#SBATCH --output=birds_%j.out           # %j = job ID
#SBATCH --error=birds_%j.err

module load python                      # match the module you used in step 1
source activate birds                   # or `conda activate birds`

export CUB_DATA_DIR=$HOME/CUB_200_2011  # point at your extracted dataset
python bird_classification_cs3315.py
```

**5. Submit and monitor:**

```bash
sbatch run_birds.sbatch     # queues the job, prints a job ID
squeue -u $USER             # check status (PD = pending, R = running)
tail -f birds_<jobid>.out   # watch live training/attack output
scancel <jobid>             # cancel if needed
```

For a quick test or debugging, grab an interactive GPU session instead of submitting a batch
job, then run Python directly:

```bash
srun --partition=<gpu_partition> --gres=gpu:a40:1 --cpus-per-task=4 --mem=32G \
     --time=01:00:00 --pty bash
conda activate birds
python bird_classification_cs3315.py
```

## Author

LT David Escalera, USN — CS 3315, Naval Postgraduate School
