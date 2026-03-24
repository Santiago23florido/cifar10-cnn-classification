# CIFAR-10 CNN Classification

This repository contains a course project on supervised image classification with convolutional neural networks (CNNs) on CIFAR-10.  
The work combines code, notebook-driven experiments, visual analysis, and a LaTeX report.

## Project scope

The project studies a CNN pipeline end to end:

- dataset structuring and preprocessing
- baseline CNN design
- learning dynamics under different batch sizes and optimizers
- hyperparameter identification and classification
- model refinement with a deeper multi-convolution architecture
- overfitting analysis and regularization
- activation-map visualization and interpretation

The main deliverable is the report in [`docs/rappport/`](docs/rappport), supported by the notebook [`TP3_CNN.ipynb`](TP3_CNN.ipynb).

## What was done

The repository is organized around the report sections:

1. **Data structuring**  
   Reduced CIFAR-10 splits are built from the original dataset, and images are standardized before training.

2. **Baseline architecture**  
   A first CNN is defined and analyzed as the reference model.

3. **Learning study**  
   The effect of batch size and optimizer choice is measured under a controlled protocol.

4. **Hyperparameter analysis**  
   Hyperparameters are identified across the notebook and helper code, then classified by nature and discussed in terms of expected impact.

5. **Model refinement**  
   The baseline is replaced by a deeper CNN with several convolutional stages, and the improved architecture is evaluated with training curves and confusion matrices.

6. **Overfitting study**  
   A clear overfitting example is extracted from the workflow, then regularization mechanisms such as dropout and L2 are compared.

7. **Activation maps**  
   Activation maps from shallow and deep layers are visualized and interpreted, including their evolution across training epochs.

## Repository structure

- [`TP3_CNN.ipynb`](TP3_CNN.ipynb): main notebook, used as the central entry point for the experimental workflow
- [`scripts/`](scripts): reusable Python code grouped by report section
  - [`scripts/section_4/`](scripts/section_4)
  - [`scripts/section_5/`](scripts/section_5)
  - [`scripts/section_6/`](scripts/section_6)
  - [`scripts/section_7/`](scripts/section_7)
  - [`scripts/section_8/`](scripts/section_8)
- [`results/`](results): cached experimental outputs used to avoid recomputing every study
- [`docs/rappport/`](docs/rappport): LaTeX report source and generated figures
- [`cours/`](cours): local course material used as theoretical reference during the report writing

## Requirements

The project is configured with Poetry and targets Python 3.12.

Main dependencies:

- `numpy`
- `matplotlib`
- `tensorflow` / `tensorflow-intel`
- `jupyterlab`
- `ipykernel`

See [`pyproject.toml`](pyproject.toml) for the exact versions.

## Setup

Install the environment with Poetry:

```bash
poetry install --with dev
```

Then start Jupyter:

```bash
poetry run jupyter lab
```

## Dataset note

The code expects a **local CIFAR-10 archive** to already exist in the Keras cache, for example under:

- `~/.keras/datasets/cifar-10-batches-py-target_archive`
- `~/.keras/datasets/cifar-10-batches-py-target.tar.gz`

The data-loading helpers read the archive locally instead of downloading it during the experiments.

## How to use the project

1. Open [`TP3_CNN.ipynb`](TP3_CNN.ipynb).
2. Run the notebook sections in order.
3. Use the cached JSON results in [`results/`](results) to reproduce figures quickly.
4. Recompile the report in [`docs/rappport/`](docs/rappport) if the text or figures are updated.

## Outputs

The repository contains:

- trained-model artifacts such as [`my_model.h5`](my_model.h5)
- cached experimental summaries in [`results/`](results)
- report figures in [`docs/rappport/imgs/`](docs/rappport/imgs)
- the compiled PDF report in [`docs/rappport/rap.pdf`](docs/rappport/rap.pdf)

## Notes

- The notebook is intentionally the visible orchestration layer.
- Longer or repetitive logic is factored into `scripts/section_*`.
- Some results are cached to keep reruns manageable on CPU-only setups.
