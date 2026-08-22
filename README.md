# Robust Training against Malicious Platforms (RTMP)

This directory contains a public release version of the paper code.

## Installation
Clone the repository and navigate to the project directory.

```bash
git clone https://github.com/islab-shi/RTMP.git
cd RTMP
```
Install the required dependencies.
```
pip install -r requirements.txt
```

## Data Preparation

Prepare ImageNet locally with the following directory structure. The dataset is not included in this repository.

```text
/path/to/imagenet/
  train/
    class_x/xxx.JPEG
    ...
  val/
    class_y/yyy.JPEG
    ...
```

## Quick Start

```bash
python autoDefence_imagenet_paper.py \
  --dataset-dir /path/to/imagenet \
  --output-dir ./auto_defence_imagenet
```


## Outputs

The script writes the following outputs under `--output-dir`:

- `atk_result_pkl/`: Exploration results.
- `point/`: Sensitive-kernel distribution plots.
- `atk_result_json/`: Single-kernel attack configurations.
- `malicious_model/`: Models after attack injection.
- `fineturning_model/`: Models after fine-tuning.
- `acc_drop/`: Accuracy curves for each defence target.

## Citation

If this project contributes to your research, we would appreciate it if you could cite our work:

```bibtex
@inproceedings{guoiconip2026,
        author={Guo, Chao and Shi, Youhua},
        booktitle={Neural Information Processing},
        title={Robust Training to Secure Automated AI Accelerator Generation Against Malicious Platforms},
        year={2026},
}
```