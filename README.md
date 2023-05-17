# BatchScore
This repository contains the code for batch scoring images of unlimited max number(minimum of 2) using the gigachad open-source PickScore model and a single knockout(hunger games style) method.

## Demo
For a very simple demo to compare just 2 images with a prompt visit their hosted space for PickScore at [HF Spaces](https://huggingface.co/spaces/yuvalkirstain/PickScore).

## Installation
Create a virual env and download torch:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

and then install the rest of the requirements:
```bash
pip install -r requirements.txt
pip install -e .
```

## Inference with BatchScore
For running inference with PickScore as a preference predictor over a folder of images and a given text prompt see the batchscore.py script, example:
```
batchscore.py --dir "path/to/dir" --prompt "cool prompt here"

```

## PickScore
They open-sourced the [Pick-a-Pic dataset](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1) and [PickScore model](https://huggingface.co/yuvalkirstain/PickScore_v1). Readers are encouraged to experiment with the [Pick-a-Pic's web application](https://pickapic.io/) and contribute to the dataset.

## The Paper
[Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://arxiv.org/abs/2305.01569).

## Citation
```bibtex
@inproceedings{Kirstain2023PickaPicAO,
  title={Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation},
  author={Yuval Kirstain and Adam Polyak and Uriel Singer and Shahbuland Matiana and Joe Penna and Omer Levy},
  year={2023}
}
```
