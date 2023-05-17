# PickScore
This repository contains the code for the paper [Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://arxiv.org/abs/2305.01569). 

We also open-source the [Pick-a-Pic dataset](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1) and [PickScore model](https://huggingface.co/yuvalkirstain/PickScore_v1). We encourage readers to experiment with the [Pick-a-Pic's web application](https://pickapic.io/) and contribute to the dataset.

## Demo
We created a simple demo for PickScore at [HF Spaces](https://huggingface.co/spaces/yuvalkirstain/PickScore), check it out :)

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

## Inference with PickScore
For running inference with PickScore as a preference predictor over a folder of images and a given text prompt see the batchscore.py script, example:
```
batchscore.py --dir "path/to/dir" --prompt "cool prompt here"

```


## Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{Kirstain2023PickaPicAO,
  title={Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation},
  author={Yuval Kirstain and Adam Polyak and Uriel Singer and Shahbuland Matiana and Joe Penna and Omer Levy},
  year={2023}
}
```
