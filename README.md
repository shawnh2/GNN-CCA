# GNN-CCA
This is a DGL implementation of [GNN-CCA](https://arxiv.org/abs/2201.06311) for multi-view detections.

The original GNN-CCA was implemented in [PyGeometric](https://github.com/pyg-team/pytorch_geometric). This repo re-implements in [DGL](https://github.com/dmlc/dgl). Both are using PyTorch.

## Installation
1. Install PyTorch (>= 1.9.0) and [DGL](https://www.dgl.ai/pages/start.html)
2. Install other requirements:
``` 
pip install -r requirements.txt
```
3. Install [torchreid](https://github.com/KaiyangZhou/deep-person-reid) (follow its instruction)

## Preparation
### Dataset
Assume `DATA_NAME` is the directory in `dataset` folder.

1. Download dataset. Please refer to `dataset/${DATA_NAME}/README.md`
2. Run `python preprocess/${DATA_NAME}.py` with `%{DATA_NAME}` be the lower case. For example:
```
python preprocess/epfl.py
```

### Model
1. Download ReID model from [here](https://drive.google.com/file/d/1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB/view) and assume its path is `PATH_TO_REID_MODEL`.

## Training
Training model on a specific dataset.

For example, training on EPFL dataset with all sequences:
```bash
python train.py --device cuda --reid-path ${PATH_TO_REID_MODEL} --epfl --seq-name all
```
training on EPFL dataset with specific sequences:
```bash
python train.py --device cuda --reid-path ${PATH_TO_REID_MODEL} --epfl --seq-name campus terrace passageway
```
You can also change the ReID model (served as the feature extractor) refer to [here](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html), and assume its name is `NAME_OF_REID_MODEL`.
Then you can train your model by running:
```bash
python train.py --reid-name ${NAME_OF_REID_MODEL} --reid-path ${PATH_TO_REID_MODEL} ...
```

## Testing

## Demo

## Citation
```
@article{luna2022gnncca,
  title={Graph Neural Networks for Cross-Camera Data Association},
  author={Luna, Elena and SanMiguel, Juan C. and Martínez, José M. and Carballeira, Pablo},
  journal={arXiv preprint arXiv:2201.06311},
  year={2022}
}
```
