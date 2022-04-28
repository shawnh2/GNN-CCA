# GNN-CCA
This is a DGL implementation of [GNN-CCA](https://arxiv.org/abs/2201.06311) for multi-view detections.

The original GNN-CCA was implemented in [PyGeometric](https://github.com/pyg-team/pytorch_geometric). This repo re-implements in [DGL](https://github.com/dmlc/dgl). Both are using PyTorch.

![cover](doc/seq_3_frame_2001.jpg)

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
2. Run `python preprocess/${DATA_NAME}.py` with `${DATA_NAME}` be the lower case. For example:
```
python preprocess/epfl.py
```

### Model
1. Download ReID model from [here](https://drive.google.com/file/d/1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB/view) and assume its path is `PATH_TO_REID_MODEL`.

## Training
Training model on a specific dataset.

For example, training on EPFL dataset with all sequences:
```bash
python run.py --train --reid-path ${PATH_TO_REID_MODEL} --epfl --seq-name all
```
training on EPFL dataset with specific sequences:
```bash
python run.py --train --reid-path ${PATH_TO_REID_MODEL} --epfl --seq-name terrace passageway
```
You can also change the ReID model (served as the feature extractor) refer to [here](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html), and assume its name is `NAME_OF_REID_MODEL`.
Then you can train your model by running:
```bash
python run.py --reid-name ${NAME_OF_REID_MODEL} --reid-path ${PATH_TO_REID_MODEL} ...
```
After each epoch, the trained model will be saved in the directory assigned by `--output`.

Finally, denote `PATH_TO_MODEL` as the trained model.

## Testing
Testing model on a specific dataset.

For example, testing on EPFL dataset with all sequences:
```bash
python run.py --test --reid-path ${PATH_TO_REID_MODEL} --ckpt ${PATH_TO_MODEL} --epfl --seq-name all
```
and you can also plot the results of some sequences by adding `--visualize`:
```bash
python run.py --test --reid-path ${PATH_TO_REID_MODEL} --ckpt ${PATH_TO_MODEL} --epfl --seq-name terrace laboratory --visualize
```
The results will be saved in the directory assigned by `--output`.

## Citation
```
@article{luna2022gnncca,
  title={Graph Neural Networks for Cross-Camera Data Association},
  author={Luna, Elena and SanMiguel, Juan C. and Martínez, José M. and Carballeira, Pablo},
  journal={arXiv preprint arXiv:2201.06311},
  year={2022}
}
```
