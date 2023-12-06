# QTNet
## [NeurIPS 2023] **Q**uery-based **T**emporal Fusion with Explicit Motion for 3D Object Detection
[![paper](https://img.shields.io/badge/OpenReview-Paper-<COLOR>.svg)](https://openreview.net/pdf?id=gySmwdmVDF)

## Introduction
This repository is an official implementation of QTNet.

In this paper, we propose a simple and effective Query-based Temporal Fusion Network (QTNet). The main idea is to exploit the object queries in previous frames to enhance the representation of current object queries by the proposed Motion-guided Temporal Modeling (MTM) module, which utilizes the spatial position information of object queries along the temporal dimension to construct their relevance between adjacent frames reliably. Our method can be integrated into some advanced LiDAR-only or multi-modality 3D detectors and even brings new SOTA performance with negligible computation cost and latency on the nuScenes dataset.

## News
- [2023-12-06] The memory bank training code is released.
- [2023-09-22] QTNet is accepted by NeurIPS 2023. :fire:

## Preparation
* Memory Bank Data

  Please install the [memory bank](https://drive.google.com/file/d/1bzZzu_PHtFt19HKyWWvHdTSzCGMpSMFD/view?usp=sharing) of TransFusion-L, which contains query features and detection results.

* Training and Validation Infos

  You need to run the data conver script in this repo (create_data). We also provided the processed [training](https://drive.google.com/file/d/1CxQMho0qa1UuRIsBlPnqJrRG-671K-tc/view?usp=sharing) and [validation](https://drive.google.com/file/d/1nZPn6SIIAAWrC_h-o-hReDSnuR5_LCU1/view?usp=sharing) infos.

After preparation, you will be able to see the following directory structure:

  ```
  QTNet
  ├── configs
  │   ├── qtnet.py
  ├── mmdet3d
  ├── tools
  ├── data
  │   ├── nuscenes
  │     ├── memorybank
  │     ├── ...
  ├── ...
  ├── README.md
  ```

## Train & inference
You can train the model following:
```bash
tools/dist_train.sh configs/qtnet.py 8 --work-dir work_dirs/qtnet_4frames/
```
You can evaluate the model following:
```bash
tools/dist_test.sh configs/qtnet.py work_dirs/qtnet_4frames/latest.pth 8 --eval mAP
```

## Results on NuScenes Val Set.
| Model |     Setting      | mAP  | NDS  |           Config           | Download  |
|:-----:|:----------------:|:----:|:----:|:--------------------------:|:---------:|
| QTNet | LiDAR - 4 frames | 66.5 | 70.9 | [config](configs/qtnet.py) | [model](https://drive.google.com/file/d/1zHZ4dI-EMnxLF_ZuOPdKVauoLBPF6_an/view?usp=sharing) |

## Results on NuScenes Test Set.
| Model |     Setting      | mAP  | NDS  |
|:-----:|:----------------:|:----:|:----:|
| QTNet | LiDAR - 4 frames | 68.4 | 72.2 |

## TODO
- [x] Release the paper.
- [x] Release the code of QTNet (Memory Bank Training).
- [ ] Release the code of QTNet (End-to-End Training).

## Acknowledgements

We thank these great works and open-source repositories:
[TransFusion](https://github.com/XuyangBai/TransFusion), [DeepInteraction](https://github.com/fudan-zvg/DeepInteraction), and [MMDetection3d](https://github.com/open-mmlab/mmdetection3d).


## Citation
```bibtex
@inproceedings{hou2023querybased,
  title={Query-based Temporal Fusion with Explicit Motion for 3D Object Detection},
  author={Jinghua Hou and Zhe Liu and dingkang liang and Zhikang Zou and Xiaoqing Ye and Xiang Bai},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
}
```