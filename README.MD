# Examining Joint Demosaicing and Denoising for Single-, Quad-, and Nona-Bayer Patterns

**SaiKiran Tedla, Abhijith Punnappurath, Luxi Zhao, Michael S. Brown**  
Samsung AI Center Toronto & York University

📄 [Paper (PDF)](https://arxiv.org/abs/2504.07145)
📄 [Dataset](https://sites.google.com/view/tedlasai/home/examining-demosaicing)
---




## 📌 Citation

If you use our dataset or code, please cite:

```bibtex
@inproceedings{Tedla2025ExaminingDemosaic,
  title={{Examining Joint Demosaicing and Denoising for Single-, Quad-, and Nona-Bayer Patterns}},
  author={{Tedla, SaiKiran and Punnappurath, Abhijith and Zhao, Luxi and Brown, Michael S}},
  booktitle={{Proceedings of the IEEE International Conference on Computational Photography (ICCP)}},
  year={{2025}}
}
```

---

## 🚀 Getting Started

This section describes how to train and test our unified demosaicing and denoising model.

### 🔧 Environment Setup

```bash
conda env create -f environment.yml
conda activate examine_demosaic
```

You'll also need a [Weights & Biases (wandb)](https://wandb.ai/) account for experiment tracking. Update the `wandb` settings in the YAML files accordingly.

<!-- Additionally download rcan.py from https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py and paste it `PyTorch/arch` and change the first line to import "
Additionally download common.py from https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/common.py and paste it `PyTorch/arch`. -->


### 📂 Dataset

Download the dataset and then update config files accordingly with the appropriate paths. 



### 🏋️‍♂️ Training

To train the model, run:

```bash
python PyTorch/runner.py --config PyTorch/configs/unified_train.yaml
```

Modify the config file to:

- Choose the desired ISO levels  
- Set the correct paths to training patches and full-resolution images  
- Update your wandb project and entity details  

---

### 🧪 Testing

To test the model, run:

```bash
python PyTorch/runner.py --config PyTorch/configs/unified_test.yaml
```

Again, be sure to:

- Set the appropriate ISO levels  
- Provide the correct paths to the test dataset  
- Provide the path to the checkpoint file
---
