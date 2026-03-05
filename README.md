# Species Classification Using Deep Learning on Kirganelia Herbarium Images

The goal of this project is to evaluate how well deep learning models can classify **30 species of Kirganelia** using **cropped leaf images from herbarium specimens**.
The experiments compare three convolutional neural network (CNN) architectures: ResNet50, EfficientNetV2-S, and MobileNetV3-Large.
All models are trained using transfer learning with ImageNet weights and evaluated under different strategies for handling class imbalance.

---

# Dataset

The dataset used in this project consists of **12,946 cropped leaf images** derived from **314 herbarium specimens** representing **30 Kirganelia species**.
Original herbarium images were obtained from the Naturalis Biodiversity Center digital collections.
Each specimen sheet was automatically cropped into multiple **224 × 224 pixel leaf patches**, which were then used as input for the deep learning models.

---

# Image Preprocessing

Before training the models, herbarium sheet images are processed using an automatic cropping script.
The script detects leaf areas based on color features (HSV and LAB color space), grayscale intensity, texture (Laplacian), and edges (Sobel and Canny).
Detected leaf regions are then extracted using a **224 × 224 sliding window**, producing multiple cropped leaf images per specimen.

Script used for cropping:

```
get_cropped_images.py
```

This script generates the dataset used for training the models.

---

# Repository Structure

The repository is organized in three experimental stages.

```
.
├── get_cropped_images.py
│
├── 1. initial experiment
│   ├── effnetv2s_initial_experiment.py
│   ├── mobnetv3l_initial_experiment.py
│   └── resnet50_initial_experiment.py
│
├── 2. hyperparameter tuning
│   ├── effnetv2s_hyperparameter_tuning_cv.py
│   ├── mobnetv3l_hyperparameter_tuning_cv.py
│   └── resnet50_hyperparameter_tuning_cv.py
│
└── 3. final training and evaluation
    ├── effnetv2s_final_training.py
    ├── mobnetv3l_final_training.py
    └── resnet50_final_training.py
```

---

# Experimental Workflow

The experiments follow three main stages.

## 1. Initial Experiment

The first stage compares different **imbalance handling strategies** and **epochs**.

Scripts:

- `effnetv2s_initial_experiment.py`
- `mobnetv3l_initial_experiment.py`
- `resnet50_initial_experiment.py`

Imbalance handling strategies:

- baseline (cross-entropy loss)
- weighted loss
- focal loss
- focal + weighted loss
- balanced sampling
- all combinations

Training epochs: 5, 10, 15, and 20.

The goal is to identify which strategy and epoch performs best for each architecture.

---

## 2. Hyperparameter Tuning

After selecting the best imbalance strategy and epoch, the next step performs **hyperparameter tuning using cross-validation**.

Scripts:

- `effnetv2s_hyperparameter_tuning_cv.py`
- `mobnetv3l_hyperparameter_tuning_cv.py`
- `resnet50_hyperparameter_tuning_cv.py`

This stage tunes parameters such as:

- learning rate
- batch size
- weight decay
- dropout rate
- optimizer
- label smoothing
- gradient clipping
- warm-up epochs
- learning rate scheduler

---

## 3. Final Training and Evaluation

In the final stage, models are trained using the **best hyperparameters and imbalance strategy** identified in previous experiments.

Scripts:

- `effnetv2s_final_training.py`
- `mobnetv3l_final_training.py`
- `resnet50_final_training.py`

These scripts perform: final model training, evaluation on the test set, and generation of classification metrics.
Evaluation metrics include accuracy, precision, recall, F1-score, Macro-F1, Weighted F1, Minority F1, and Cohen's Kappa.

---

# Requirements

Typical Python dependencies include:

```
python >= 3.9
pytorch
torchvision
numpy
opencv-python
scikit-learn
matplotlib
pandas
```

You can install them using:

```
pip install -r requirements.txt
```

---

# License

This project is intended for academic and research purposes.
