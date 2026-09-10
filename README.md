# 🧠 Brain Tumor Image Classification Using ResNet18

A deep learning project for classifying brain images into four categories using
transfer learning and fine-tuning with ResNet18 in PyTorch.
<img src="https://github.com/shivajeetyadav289-wq/Brain-tumer-image-classification-using-ResNet18/blob/main/Image/Screenshot%202026-09-10%20074616.png" alt="App Screenshot" width="500">
## Project Overview

This project develops an image classification pipeline capable of predicting
one of four classes:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The project includes:

- Image preprocessing
- Exploratory data analysis
- CNN baseline
- ResNet18 transfer learning
- Fine-tuning
- Model evaluation
- Confusion matrix analysis
- Grad-CAM interpretability
- Streamlit inference application

## Model

The final model uses:

- Architecture: ResNet18
- Framework: PyTorch
- Transfer learning: ImageNet-pretrained weights
- Input size: 224 × 224
- Number of classes: 4
- Optimizer: Adam
- Loss function: Cross Entropy Loss

During fine-tuning, the later ResNet layers and classification head were
unfrozen to adapt the pretrained network to the brain-image dataset.

## Dataset

The dataset contains four balanced classes.

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|------|-------:|-----------:|---------:|----------:|------:|
| Training | 1400 | 1400 | 1400 | 1400 | 5600 |
| Testing | 400 | 400 | 400 | 400 | 1600 |

The original training set was split into:

- 80% training
- 20% validation

The independent testing set was kept separate for final evaluation.

The dataset itself is not included in this repository.

## Preprocessing

Images are:

1. Converted to RGB
2. Resized to 224 × 224
3. Converted to tensors
4. Normalized using ImageNet mean and standard deviation

Training images additionally use:

- Random horizontal flip
- Random rotation

## Results

The final fine-tuned ResNet18 achieved:

**Test Accuracy: 95.19%**

### Classification Metrics

| Class | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| Glioma | 98.81% | 83.25% | 90.37% |
| Meningioma | 90.76% | 98.25% | 94.36% |
| No Tumor | 93.68% | 100.00% | 96.74% |
| Pituitary | 98.51% | 99.25% | 98.88% |

Macro averages:

- Precision: 95.44%
- Recall: 95.19%
- F1-score: 95.08%

## Confusion Matrix

The main classification errors were associated with glioma.

The confusion matrix showed:

- 40 glioma images classified as meningioma
- 24 glioma images classified as no tumor
- 3 glioma images classified as pituitary

This indicates that glioma was the most challenging class for the final model.

## Model Interpretability

Grad-CAM was used to visualize image regions that contributed to the model's
prediction.

Grad-CAM should be interpreted as a model interpretability technique rather
than as a validated tumor localization or segmentation method.

## Streamlit Application

The project includes a Streamlit application that allows users to:

1. Upload a brain image
2. Obtain a predicted class
3. View prediction confidence
4. View class probabilities
5. View a Grad-CAM visualization

Run the application with:

```bash
python -m streamlit run app.py

conda create -n brain-tumor python=3.11
conda activate brain-tumor

pip install -r requirements.txt
```
## Poject structure
``` text
neural-net-brain-tumor/
├── data/
├── models/
├── notebooks/
├── results/
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── inference.py
├── app.py
├── test_inference.py
├── requirements.txt
├── .gitignore
└── README.md
```
## Limitations

This project is intended for educational and research purposes.

The model should not be used as a clinical diagnostic system.

Performance may vary with images from different datasets, scanners,
acquisition protocols, preprocessing pipelines, and patient populations.

The model was evaluated on a specific held-out test dataset and has not been
clinically validated.

## Future Improvements

Potential future work includes:

External dataset validation
Class imbalance handling for other datasets
More systematic hyperparameter optimization
Comparison with EfficientNet and other architectures
Improved explainability
Tumor segmentation
Multi-modal imaging analysis
Model deployment
Containerization with Docker
