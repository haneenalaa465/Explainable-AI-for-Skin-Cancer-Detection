# Skin Lesion Classification with Explainable AI

This project leverages machine learning and deep learning techniques for automated skin cancer classification using the HAM10000 dataset. It emphasizes model interpretability through Explainable AI (XAI) methods such as Grad-CAM, SHAP, and LIME, addressing the critical research gap in transparency and trustworthiness of AI-driven medical diagnostics.

## Project Overview

Skin cancer is one of the deadliest and fastest-spreading cancers in the world, and early detection is crucial for effective treatment. This project builds an automated system for classifying skin lesions using deep learning, with a focus on making the model's decisions interpretable through explainable AI techniques.

The implemented CNN model can classify dermoscopic images into seven categories:
1. Actinic Keratoses (akiec)
2. Basal Cell Carcinoma (bcc)
3. Benign Keratosis-like Lesions (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic Nevi (nv)
7. Vascular Lesions (vasc)


## Dataset

This project uses the HAM10000 ("Human Against Machine with 10000 training images") dataset, which consists of 10,015 dermatoscopic images of pigmented skin lesions. The dataset was split into training, validation, and test sets with stratified sampling to maintain class distribution.


## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/haneenalaa465/Explainable-AI-for-Skin-Cancer-Detection.git
cd Explainable-AI-for-Skin-Cancer-Detection
```

2. Create and activate a virtual environment:
```bash
make create_environment
conda activate Explainable-AI-for-Skin-Cancer-Detection  # or source venv/bin/activate
```

3. Install dependencies:
```bash
make requirements
```

4. Download and prepare the dataset:
```bash
make data
```

### Training

To train the model:
```bash
make train
```

### Making Predictions with Explanations

To make predictions and get explanations:
```bash
make explain
```

## Explainability

This project implements two main explainable AI techniques:

1. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - Generates local explanations by perturbing input images
   - Highlights regions that support or contradict predictions

2. **SHAP (SHapley Additive exPlanations)**:
   - Based on cooperative game theory
   - Assigns importance values to each pixel
   - Shows how each part of the image contributes to predictions

These explanations make the model's decision-making process more transparent, which is crucial for medical applications.

## Project Structure

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- This README file
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data (organized by class)
│   ├── processed      <- Final, canonical data sets for modeling
│   └── raw            <- Original, immutable data dump (HAM10000)
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks for exploration and explanation
│
├── reports            <- Generated analysis as HTML, PDF, etc.
│   └── figures        <- Generated graphics and figures
│
├── requirements.txt   <- Package dependencies
│
└── XAI                <- Source code
    ├── __init__.py    <- Makes XAI a Python module
    ├── config.py      <- Configuration parameters
    ├── dataset.py     <- Data loading and processing utilities
    ├── features.py    <- Feature extraction scripts
    ├── modeling       <- Model-related code
    │   ├── model.py   <- CNN model definition
    │   ├── train.py   <- Training script
    │   └── predict.py <- Prediction and explanation script
    └── plots.py       <- Visualization utilities
```

## References

1. Shetty, B., Fernandes, R., Rodrigues, A.P. et al. (2022). Skin lesion classification of dermoscopic images using machine learning and convolutional neural network. *Scientific Reports*, 12, 18134. https://doi.org/10.1038/s41598-022-22644-9

2. Alzubaidi, L., Al-Amidie, M., Al-Asadi, A. et al. (2021). Novel Transfer Learning Approach for Medical Imaging with Limited Labeled Data. *Cancers*, 13(7), 1590. https://doi.org/10.3390/cancers13071590


## License

This project is licensed under the MIT License - see the LICENSE file for details.
