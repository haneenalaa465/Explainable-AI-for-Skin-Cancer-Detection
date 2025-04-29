import torch
from XAI.modeling.models.Custom_CNN import CustomCNN
from XAI.modeling.models.D2CNN import D2CNN
from XAI.modeling.models.Modified_inception import ModifiedInceptionV3
from XAI.modeling.models.SkinResNet50 import FineTunedResNet50
from XAI.modeling.models.SkinLesionCNN import SkinLesionCNN
from XAI.modeling.models.custom_cnn_2 import SkinCancerCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models = [
    SkinLesionCNN,
    D2CNN,
    CustomCNN,
    ModifiedInceptionV3,
    FineTunedResNet50,
    SkinCancerCNN
]
print("Models Initialized")
