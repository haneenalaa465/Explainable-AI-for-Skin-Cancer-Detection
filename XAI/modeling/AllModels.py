import torch
# Deep Learning Models
from XAI.modeling.models.Custom_CNN import CustomCNN
from XAI.modeling.models.D2CNN import D2CNN
from XAI.modeling.models.Modified_inception import ModifiedInceptionV3
from XAI.modeling.models.SkinResNet50 import FineTunedResNet50
from XAI.modeling.models.SkinLesionCNN import SkinLesionCNN
from XAI.modeling.models.custom_cnn_2 import SkinCancerCNN
from XAI.modeling.models.SkinEfficientNetB5 import SkinEfficientNetB5
from XAI.modeling.models.MobileNet import MobileNetV2

# Machine Learning Models
from XAI.modeling.models.DecisionTreeModel import DTModel
from XAI.modeling.models.RandomForestModel import RFModel

# Set up device for DL models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Deep Learning models list
dl_models = [
    SkinLesionCNN,
    D2CNN,
    CustomCNN,
    ModifiedInceptionV3,
    FineTunedResNet50,
    SkinCancerCNN,
    SkinEfficientNetB5,
    MobileNetV2
]

# Machine Learning models list with default hyperparameters
ml_models = [
    DTModel,  # Default parameters
    RFModel(n_estimators=200, random_state=0)  # With specified parameters
]

print("Models Initialized")
