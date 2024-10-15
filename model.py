import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes):
    # Load a pre-trained EfficientNet-B0 model
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)   
    
    return model

def load_model(model_path, num_classes):
    # Create the model structure
    model_ft  = create_model(1).to('cpu')
    return model_ft

