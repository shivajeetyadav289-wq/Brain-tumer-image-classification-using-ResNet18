import torch
import torch.nn as nn
from torchvision import models


NUM_CLASSES = 4


def create_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, NUM_CLASSES)

    return model


def load_model(model_path, device):

    model = create_model()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model