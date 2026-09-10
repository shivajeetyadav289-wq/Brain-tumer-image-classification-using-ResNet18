import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.model import load_model


CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


transform = transforms.Compose([
    transforms.Lambda(
        lambda img: img.convert("RGB")
    ),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(
    image_path,
    model,
    device
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = output.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    return (
        CLASS_NAMES[predicted_class],
        confidence,
        probabilities[0].cpu().numpy()
    )


def generate_gradcam(
    image_path,
    model,
    device
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_resized = image.resize(
        (224, 224)
    )

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)


    # Determine predicted class

    with torch.no_grad():

        output = model(
            image_tensor
        )

        predicted_class = output.argmax(
            dim=1
        ).item()


    # Last convolutional layer

    target_layers = [
        model.layer4[-1]
    ]


    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )


    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]


    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets
    )[0]


    rgb_image = (
        np.array(image_resized)
        .astype(np.float32)
        / 255.0
    )


    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    return visualization