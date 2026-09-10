import streamlit as st
import tempfile
from pathlib import Path

import torch

from src.model import load_model
from src.inference import (
    predict_image,
    generate_gradcam,
    CLASS_NAMES
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# Constants
# ============================================================

MODEL_PATH = (
    "models/brain_tumor_resnet18_final.pth"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load model only once
# ============================================================

@st.cache_resource
def get_model():

    model = load_model(
        MODEL_PATH,
        device
    )

    return model


model = get_model()


# ============================================================
# Header
# ============================================================

st.title(
    "🧠 Brain Tumor Image Classifier"
)

st.markdown(
    """
    ### ResNet18 Deep Learning Model

    Upload a brain image to classify it into one of
    four categories:

    **Glioma · Meningioma · No Tumor · Pituitary**
    """
)


st.info(
    "This application is intended for educational "
    "and research purposes only. It is not a clinical "
    "diagnostic tool."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("About the Model")

    st.write(
        """
        **Architecture:** ResNet18

        **Transfer Learning:** Yes

        **Input Size:** 224 × 224

        **Number of Classes:** 4

        **Test Accuracy:** 95.19%
        """
    )

    st.divider()

    st.header("Classes")

    for class_name in CLASS_NAMES:

        st.write(
            f"• {class_name.capitalize()}"
        )

    st.divider()

    st.write(
        f"Running on: `{device}`"
    )


# ============================================================
# Upload image
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a brain image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# Main application
# ============================================================

if uploaded_file is not None:

    st.divider()

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            uploaded_file,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "Model Prediction"
        )

        if st.button(
            "🔍 Analyze Image",
            use_container_width=True
        ):

            try:

                # Create temporary file

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(
                        uploaded_file.name
                    ).suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name


                # --------------------------------------------
                # Prediction
                # --------------------------------------------

                (
                    predicted_class,
                    confidence,
                    probabilities
                ) = predict_image(
                    temp_path,
                    model,
                    device
                )


                # --------------------------------------------
                # Prediction result
                # --------------------------------------------

                st.success(
                    f"Prediction: "
                    f"{predicted_class.capitalize()}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )


                # --------------------------------------------
                # Probability table
                # --------------------------------------------

                st.subheader(
                    "Class Probabilities"
                )


                probability_data = {

                    CLASS_NAMES[i].capitalize():
                    f"{probabilities[i] * 100:.2f}%"

                    for i in range(
                        len(CLASS_NAMES)
                    )
                }


                st.table(
                    probability_data
                )


                # --------------------------------------------
                # Grad-CAM
                # --------------------------------------------

                st.subheader(
                    "Grad-CAM"
                )

                st.caption(
                    "Highlighted regions indicate areas "
                    "that contributed to the model's prediction."
                )


                gradcam_image = generate_gradcam(
                    temp_path,
                    model,
                    device
                )


                st.image(
                    gradcam_image,
                    caption="Model attention visualization",
                    use_container_width=True
                )


            except Exception as e:

                st.error(
                    "An error occurred while "
                    "processing the image."
                )

                st.exception(e)