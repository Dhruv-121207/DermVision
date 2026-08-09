import streamlit as st
from PIL import Image

from predict import predict_image
from gradcam import generate_gradcam

st.set_page_config(
    page_title="DermVision",
    layout="centered"
)

st.sidebar.title("Model information")
st.sidebar.markdown("**Architecture:** ResNet18")
st.sidebar.markdown("**Input size:** 224 × 224")
st.sidebar.markdown("**Validation Accuracy:** 90.08%")
st.sidebar.markdown("**Test Accuracy:** 88.02%")
st.sidebar.markdown("**Classes**")
st.sidebar.markdown("- Actinic keratoses (AKIEC)")
st.sidebar.markdown("- Basal cell carcinoma (BCC)")
st.sidebar.markdown("- Benign keratosis-like lesions (BKL)")
st.sidebar.markdown("- Dermatofibroma (DF)")
st.sidebar.markdown("- Melanoma (MEL)")
st.sidebar.markdown("- Melanocytic nevus (NV)")
st.sidebar.markdown("- Vascular lesions (VASC)")

st.title("DermVision")
st.subheader("AI-powered Skin Lesion Classification using ResNet18")

st.write(
    "Upload a dermoscopy image to predict the most likely skin lesion category. "
    "The model returns probability estimates for all supported classes and provides "
    "a Grad-CAM visualization highlighting image regions that influenced its prediction."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image",
        width=350
    )

    if st.button("Predict"):
        
        with st.spinner("Analyzing image..."):
            result = predict_image(image)

        st.success("Analysis complete")

        st.markdown("### Model prediction")

        st.markdown(
            f"""
            <div style="padding:20px;border-radius:10px;background-color:#1f2937;border:1px solid #374151;">
                <h3 style="margin:0;color:white;">{result['predicted_class']}</h3>
                <p style="margin-top:10px;font-size:18px;color:white;">
                    Confidence: <strong>{result['confidence']:.2f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.info(
            "A high confidence score does not guarantee that the prediction is correct. "
            "The model can still make incorrect predictions even at high confidence, "
            "particularly for visually similar or underrepresented lesion types."
        )

        st.markdown("### Top 3 predictions")

        top_predictions = list(result["probabilities"].items())[:3]

        for class_name, probability in top_predictions:
            st.write(f"**{class_name}** — {probability:.2f}%")

        if len(top_predictions) >= 2:
            difference = top_predictions[0][1] - top_predictions[1][1]
            if difference < 10:
                st.warning(
                    "The top two predictions are very close. The model is relatively uncertain "
                    "between these lesion categories."
                )

        st.markdown("### Class probabilities")

        items = list(result["probabilities"].items())

        left_col, right_col = st.columns(2)

        with left_col:
            for class_name, probability in items[:4]:
                st.write(f"**{class_name}**")
                st.progress(probability / 100)
                st.write(f"{probability:.2f}%")

        with right_col:
            for class_name, probability in items[4:]:
                st.write(f"**{class_name}**")
                st.progress(probability / 100)
                st.write(f"{probability:.2f}%")

        st.markdown("### Grad-CAM explanation")

        gradcam_image = generate_gradcam(image)

        st.image(
            gradcam_image,
            caption="Highlighted regions that influenced the model prediction",
            width=350
        )

        st.markdown("---")

        st.warning(
            "This application is developed for educational and research purposes only. "
            "It is not intended for medical diagnosis, treatment, or clinical decision-making. "
            "Always consult a qualified dermatologist for professional medical advice."
        )