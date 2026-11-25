
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
# -------------------------------
# 1. Project configuration
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_fer_improved.keras"

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# -------------------------------
# 2. Load trained model
# -------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

model = load_model()

# -------------------------------
# 3. Preprocess uploaded image
# -------------------------------


def preprocess_image(image):

    # Convert PIL image to RGB
    image = image.convert("RGB")

    # Convert to NumPy array
    image_array = np.array(image)

    # Convert RGB to grayscale
    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Load Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        raise ValueError(
            "No face detected. Please upload a clear, front-facing image."
        )

    # Select the largest detected face
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )
    # Add padding around detected face
    padding = int(0.15 * max(w, h))
    
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(gray.shape[1], x + w + padding)
    y2 = min(gray.shape[0], y + h + padding)
 
        # Crop the face
    face = gray[y:y+h, x:x+w]

    # Resize to FER-2013 dimensions
    face = cv2.resize(face, (48, 48))
    st.image(
    face,
    caption="Detected and cropped face (48 × 48)",
    width=150
)
    # Normalize
    face = face.astype("float32") / 255.0

    # Add batch and channel dimensions
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    return face

# -------------------------------
# 4. Streamlit interface
# -------------------------------

st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="centered"
)

st.title("Facial Expression Recognition")
st.write(
    "Upload a facial image to predict its emotional expression "
    "using our trained CNN model."
)

st.info(
    "For best results, upload a clear, front-facing, "
    "closely cropped face image."
)

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, width=300)

    if st.button("Predict Emotion"):

        processed_image = preprocess_image(image)

        # Model prediction
        predictions = model.predict(
            processed_image,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(predictions))
        predicted_emotion = CLASS_NAMES[predicted_index]

        st.success(
            f"Predicted Emotion: {predicted_emotion}"
        )

        st.subheader("Class Scores")

        for name, score in zip(CLASS_NAMES, predictions):

            st.write(f"**{name}: {score * 100:.2f}%**")

            st.progress(float(score))
