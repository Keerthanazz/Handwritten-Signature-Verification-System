import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load and preprocess images
def preprocess_image(image):
    img = image.convert("L")  # Convert to grayscale
    img = img.resize((128, 128))  # Resize to 128x128
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for CNN
    return img

# Build CNN Model
def build_signature_model():
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Output: Genuine or Forged

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the pre-trained model
loaded_model = build_signature_model()
loaded_model.load_weights('signature_verification_model.h5')  # Load saved model weights

# Webcam input processing for signature
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (128, 128))  # Resize to match the model input
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Return a processed frame (for display)

# Streamlit UI for Signature Verification
st.title("Handwritten Signature Verification System")

# Reference Signature Upload
st.subheader("Upload Reference Signature:")
uploaded_reference = st.file_uploader("Upload Reference Signature...", type="png")
if uploaded_reference is not None:
    reference_image = Image.open(uploaded_reference)
    st.image(reference_image, caption='Reference Signature', use_column_width=True)

# Webcam Input for Live Signature
st.subheader("Capture Live Signature:")
webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if st.button("Verify Signatures") and uploaded_reference is not None and webrtc_ctx.video_transformer:
    # Preprocess the reference signature
    preprocessed_reference = preprocess_image(reference_image)

    # Capture the live signature
    live_signature_frame = webrtc_ctx.video_transformer.frame
    if live_signature_frame is not None:
        live_signature_image = Image.fromarray(cv2.cvtColor(live_signature_frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB))
        preprocessed_live = preprocess_image(live_signature_image)

        # Expand dimensions to match model input
        preprocessed_reference = np.expand_dims(preprocessed_reference, axis=0)
        preprocessed_live = np.expand_dims(preprocessed_live, axis=0)

        # Compare the two signatures
        prediction_reference = loaded_model.predict(preprocessed_reference)[0][0]
        prediction_live = loaded_model.predict(preprocessed_live)[0][0]

        # Determine if the signatures match
        if prediction_reference < 0.5 and prediction_live < 0.5:
            st.success("Both signatures are genuine.")
        elif prediction_reference >= 0.5:
            st.error("Reference signature is forged.")
        elif prediction_live >= 0.5:
            st.error("Live signature is forged.")
        else:
            st.error("Signatures do not match.")
