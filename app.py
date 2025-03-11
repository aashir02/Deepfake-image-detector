
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from resnet_model import CustomResNet18  # Import your trained model

# Function to preprocess the uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

def main():
    # Set page title and configure layout
    st.set_page_config(page_title="Real vs. Fake Image Detector", layout="wide")

    # Custom CSS to enhance appearance
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #f0f2f6;
            }
            .st-bw {
                color: #4CAF50;
            }
            .st-bw:hover {
                background-color: #4CAF50;
            }
            .stTextInput>div>div>div>input {
                border-radius: 0.25rem;
            }
            .stButton>button {
                color: white;
                background-color: #4CAF50;
                padding: 0.5rem 1rem;
                border: none;
                border-radius: 0.25rem;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .heading-wrapper {
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and page layout
    st.markdown("<h1 class='heading-wrapper'>Real vs. Fake Image Detector</h1>", unsafe_allow_html=True)
    st.write("Upload an image to detect if it's real or fake.")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_container_width=True)

        # Preprocess the uploaded image
        image = Image.open(uploaded_image)
        image = image.convert("RGB")
        processed_image = preprocess_image(image)

        # Load the trained model
        model = CustomResNet18(num_classes=1)
        model.load_state_dict(torch.load(r"C:\Users\ashir\OneDrive\Pictures\Deepfake_image_detector\model.pth", map_location=torch.device('cpu')))
        model.eval()

        # Make prediction
        with torch.no_grad():
            outputs = model(processed_image)
            prediction = torch.sigmoid(outputs).item()
        
        # Determine prediction label and confidence score
        if prediction >= 0.5:
            prediction_label = "Fake"
            confidence_score = prediction * 100
        else:
            prediction_label = "Real"
            confidence_score = (1 - prediction) * 100
        
        # Display prediction result with confidence score
        st.write(f"Prediction: **{prediction_label}** (Confidence: {confidence_score:.0f}%)")

        # Smaller preview of the image after prediction
        st.subheader("Processed Image Preview")
        st.image(image, caption='Processed Image', use_container_width=True, width=300)

if __name__ == "__main__":
    main()
