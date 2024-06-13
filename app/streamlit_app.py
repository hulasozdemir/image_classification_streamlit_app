import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('best_model.keras')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title("Image Classification with CNN")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    img = image.resize((32, 32))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    score = np.argmax(predictions)
    st.write(f'This image is most likely a {class_names[score]} with a {100 * np.max(predictions):.2f}% confidence.')
