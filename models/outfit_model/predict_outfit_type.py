import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
 
MODEL_PATH = r"D:\BE 24-27\5th sem\SE\SE LAB\Project\BAE\BAE--Bringing-Aesthetics-to-Emotions\models\outfit_model\mobilenetv2_top_bottom_savedmodel"

# Load the SavedModel as a TFSMLayer (Keras 3+ compatible)
model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Example: run prediction on a test image
test_image_path = r"C:\Users\DELL\Downloads\topwear example male.png"
img_array = preprocess_image(test_image_path)

output = model(img_array)
pred = list(output.values())[0].numpy()

print("Raw Prediction:", pred)
predicted_class = "Topwear" if pred[0][0] < 0.5 else "Bottomwear"
print("Predicted Class:", predicted_class)