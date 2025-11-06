from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, base64, numpy as np, tensorflow as tf, os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/mood_model/mobilenetv2_mood_3class.h5")
MOOD_LABELS = ['happy', 'neutral', 'sad']

# === Load Model ===
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(" Model loaded successfully from:", MODEL_PATH)

# === Routes ===
@app.route('/')
def home():
    return jsonify({'message': 'BAE Mood Detection API is live!'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data['image']

        # Decode base64 image
        img_bytes = base64.b64decode(img_data.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess for MobileNetV2
        img = cv2.resize(img, (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict mood
        preds = model.predict(x)
        mood = MOOD_LABELS[np.argmax(preds)]
        confidence = float(np.max(preds))

        print(f"Prediction: {mood} ({confidence*100:.2f}%)")
        return jsonify({'mood': mood, 'confidence': f"{confidence*100:.2f}%"})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)