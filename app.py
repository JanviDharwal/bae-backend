from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, base64, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pymongo import MongoClient
import cloudinary, cloudinary.uploader
from bson import ObjectId
import random
from datetime import datetime
from PIL import Image
import io, os

# =======================================
# FLASK APP
# =======================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =======================================
# CLOUDINARY CONFIG
# =======================================
cloudinary.config(
    cloud_name="dggce9lgq",
    api_key="595392624381522",
    api_secret="HBAkBl7dzKlh-LDZHYs37K5D74c"
)

# =======================================
# MONGODB ATLAS
# =======================================
MONGO_URI = "mongodb+srv://baeUser:behencodes@cluster0.4ffhppa.mongodb.net/baeDB"
client = MongoClient(MONGO_URI)
db = client['baeDB']

users_collection = db['users']
wardrobe_collection = db['wardrobe']
favourites_collection = db['favourites']

# =======================================
# MOOD MODEL
# =======================================
MOOD_MODEL_PATH = "models/mood_model/mobilenetv2_mood_3class.h5"
MOOD_LABELS = ['happy', 'neutral', 'sad']

try:
    mood_model = tf.keras.models.load_model(MOOD_MODEL_PATH)
    print("Mood Model Loaded Successfully")
except Exception as e:
    print("Mood model load error:", e)
    mood_model = None

# =======================================
# OUTFIT MODEL
# =======================================
OUTFIT_MODEL_PATH = "models/outfit_model/mobilenetv2_top_bottom_savedmodel"

try:
    from tensorflow.keras.layers import TFSMLayer
    outfit_model = TFSMLayer(OUTFIT_MODEL_PATH, call_endpoint='serving_default')
    print("Outfit Model Loaded")
except Exception as e:
    print("Outfit model load error:", e)
    outfit_model = None

def preprocess_for_outfit(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# =======================================
# BASIC ROUTES
# =======================================
@app.route('/')
def home():
    return jsonify({"message": "BAE Backend Running"})

@app.route('/health')
def health():
    return jsonify({"status": "OK"})

# =======================================
# USER AUTH
# =======================================
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not all([username, email, password]):
        return jsonify({'success': False, 'message': 'All fields required'}), 400

    if not email.endswith("@thapar.edu"):
        return jsonify({'success': False, 'message': 'Only @thapar.edu allowed'}), 400

    if users_collection.find_one({'email': email}):
        return jsonify({'success': False, 'message': 'Email already registered'}), 400

    users_collection.insert_one({
        "username": username,
        "email": email,
        "password": password
    })

    return jsonify({'success': True, 'full_name': username, 'email': email})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = users_collection.find_one({"email": email, "password": password})

    if not user:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

    return jsonify({'success': True, 'full_name': user['username'], 'email': user['email']})

# =======================================
# PROFILE
# =======================================
@app.route('/get_profile', methods=['GET'])
def get_profile():
    email = request.args.get("email")
    user = users_collection.find_one({"email": email}, {"_id": 0, "password": 0})
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    return jsonify({'success': True, 'user': user})

@app.route('/update_profile', methods=['POST'])
def update_profile():
    data = request.get_json()
    email = data.get('email')
    username = data.get('username')

    result = users_collection.update_one({"email": email}, {"$set": {"username": username}})
    if result.matched_count == 0:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    return jsonify({'success': True, 'username': username})

# =======================================
# MOOD DETECTION
# =======================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data['image']

        img_bytes = base64.b64decode(img_data.split(',')[1])
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = mood_model.predict(x)
        mood = MOOD_LABELS[np.argmax(preds)]
        conf = float(np.max(preds))

        return jsonify({"mood": mood, "confidence": f"{conf*100:.2f}%"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =======================================
# CLOUDINARY UPLOAD (NO REMBG)
# =======================================
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]
        img = Image.open(file).convert("RGBA")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        upload = cloudinary.uploader.upload(
            buf,
            folder="wardrobe_items",
            public_id=file.filename.rsplit(".", 1)[0],
            overwrite=True,
            resource_type="image"
        )

        return jsonify({"message": "Uploaded successfully", "url": upload["secure_url"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =======================================
# WARDROBE ROUTES
# =======================================
@app.route('/wardrobe/add', methods=['POST'])
def add_wardrobe():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]
        user_id = request.form.get("userId")
        if not user_id:
            return jsonify({"error": "Missing userId"}), 400

        img = Image.open(file).convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        upload = cloudinary.uploader.upload(
            buf,
            folder="wardrobe_items",
            public_id=file.filename.rsplit(".", 1)[0],
            overwrite=True,
            resource_type="image"
        )
        image_url = upload["secure_url"]

        # Outfit Prediction
        img_arr = np.array(img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
        x = preprocess_for_outfit(img_arr)
        output = outfit_model(x)
        pred = list(output.values())[0].numpy()
        predicted_class = "Topwear" if pred[0][0] < 0.5 else "Bottomwear"

        wardrobe_collection.insert_one({
            "userId": user_id,
            "imageUrl": image_url,
            "category": predicted_class,
            "deleted": False
        })

        return jsonify({"message": "Wardrobe item added", "imageUrl": image_url, "predicted_category": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/wardrobe/all', methods=['GET'])
def get_wardrobe():
    user_id = request.args.get("userId")
    items = list(wardrobe_collection.find({"userId": user_id}))
    for i in items:
        i["id"] = str(i["_id"])
        del i["_id"]
    return jsonify({"items": items})

# =======================================
# RUN SERVER
# =======================================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
