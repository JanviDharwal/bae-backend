from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2, base64, numpy as np, tensorflow as tf, requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pymongo import MongoClient
import cloudinary, cloudinary.uploader
from bson import ObjectId
import random
from datetime import datetime
from rembg import remove
from PIL import Image
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
favourites_collection = db['favourites']   # <- NEW: favourites collection

# =======================================
# MOOD MODEL
# =======================================
MOOD_MODEL_PATH = "models/mood_model/mobilenetv2_mood3class.tflite"
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
OUTFIT_MODEL_PATH = "models/outfit_model/mobilenetv2_top_bottom.tflite"

try:
    outfit_model = tf.keras.layers.TFSMLayer(OUTFIT_MODEL_PATH, call_endpoint='serving_default')
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

    return jsonify({
        'success': True,
        'full_name': user['username'],
        'email': user['email']
    })


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

    result = users_collection.update_one(
        {"email": email},
        {"$set": {"username": username}}
    )

    if result.matched_count == 0:
        return jsonify({'success': False, 'message': 'User not found'}), 404

    return jsonify({'success': True, 'username': username})


# =======================================
# MOOD DETECTION
# =======================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))  # model input size

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
# CLOUDINARY UPLOAD WITH AUTO BG REMOVAL
# =======================================
from rembg import remove
from PIL import Image
import io

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ["png", "jpg", "jpeg", "webp"]:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        # Open image and remove background
        img = Image.open(file).convert("RGBA")
        img_no_bg = remove(img)

        # Save to buffer
        buf = io.BytesIO()
        img_no_bg.save(buf, format="PNG")
        buf.seek(0)

        # Upload to Cloudinary
        upload = cloudinary.uploader.upload(
            buf,
            folder="wardrobe_items",
            public_id=file.filename.rsplit(".", 1)[0],  # optional: keeps original name
            overwrite=True,
            resource_type="image"
        )

        return jsonify({
            "message": "Uploaded and background removed",
            "url": upload["secure_url"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# =======================================
# ADD WARDROBE ITEM
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

        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ["png", "jpg", "jpeg", "webp"]:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        # =========================
        # Remove Background
        # =========================
        img = Image.open(file).convert("RGBA")
        img_no_bg = remove(img)

        buf = io.BytesIO()
        img_no_bg.save(buf, format="PNG")
        buf.seek(0)

        # =========================
        # Upload to Cloudinary
        # =========================
        upload = cloudinary.uploader.upload(
            buf,
            folder="wardrobe_items",
            public_id=file.filename.rsplit(".", 1)[0],
            overwrite=True,
            resource_type="image"
        )
        image_url = upload["secure_url"]

        # =========================
        # Outfit Prediction
        # =========================
        # Convert to OpenCV format
        img_arr = np.array(img_no_bg)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
        x = preprocess_for_outfit(img_arr)
        output = outfit_model(x)
        pred = list(output.values())[0].numpy()
        predicted_class = "Topwear" if pred[0][0] < 0.5 else "Bottomwear"

        # =========================
        # Save to Wardrobe
        # =========================
        wardrobe_collection.insert_one({
            "userId": user_id,
            "imageUrl": image_url,
            "category": predicted_class,
            "deleted": False
        })

        return jsonify({
            "message": "Wardrobe item added with background removed",
            "imageUrl": image_url,
            "predicted_category": predicted_class
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =======================================
# GET WARDROBE
# =======================================
@app.route('/wardrobe/all', methods=['GET'])
def get_wardrobe():
    user_id = request.args.get("userId")
    items = list(wardrobe_collection.find({"userId": user_id}))

    for i in items:
        i["id"] = str(i["_id"])
        del i["_id"]

    return jsonify({"items": items})


# =======================================
# SOFT DELETE
# =======================================
@app.route('/wardrobe/delete', methods=['POST'])
def delete_wardrobe_item():
    data = request.get_json()
    user_id = data.get("userId")
    item_id = data.get("itemId")

    result = wardrobe_collection.update_one(
        {"userId": user_id, "_id": ObjectId(item_id)},
        {"$set": {"deleted": True}}
    )

    if result.matched_count == 0:
        return jsonify({"error": "Item not found"}), 404

    return jsonify({"message": "Item soft deleted"})


# =======================================
# RESTORE ITEM
# =======================================
@app.route('/wardrobe/restore', methods=['POST'])
def restore_wardrobe_item():
    data = request.get_json()
    user_id = data.get("userId")
    item_id = data.get("itemId")

    result = wardrobe_collection.update_one(
        {"userId": user_id, "_id": ObjectId(item_id)},
        {"$set": {"deleted": False}}
    )

    if result.matched_count == 0:
        return jsonify({"error": "Item not found"}), 404

    return jsonify({"message": "Item restored"})

# =======================================
# PERMANENT DELETE WARDROBE ITEM
# =======================================
@app.route('/wardrobe/deletePermanent', methods=['POST'])
def delete_wardrobe_item_permanent():
    data = request.get_json()
    user_id = data.get("userId")
    item_id = data.get("itemId")

    if not all([user_id, item_id]):
        return jsonify({"error": "Missing fields"}), 400

    result = wardrobe_collection.delete_one({
        "userId": user_id,
        "_id": ObjectId(item_id)
    })

    if result.deleted_count == 0:
        return jsonify({"error": "Item not found"}), 404

    return jsonify({"message": "Item permanently deleted"})
# =======================================
# OUTFIT RECOMMENDATION (FIXED)
# =======================================
def recommend_outfit(wardrobe_items):

    def normalize(cat):
        return cat.lower().replace(" ", "")

    tops = [
        item for item in wardrobe_items
        if normalize(item.get("category", "")) in ["topwear", "top", "shirt", "tshirt", "upper"]
    ]

    bottoms = [
        item for item in wardrobe_items
        if normalize(item.get("category", "")) in ["bottomwear", "bottom", "pant", "pants", "jeans", "trouser"]
    ]

    if not tops and not bottoms:
        return None

    if not tops:
        return {"topwear": None, "bottomwear": random.choice(bottoms)}

    if not bottoms:
        return {"topwear": random.choice(tops), "bottomwear": None}

    return {
        "topwear": random.choice(tops),
        "bottomwear": random.choice(bottoms)
    }


# =======================================
# FINAL OUTFIT GENERATOR ROUTE
# =======================================
@app.route('/generator/next', methods=['GET'])
def next_outfit():
    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "User ID required"}), 400

    wardrobe_items = list(wardrobe_collection.find({
        "userId": user_id,
        "deleted": False
    }))

    if not wardrobe_items:
        return jsonify({"topwear": None, "bottomwear": None})

    outfit_pair = recommend_outfit(wardrobe_items)

    if not outfit_pair:
        return jsonify({"topwear": None, "bottomwear": None})

    top_url = outfit_pair["topwear"]["imageUrl"] if outfit_pair["topwear"] else None
    bottom_url = outfit_pair["bottomwear"]["imageUrl"] if outfit_pair["bottomwear"] else None

    return jsonify({
        "topwear": top_url,
        "bottomwear": bottom_url
    })


# =======================================
# FAVOURITES: SAVE + GET ROUTES
# =======================================
@app.route('/saveFavourite', methods=['POST'])
def save_favourite():
    try:
        data = request.get_json()
        user_id = data.get("userId")
        topwear = data.get("topwear")   # expected string URL
        bottomwear = data.get("bottomwear")  # expected string URL

        if not all([user_id, topwear, bottomwear]):
            return jsonify({"error": "Missing fields"}), 400

        fav_doc = {
            "userId": user_id,
            "topwear": topwear,
            "bottomwear": bottomwear,
            "createdAt": datetime.utcnow()
        }

        favourites_collection.insert_one(fav_doc)

        return jsonify({"success": True, "message": "Favourite saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/getFavourites', methods=['GET'])
def get_favourites():
    try:
        user_id = request.args.get("userId")
        if not user_id:
            return jsonify({"error": "userId required"}), 400

        items = list(favourites_collection.find({"userId": user_id}))
        results = []
        for it in items:
            results.append({
                "id": str(it.get("_id")),
                "topwear": it.get("topwear"),
                "bottomwear": it.get("bottomwear"),
                "createdAt": it.get("createdAt")
            })

        return jsonify({"favourites": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    try:
        if "images" not in request.files:
            return jsonify({"error": "No files uploaded. Use key 'images'"}), 400

        files = request.files.getlist("images")
        if len(files) == 0:
            return jsonify({"error": "No image found"}), 400

        processed = []

        for file in files:
            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext not in ["png", "jpg", "jpeg", "webp"]:
                return jsonify({"error": f"Unsupported file type: {ext}"}), 400

            # Open with PIL
            img = Image.open(file).convert("RGBA")
            # Remove background
            output = remove(img)
            # Save to bytes buffer
            buf = io.BytesIO()
            output.save(buf, format="PNG")
            buf.seek(0)

            new_filename = file.filename.rsplit(".", 1)[0] + "_nobg.png"
            processed.append((new_filename, buf))

        # Single image → return directly
        if len(processed) == 1:
            filename, buffer = processed[0]
            return send_file(
                buffer,
                mimetype="image/png",
                as_attachment=True,
                download_name=filename
            )

        # Multiple images → return ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname, buff in processed:
                zipf.writestr(fname, buff.getvalue())
        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name="background_removed_images.zip"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# =======================================
# RUN SERVER
# =======================================
if __name__ == "__main__":
    app.run(port=5000, debug=True)
