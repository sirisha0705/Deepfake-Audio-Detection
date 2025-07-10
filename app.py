from flask import Flask, request, render_template, jsonify, send_from_directory

app = Flask(__name__, static_folder='static', template_folder='templates')

from flask_cors import CORS
import os
import tempfile
import pickle
import numpy as np
import librosa
import tensorflow as tf
from werkzeug.utils import secure_filename
import shutil  # For cleaning up temp dir

app = Flask(__name__)
CORS(app)

# Paths to model and scalers
MODEL_PATH = 'models/model.keras'
SCALER_PATH = 'models/scaler.pkl'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'  # ✅ Fixed path

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_and_preprocessors():
    try:
        print("[INFO] Loading model and preprocessors...")
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = pickle.load(open(SCALER_PATH, 'rb'))
        label_encoder = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))
        print("[INFO] Model and preprocessors loaded.")
        return model, scaler, label_encoder
    except Exception as e:
        print(f"[ERROR] Failed to load model or preprocessors: {e}")
        return None, None, None

def extract_features(audio_path, n_mfcc=26):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
        if mfccs.shape[0] < 5:
            print("[WARN] Not enough frames in audio for analysis.")
            return None
        return mfccs
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None

def classify_audio(features, model, scaler, label_encoder):
    try:
        if model is None or scaler is None or label_encoder is None:
            return {"label": "UNKNOWN", "probability": 0.0, "explanation": "Model or preprocessors not loaded."}

        scaled = scaler.transform(features)
        window_size = 5
        n_windows = scaled.shape[0] - window_size + 1

        if n_windows < 1:
            return {"label": "UNKNOWN", "probability": 0.0, "explanation": "Audio too short to analyze."}

        windows = np.array([scaled[i:i+window_size] for i in range(n_windows)])
        predictions = model.predict(windows, verbose=0).flatten()
        avg_prob = predictions.mean()
        print(f"[INFO] Prediction average probability: {avg_prob:.4f}")

        label_index = int(avg_prob >= 0.5)
        label = label_encoder.inverse_transform([label_index])[0]
        explanation = "AI-generated speech patterns." if label == "FAKE" else "Natural vocal characteristics."

        return {
            "label": label,
            "Accuracy": float(avg_prob),
            "explanation": explanation
        }
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        return {"label": "ERROR", "probability": 0.0, "explanation": str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Upload MP3, WAV, or OGG."}), 400

    temp_dir = tempfile.mkdtemp()
    try:
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_path)
        print(f"[INFO] File saved to {temp_path}")

        features = extract_features(temp_path)
        if features is None:
            return jsonify({"error": "Could not extract audio features"}), 400

        model, scaler, label_encoder = load_model_and_preprocessors()
        result = classify_audio(features, model, scaler, label_encoder)

        return jsonify(result), 200
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            shutil.rmtree(temp_dir)  # Cleans up entire temp folder
            print(f"[INFO] Cleaned up {temp_dir}")
        except Exception as e:
            print(f"[WARN] Could not clean up temp directory: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "API is running"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
