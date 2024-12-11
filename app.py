import os
import numpy as np
import librosa
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2


app = Flask(__name__)

MODEL_PATH = "Deep-Learning/FlaskApp/model_balanced.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_labels = ['Asthma', 'BRON', 'COPD', 'Heart Failure', 'Lung Fibrosis', 'N', 'Plueral Effusion', 'pneumonia']

def preprocess_audio(file_path, sr=22050, n_mels=224, fmax=8000, n_fft=2048, desired_width=224):
    y, sr = librosa.load(file_path, sr=sr)

    total_samples = len(y)
    hop_length = total_samples // desired_width

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, n_fft=n_fft, hop_length=hop_length)

    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    mel_spectrogram_resized = cv2.resize(mel_spectrogram_db, (desired_width, n_mels))

    mel_spectrogram_normalized = (mel_spectrogram_resized - np.min(mel_spectrogram_resized)) / (
        np.max(mel_spectrogram_resized) - np.min(mel_spectrogram_resized)
    )

    #final soape is (224, 224,3)
    mel_spectrogram_rgb = np.stack([mel_spectrogram_normalized] * 3, axis=-1)  

    return mel_spectrogram_rgb

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith(".wav"):
        try:
            temp_path = "temp.wav"
            file.save(temp_path)

            mel_spectrogram = preprocess_audio(temp_path)
            # dimension expansion of input data
            input_data = np.expand_dims(mel_spectrogram, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])

            predicted_class = class_labels[np.argmax(predictions)]
            confidence = float(np.max(predictions))

            os.remove(temp_path)
            
            return jsonify({"prediction": predicted_class, "confidence": float(np.max(predictions))})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Unsupported file format. Please upload a .wav file."}), 400

if __name__ == "__main__":
    app.run(debug=True)
