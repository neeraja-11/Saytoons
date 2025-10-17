from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import numpy as np
import pyaudio
import time
import queue
import threading
import noisereduce as nr
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)

# ======= Speech-to-Text Configuration =======
MODEL_NAME = "base.en"   # or "small.en", "medium.en"
DEVICE = "cpu"          # or "cuda" if you have GPU
COMPUTE_TYPE = "int8"   # use "float16" if using GPU

CHUNK = 1024
RATE = 16000
CHANNELS = 1
ROLLING_BUFFER_SEC = 5.0
TRANSCRIBE_INTERVAL_SEC = 0.5

# Initialize queues and events
audio_queue = queue.Queue()
stop_event = threading.Event()

# Load Whisper model
print("🔄 Loading Whisper model...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print("✅ Whisper model loaded!")
# ======= Function to fetch image path from database =======
def get_image(word):
    """
    Fetch the image path from SQLite database based on the given word.
    """
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM words WHERE word = ?", (word.lower(),))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# ======= Homepage Routes =======
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    return render_template('index.html')

# ======= Audio Processing Functions =======
def audio_callback(in_data, frame_count, time_info, status):
    if status:
        print("[DEBUG] Audio status:", status)
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    audio_queue.put(audio_data)
    return (None, pyaudio.paContinue)

def preprocess_audio(y: np.ndarray):
    if len(y) == 0:
        return y
    y = y / max(1e-8, np.max(np.abs(y)))
    try:
        y = nr.reduce_noise(y=y, sr=RATE, prop_decrease=0.9)
    except Exception:
        pass
    return y

# ======= Voice-to-Text Route =======
@app.route('/voice-to-text', methods=['POST'])
def voice_to_text():
    """
    Handles live speech-to-text using faster_whisper model.
    """
    buffer = np.array([], dtype=np.float32)
    prev_text = ""
    
    try:
        audio_data = request.get_data()
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Preprocess audio
        audio_proc = preprocess_audio(audio_np)
        
        # Transcribe
        segments, _ = model.transcribe(
            audio_proc,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False
        )
        
        text = " ".join([seg.text.strip() for seg in segments]).strip()
        return jsonify({"text": text})
        
    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500

# ======= Text-to-Image Route =======
@app.route("/get-image", methods=["POST"])
def get_image_api():
    """
    Accepts a word from the client and returns the corresponding image URL from the database.
    """
    data = request.get_json()
    word = data.get("word", "")
    image_path = get_image(word)
    if image_path:
        return jsonify({"word": word, "image_url": image_path})
    else:
        return jsonify({"error": "Image not found"}), 404

# ======= Run the app =======
if __name__ == "__main__":
    app.run(debug=True)
