from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import openai
import os

app = Flask(__name__)
CORS(app)

# ======= Set your OpenAI API Key =======
# Safer way: store key in environment variable
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
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

# ======= Voice-to-Text Route (Whisper) =======
@app.route('/voice-to-text', methods=['POST'])
def voice_to_text():
    """
    Accepts an audio file from the client and returns the transcribed text using OpenAI Whisper.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400

    audio_file = request.files['audio']

    # Send audio to Whisper API
    transcript = openai.audio.transcribe(
        model="whisper-1",
        file=audio_file
    )  # pylint: disable=no-member

    return jsonify({"text": transcript.text})

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
