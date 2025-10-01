from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)

# Function to fetch image path from database
def get_image(word):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM words WHERE word = ?", (word.lower(),))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# API endpoint
@app.route("/get-image", methods=["POST"])
def get_image_api():
    data = request.get_json()
    word = data.get("word", "")
    image_path = get_image(word)
    if image_path:
        return jsonify({"word": word, "image_url": image_path})
    else:
        return jsonify({"error": "Image not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
