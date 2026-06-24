from flask import Flask, jsonify
from flask_cors import CORS
import json
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

TARGET_FILE = Path(__file__).resolve().parent.parent / "last_target.json"

@app.route('/target', methods=['GET'])
def get_target():
    if not os.path.exists(TARGET_FILE):
        return jsonify({
            "status": "Stand-By",
            "message": "No target detected yet"
        })
    
    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Failed to read data: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting on port 5000")
    app.run(host='0.0.0.0', port=5000)
