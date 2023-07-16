from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'intent_files'
ALLOWED_EXTENSIONS = {'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_intents_from_file(file_path):
    with open(file_path, 'r') as file:
        intents = json.load(file)
    return intents


def update_intents(intents, new_patterns, new_responses):
    # Update the intents data with new patterns and responses
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            intent['patterns'].extend(new_patterns)
            intent['responses'].extend(new_responses)


def save_intents_to_file(intents, file_path):
    with open(file_path, 'w') as file:
        json.dump(intents, file, indent=4)


@app.route('/upload', methods=['POST'])
def upload_intent_files():
    if 'intent_files' not in request.files:
        return jsonify({'message': 'No intent files provided.'}), 400

    uploaded_files = request.files.getlist('intent_files')

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return jsonify({'message': 'Intent files uploaded successfully.'}), 200


@app.route('/get_intent_files', methods=['GET'])
def get_intent_files():
    intent_files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify({'intent_files': intent_files}), 200


if __name__ == '__main__':
    app.run(debug=True)
