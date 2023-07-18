from flask import Flask, request, jsonify, render_template
import os
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

import ChatBotV2            

UPLOAD_FOLDER = r"C:\Users\priya\OneDrive\Desktop\Nirbhai_PS\ChatBot2\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Directory for templates
TEMPLATES_FOLDER = r"C:\Users\priya\OneDrive\Desktop\Nirbhai_PS\ChatBot2\templates"

# Update the path for the template_folder argument in the Flask app instantiation

app = Flask(__name__, template_folder=TEMPLATES_FOLDER)

@app.route('/')
def home():
    return "Working at new route"

@app.route('/Chat')
def Chat():
    return render_template("chatbot.html")
@app.route('/upload')
def upload():
    return render_template("intent_conversion.html")


@app.route('/ChatApp', methods=['POST'])
def respond():
    data = request.get_json(force=True)  # Use force=True to parse even if Content-Type is not set to application/json
    message = data['message']
    response = ChatBotV2.classify_and_get_response(message)
    return jsonify({'response': response})

# Directory to store uploaded text files
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Remove punctuation and special characters
    tokens = [token for token in tokens if token.isalnum()]

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove single-character words
    tokens = [token for token in tokens if len(token) > 1]

    # Perform POS tagging and named entity recognition
    tagged_tokens = pos_tag(tokens)
    ner_tokens = ne_chunk(tagged_tokens)

    # Return the preprocessed tokens as a single string
    return " ".join(tokens)

def convert_text_to_json(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Create a dictionary with the intent name and text
    intent = {
        "name": "intent",
        "text": preprocessed_text
    }

    # Create a list of intents
    intents = [intent]

    # Create a dictionary with the intents
    data = {
        "intents": intents
    }

    # Convert the dictionary to JSON
    json_data = json.dumps(data, indent=4)

    return json_data


@app.route('/upload_intent', methods=['POST'])
def upload_intent():
    # Create the uploads folder if it doesn't exist
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])

    try:
        # Get the uploaded files
        files = request.files.getlist("file")

        # Convert the uploaded files to intents
        intent_files = []
        for index, file in enumerate(files):
            # Save the file
            filename = f"intent_{index + 1}.txt"
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            # Read the file contents
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            with open(file_path, "r") as f:
                text = f.read().strip()

            # Convert the text to JSON
            json_data = convert_text_to_json(text)

            # Save the JSON data to a file
            intent_json_filename = f"intent_{index + 1}.json"
            intent_json_path = os.path.join(app.config["UPLOAD_FOLDER"], intent_json_filename)
            with open(intent_json_path, "w") as f:
                f.write(json_data)

            intent_files.append(intent_json_filename)

        return jsonify({"intentFiles": intent_files})

    except Exception as e:
        error_message = "File upload failed. Please try again."
        print("Error during file upload:", e)
        return jsonify({"error": error_message}), 500


@app.route('/convert_text', methods=['POST'])
def convert_text():
    data = request.get_json(force=True)  # Use force=True to parse even if Content-Type is not set to application/json
    text = data['text']

    # Convert the text to JSON
    json_data = convert_text_to_json(text)

    # Save the JSON data to a file
    intent_json_filename = "intent.txt"
    intent_json_path = os.path.join(app.config["UPLOAD_FOLDER"], intent_json_filename)
    with open(intent_json_path, "w") as f:
        f.write(json_data)

    return jsonify({"intentFiles": [intent_json_filename]})

@app.route('/train_chatbot', methods=['POST'])
def train_chatbot():
    data = request.get_json(force=True)  # Use force=True to parse even if Content-Type is not set to application/json
    intent_files = data['intentFiles']

    # Train the chatbot using the selected intent files
    success = ChatBotV2.train_chatbot(intent_files)

    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
