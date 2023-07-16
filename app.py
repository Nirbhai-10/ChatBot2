import os
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from flask import Flask, request, jsonify, render_template

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Directory to store uploaded text files
# UPLOAD_FOLDER = "uploads"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

UPLOAD_FOLDER = r"C:\Users\priya\OneDrive\Desktop\Nirbhai\UPLOAD FOLDER"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload')
def upload():
    return render_template("files_upload/index_File.html")

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Remove punctuation and special characters
    tokens = [token for token in tokens if token.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Reconstruct the preprocessed text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def convert_text_to_json(text_file, output_file):
    intents = []

    with open(text_file, 'r', encoding='utf-8') as file:
        paragraphs = file.read().split("\n\n")

        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)

            intent = {
                "tag": preprocess_text(sentences[0].strip()),
                "patterns": [],
                "responses": [],
                "context_set": ""
            }

            for sentence in sentences[1:]:
                sentence = sentence.strip()
                tagged_words = pos_tag(word_tokenize(sentence))
                named_entities = ne_chunk(tagged_words)

                pattern = ""
                for chunk in named_entities:
                    if hasattr(chunk, 'label') and chunk.label():
                        pattern += chunk.label() + " "
                    else:
                        pattern += chunk[0] + " "

                intent["patterns"].append(preprocess_text(pattern.strip()))
                intent["responses"].append(preprocess_text(sentence))

            intents.append(intent)

    data = {
        "intents": intents
    }

    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print("Conversion completed successfully!")


# Endpoint to handle file uploads and intent conversion 
@app.route("/upload_intent", methods=["POST"])
def upload_file():
    # Check if the POST request has the file part
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # If the user does not select a file, browser may submit an empty part without filename
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the file to the upload folder
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Perform intent conversion and create intents.json file
    output_file = os.path.splitext(file_path)[0] + ".json"
    convert_text_to_json(file_path, output_file)

    return jsonify({"message": "File uploaded and converted successfully"})

if __name__ == "__main__":
    app.run(debug=True)
