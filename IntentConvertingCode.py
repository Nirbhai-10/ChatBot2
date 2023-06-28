import os
import json
import nltk
from nltk.tokenize import sent_tokenize

def convert_text_to_json(text_file, output_file):
    intents = []

    with open(text_file, 'r', encoding='utf-8') as file:
        paragraphs = file.read().split("\n\n")

        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)

            intent = {
                "tag": sentences[0].strip(),
                "patterns": [],
                "responses": [],
                "context_set": ""
            }

            for sentence in sentences[1:]:
                sentence = sentence.strip()
                if sentence.endswith("?"):
                    intent["patterns"].append(sentence)
                else:
                    intent["responses"].append(sentence)

            intents.append(intent)

    data = {
        "intents": intents
    }

    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print("Conversion completed successfully!")

# Example usage
text_file = r"C:\Users\priya\OneDrive\Desktop\Nirbhai\The Joint Entrance Examination JEE.txt"
output_directory = r"C:\Users\priya\OneDrive\Desktop\Nirbhai\intents"
os.makedirs(output_directory, exist_ok=True)
output_file = os.path.join(output_directory, "intents.json")

convert_text_to_json(text_file, output_file)
