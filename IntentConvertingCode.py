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

#The iterative refinement process has been added to continuously train, evaluate, and refine the intent categorization model based on user feedback. The process continues until the user chooses to exit.

#The predict_intent function and the get_user_feedback function are placeholders, and you need to implement them according to your chosen intent categorization approach.

#The get_user_input_data function is also a placeholder and needs to be implemented to gather new patterns and responses from the user for the incorrect intents.

#The load_intents_from_file, update_intents, and save_intents_to_file functions are placeholders and need to be implemented to load the intents data from the file, update the intents with new patterns and responses, and save the updated intents back to the file, respectively.



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

# Train the initial model
text_file = r"C:\Users\priya\OneDrive\Desktop\Nirbhai\The Joint Entrance Examination JEE.txt"
output_directory = r"C:\Users\priya\OneDrive\Desktop\Nirbhai\intents"
os.makedirs(output_directory, exist_ok=True)
output_file = os.path.join(output_directory, "intents.json")

convert_text_to_json(text_file, output_file)

# Iterative refinement process
while True:
    user_input = input("Enter user query (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    
    # TODO: Use the trained model to predict the intent of user input
    predicted_intent = predict_intent(user_input)
    
    # TODO: Get user feedback on the predicted intent (correct or incorrect)
    user_feedback = get_user_feedback(predicted_intent)
    
    if user_feedback == "correct":
        continue  # No refinement needed, move to the next query
    
    if user_feedback == "incorrect":
        new_patterns, new_responses = get_user_input_data(user_input)
        
        # Update the intents data with new patterns and responses
        intents = load_intents_from_file(output_file)
        update_intents(intents, predicted_intent, new_patterns, new_responses)
        save_intents_to_file(intents, output_file)
        
        print("Intents data updated successfully!")
