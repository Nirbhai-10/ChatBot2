import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import os
import torch
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer

from nltk.stem.lancaster import LancasterStemmer

# Check if the model file exists
model_file = "model.tflearn"

if os.path.isfile(model_file):
    # Load the existing model
    model = tflearn.DNN.load(model_file)
else:
    # Load the intents data
    with open(r"C:\Users\priya\OneDrive\Desktop\Nirbhai\intents.json") as file:
        data = json.load(file)

    nltk.download('wordnet')
    nltk.download('punkt')

    stemmer = LancasterStemmer()

    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Looping through our data
    for intent in data['intents']:
        for pattern in intent['patterns']:
            pattern = pattern.lower()
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    print(training)

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(labels), activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=500, batch_size=8, show_metric=True)
    model.save(model_file)

# Load pre-trained BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load pre-trained GPT model and tokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

def preprocess_input(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def generate_bag_of_words(sentence, words):
    sentence_words = preprocess_input(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def bert_encode(sentence):
    input_ids = bert_tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():
        bert_embeddings = bert_model(input_ids)[0][:, 0, :].numpy()
    return bert_embeddings

def gpt_generate_response(sentence):
    input_ids = gpt_tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        gpt_output = gpt_model.generate(input_ids, max_length=50, num_return_sequences=1)
    response = gpt_tokenizer.decode(gpt_output[0], skip_special_tokens=True)
    return response

def classify_and_get_response(sentence):
    bag_of_words = generate_bag_of_words(sentence, words)
    results = model.predict([bag_of_words])[0]
    index = np.argmax(results)
    max_probability = results[index]
    confidence_threshold = 0.5

    if max_probability > confidence_threshold:
        tag = labels[index]
        intent = next((intent for intent in data['intents'] if intent['tag'] == tag), None)
        if intent:
            responses = intent['responses']
            weights = intent.get('weight', [1] * len(responses))
            response = random.choices(responses, weights=weights)[0]
            return response
    else:
        # Use GPT for generating response
        gpt_response = gpt_generate_response(sentence)
        return gpt_response
    
    return "I'm not sure."


def get_response(input_sentence):
    sentence = preprocess_input(input_sentence)
    response = classify_and_get_response(sentence)
    
    if response == "I'm not sure.":
        # Use BERT for generating response
        bert_embeddings = bert_encode(input_sentence)
        results = model.predict([bert_embeddings])[0]
        index = np.argmax(results)
        max_probability = results[index]
        confidence_threshold = 0.7

        if max_probability > confidence_threshold:
            tag = labels[index]
            intent = next((intent for intent in data['intents'] if intent['tag'] == tag), None)
            if intent:
                responses = intent['responses']
                weights = intent.get('weight', [1] * len(responses))
                response = random.choices(responses, weights=weights)[0]
        else:
            # Use GPT for generating response
            gpt_response = gpt_generate_response(input_sentence)
            response = gpt_response
    
    return response

print(training)