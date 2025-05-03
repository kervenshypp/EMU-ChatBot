import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import pickle
import os

# --- Initial Setup ---

# This section sets up the files and loads the necessary resources
#All of your training data is loaded from JSON files in the 'Training-Data' folder
folder_path = 'Training-Data'
files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
print("JSON Files in Training-Data folder:", files)

# Load the words and lables for the trained model
with open('words.pkl', 'rb') as f:
    words = pickle.load(f) # <-- List of all words that were used for classification

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)  # This is the LabelEncoder object, it maps out the intents to numeric lables

# Initialize the lemmatizer, basically makes the words uniform
lemmatizer = WordNetLemmatizer()

# This loads all the intents from the JSON files in Training-Data folder
intents = {"intents": []}
for filename in files:
    with open(os.path.join(folder_path, filename), "r") as json_data:
        current_data = json.load(json_data)
        intents["intents"].extend(current_data.get("intents", []))

# Load the trained model
# This model is trained by TensorFlowand can predict intents based on inputs
model = load_model('chatbot_model.h5')

# Preprocess the input text
def pre_process_input(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w.lower()) for w in words]
    return words

# Get the bag of words for the input
def bag_of_words(sentence, words):
    bag = [0] * len(words)
    words_input = preprocess_input(sentence) # Preprocess input
    for w in words_input:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Get the predicted intent
def predict_intent(text):
    bow = bag_of_words(text, words)
    bow = np.expand_dims(bow, axis=0)
    prediction = model.predict(bow)[0]
    if max(prediction) < 0.5:  # 50% confidence threshold
        return None
    return prediction

# Get the response based on the predicted intent
def get_response(prediction, label_encoder):  
    index = np.argmax(prediction) # Searching for index wth the highest confidence
    intent = label_encoder.inverse_transform([index])[0] # Convert index to intent
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent: # Pick a random response if the predicted intent matches
            return random.choice(intent_data['responses'])

# Main chatbot function
def chatbot(input_text):
    prediction = predict_intent(input_text) # Get the model's prediction
    if prediction is None:
        return "I'm not sure how to answer that." # If confidence is low, then return the default response
    response = get_response(prediction, labels)  # Fetch the correseponding response
    return response

# Simple chatbot loop for user interaction
if __name__ == "__main__":
    print("Chatbot is ready to talk! Type 'exit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        else:
            print(f"Chatbot: {chatbot(user_input)}")
