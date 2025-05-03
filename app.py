import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import pickle

# Load the necessary files
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Now you can use 'words', 'labels', and 'classes' for your chatbot logic

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Load the trained model
model = load_model('chatbot_model.h5')

# Preprocess the input text
def preprocess_input(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w.lower()) for w in words]
    return words

# Get the bag of words for the input
def bag_of_words(input_words, words):  # Use 'words' instead of 'vocabulary'
    bag = [0] * len(words)  # Change to 'words'
    words_input = preprocess_input(input_words)
    for w in words_input:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Get the predicted intent
def predict_intent(text):
    bow = bag_of_words(text, words)  # Pass 'words' here
    bow = np.expand_dims(bow, axis=0)
    prediction = model.predict(bow)[0]
    return prediction


# Get the response based on the predicted intent
def get_response(prediction):
    # Get the index of the highest probability
    index = np.argmax(prediction)

    # Decode the index back into the label (intent)
    intent = labels.inverse_transform([index])[0]  # Use inverse_transform to get the label

    # Find the response for the intent
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            return random.choice(intent_data['responses'])


# Main chatbot function
def chatbot(input_text):
    prediction = predict_intent(input_text)
    response = get_response(prediction)
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
