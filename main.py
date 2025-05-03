import pickle
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Initialize lemmatizer and label encoder
lemmatizer = WordNetLemmatizer()
label_encoder = LabelEncoder()


# Load resources (classes, words, and labels)
def load_resources():
    # Load the pre-trained model
    model = tf.keras.models.load_model('chatbot_model.h5')

    # Load classes, words, and labels
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)

    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)

    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    return model, classes, words, labels


# Clean and preprocess the user input
def clean_input(user_input):
    # Tokenize the user input and apply lemmatization
    word_list = nltk.word_tokenize(user_input)
    word_list = [lemmatizer.lemmatize(w.lower()) for w in word_list]
    return word_list


# Convert the input to a bag of words
def bag_of_words(user_input, words):
    # Get the tokenized input
    user_input_words = clean_input(user_input)

    # Create an empty array for the bag of words
    bag = np.zeros(len(words), dtype=int)

    # Mark the presence of words in the user input
    for s in user_input_words:
        if s in words:
            index = words.index(s)
            bag[index] = 1

    return bag


# Predict the intent based on the user input
def predict_intent(user_input, model, classes, words, labels):
    # Convert input to a bag of words
    bow_input = bag_of_words(user_input, words)

    # Make prediction
    prediction = model.predict(np.array([bow_input]))[0]

    # Get the index of the highest probability
    predicted_class_index = np.argmax(prediction)

    # Get the corresponding intent
    predicted_class = classes[predicted_class_index]

    # Return the predicted class and its associated response
    return predicted_class


# Get the response from the corresponding intent
def get_response(intent, labels):
    # Define some responses based on intents (customize this as per your requirement)
    responses = {
        'greeting': ['Hello! How can I assist you today?', 'Hi there! What can I help you with?'],
        'goodbye': ['Goodbye! Have a great day!', 'Talk soon! Stay safe!'],
        'admissions': ['You can apply online through our website at https://emu.edu/admissions.'],
        'financial_aid': [
            'We offer a range of financial aid options including scholarships, grants, and student loans.'],
        'fallback': ['I\'m sorry, I didn\'t quite understand that. Could you rephrase?']
    }

    # Return the response related to the predicted intent
    return np.random.choice(responses.get(intent, ['Sorry, I don\'t have an answer for that.']))


# Main chat function
def chatbot(user_input):
    # Load resources (model, classes, words, labels)
    model, classes, words, labels = load_resources()

    # Predict the intent
    intent = predict_intent(user_input, model, classes, words, labels)

    # Get the corresponding response
    response = get_response(intent, labels)

    return response


# Run the chatbot (for testing)
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye! Have a great day!")
            break
        print(f"Chatbot: {chatbot(user_input)}")
