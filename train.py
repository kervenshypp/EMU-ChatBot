import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize the lemmatizer and other variables
lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Prepare the data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Tokenize each word in the intents and append to words list
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((pattern, intent['tag']))

    # Add the tag to the classes if it's not already there
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Lemmatize the words and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Save words as 'words.pkl'
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

# Save classes as 'classes.pkl'
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Create the training set
training_sentences = []
training_labels = []

# Create the bag of words model for each sentence
for doc in documents:
    bag = []
    pattern_words = nltk.word_tokenize(doc[0])
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]

    # Create the bag of words
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    training_sentences.append(bag)
    label = doc[1]
    training_labels.append(label)

# Convert training_labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)

# Save the label encoder
with open('labels.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Convert training data into numpy arrays
training_sentences = np.array(training_sentences)
training_labels = np.array(training_labels)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(training_sentences[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9), metrics=['accuracy'])

# Train the model
model.fit(training_sentences, training_labels, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')

print("Training complete and model saved!")
