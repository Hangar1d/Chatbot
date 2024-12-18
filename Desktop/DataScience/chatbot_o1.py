import numpy as np
import nltk #natural langauge toolkit 
import ssl
import json
import random
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

import streamlit as st
from deep_translator import GoogleTranslator
from datetime import datetime
import os
import pandas as pd  # Add this import for data handling
from googletrans import Translator

# fix SSL certificate error. It's for macbook
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# download NLTK resources | 
nltk.download('punkt', quiet=True) 
nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize

# Initialize the chatbot with intents and prepare for training => 
class MentalHealthChatbot:
    def __init__(self, intents_file):
        """
        Initialize the chatbot with intents and prepare for training

        :param intents_file: Path to the intents JSON file
        """
        self.lemmatizer = WordNetLemmatizer() 
        self.translator = Translator()
        self.conversation_history = []
        self.model = None  # Initialize the neural network model attribute
        self.user_data = {}  # Initialize a dictionary to store user data
        
        # Load intents (intent.json file iig neeh)
        # Data collection (The data is collected from the intent.json file)
        with open(intents_file, 'r') as file:
            self.intents = json.load(file)

        # Lists to store training data
        self.words = [] # words list that use for training
        self.classes = [] # angilal turul 
        self.documents = [] # surgaltiin ugugdul
        self.ignore_chars = ['?', '!', '.', ',']
        
# Preprocess the (intents.json) data for training => 
    def preprocess_data(self):
        """
        Preprocess the (intents.json) data for training
        """
        # Process each intent (data cleaning)
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word | 토큰화 
                word_tokens = word_tokenize(pattern.lower())

                # Add to words list (tokenized words)
                self.words.extend(word_tokens)

                # Add documents for training
                self.documents.append((word_tokens, intent['tag']))

                # Add to classes if not already present
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # Lemmatize and remove duplicates (dawhardsan ugsiig arilgah)
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_chars]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

# Create training data for neural network => 
    def create_training_data(self):
        """
        Create training data for neural network
        """
        # Create training data
        training = []
        output_empty = [0] * len(self.classes) # output will be empty for first time

        for doc in self.documents:
            # Initialize bag of words
            bag = []

            # Lemmatize pattern words
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in doc[0]]
# data processing 
            # Create bag of words array
            bag = [1 if w in pattern_words else 0 for w in self.words]

            # Create output row
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            training.append([bag, output_row])

        # Shuffle training data
        random.shuffle(training)
        training = np.array(training, dtype=object)

        # Split into X and Y
        train_x = list(training[:, 0]) # input data
        train_y = list(training[:, 1]) # output data

        return np.array(train_x), np.array(train_y)

# Build and train neural network model => 
    def build_model(self, train_x, train_y):
        """
        Build and train neural network model

        :param train_x: Training input data
        :param train_y: Training output data
        :return: Trained model
        """
        # Create model 
        model = Sequential([
            Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(train_y[0]), activation='softmax')
        ])

        # Compile model
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train model
        model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

        return model
    
# Predict intent for input text => 
    def predict_intent(self, model, input_text):
        """
        Predict intent and generate a multilingual response
        """
        try:
            # Detect input language
            detected_lang = self.translator.detect(input_text).lang

            # Translate input to English
            input_translated = self.translator.translate(input_text, src=detected_lang, dest='en').text

            # Tokenize and lemmatize input
            input_words = word_tokenize(input_translated.lower())
            input_words = [self.lemmatizer.lemmatize(word) for word in input_words]

            # Create bag of words
            bag = [1 if w in input_words else 0 for w in self.words]

            # Predict class
            result = model.predict(np.array([bag]))[0]
            threshold = 0.25
            results = [[i, r] for i, r in enumerate(result) if r > threshold]

            if not results:
                return self._translate_response("I'm not sure I understand. Could you please rephrase that?", detected_lang)

            # Sort predictions by confidence
            results.sort(key=lambda x: x[1], reverse=True)
            tag = self.classes[results[0][0]]

            # Generate response
            for intent in self.intents['intents']:
                if intent['tag'] == tag:
                    response = random.choice(intent['responses'])
                    return self._translate_response(response, detected_lang)

            return self._translate_response("I'm here to help. Can you tell me more?", detected_lang)

        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Sorry, I'm having trouble understanding right now."

    def _translate_response(self, response, lang):
        """
        Translate the chatbot's response back to the detected language
        """
        try:
            if lang != 'en':  # Only translate if the language is not English
                response = self.translator.translate(response, src='en', dest=lang).text
        except Exception as e:
            print(f"Translation Error: {e}")
            response = "I'm sorry, but I couldn't translate my response."
        return response

    def train_and_save(self, model_file='chatbot_model.h5',
                        words_file='words.pkl',
                        classes_file='classes.pkl'):
        """
        Train the model and save necessary files

        :param model_file: File to save trained model
        :param words_file: File to save words list
        :param classes_file: File to save classes list
        """
        # Preprocess data
        self.preprocess_data()

        # Create training data
        train_x, train_y = self.create_training_data()

        # Build and train model
        self.model = self.build_model(train_x, train_y)  # Store the model as instance variable

        # Save model and processing data 
        self.model.save(model_file)
        pickle.dump(self.words, open(words_file, 'wb'))
        pickle.dump(self.classes, open(classes_file, 'wb'))

        print("Model training completed and saved!")
        return self.model

    def update_user_data(self, user_id, mood, condition):
        """
        Update user data with current mood and condition
        """
        if user_id not in self.user_data:
            self.user_data[user_id] = {
                'moods': [],
                'conditions': []
            }
        
        # Append current mood and condition
        self.user_data[user_id]['moods'].append(mood)
        self.user_data[user_id]['conditions'].append(condition)

        # Limit the history to the last 30 entries
        self.user_data[user_id]['moods'] = self.user_data[user_id]['moods'][-30:]
        self.user_data[user_id]['conditions'] = self.user_data[user_id]['conditions'][-30:]

    def compare_moods(self, user_id):
        """
        Compare today's mood with yesterday's mood
        """
        if user_id in self.user_data and len(self.user_data[user_id]['moods']) > 1:
            today_mood = self.user_data[user_id]['moods'][-1]
            yesterday_mood = self.user_data[user_id]['moods'][-2]
            # Logic to compare moods and provide advice
            if today_mood != yesterday_mood:
                return f"Your mood has changed from {yesterday_mood} to {today_mood}. It's important to reflect on what might have caused this change."
        return "I don't have enough data to compare your moods yet."

    def generate_mood_analysis(self, user_id):
        comparisons = self.compare_moods(user_id)
        response = ""

        if "yesterday" in comparisons:
            response += f"Yesterday, you felt {comparisons['yesterday']['mood']} with a rating of {comparisons['yesterday']['rating']}. "
        if "last_week" in comparisons:
            response += f"Last week, your mood was {comparisons['last_week']['mood']}. "
        if "last_month" in comparisons:
            response += f"Last month, you showed improvements compared to this month."

        response += "\nKeep focusing on activities that bring balance and joy to your life."
        return response

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MentalHealthChatbot('intents.json')
        
        # Add greeting message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": "Hi, I'm Khanora, your mental health assistant. How can I help you today?"})
        
        # Check if the model needs to be retrained
        intents_file = 'intents.json'
        last_modified_time = st.session_state.get('last_modified_time', None)
        current_modified_time = os.path.getmtime(intents_file)

        if last_modified_time is None or current_modified_time > last_modified_time:
            st.session_state.chatbot.train_and_save()
            st.session_state.last_modified_time = current_modified_time  # Update the last modified time
        else:
            # Load the existing model
            try:
                model = tf.keras.models.load_model('chatbot_model.h5')
                st.session_state.chatbot.model = model
                st.session_state.chatbot.words = pickle.load(open('words.pkl', 'rb'))
                st.session_state.chatbot.classes = pickle.load(open('classes.pkl', 'rb'))
            except:
                st.write("Training the model. Please wait...")
                st.session_state.chatbot.train_and_save()

def main():
    st.set_page_config(
        page_title="Mental Health Assistant",
        page_icon="🤖",
        layout="centered"
    )
    
    initialize_session_state()
    
    # Custom CSS
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            background-color: #f0f2f6;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #2b313e;
            color: #ffffff;
        }
        .chat-message.bot {
            background-color: #f0f2f6;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🤖 Mental Health Assistant")
    st.markdown("""
Welcome! I’m here to listen, support, and provide a safe space for you to share your thoughts and feelings. You can talk to me in the language you’re most comfortable with, and I’ll do my best to understand and respond with care. Let’s have a conversation about anything on your mind.
    """)
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # User input
    user_input = st.chat_input("What's going on with you?")
    
    if user_input:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get and display bot response
        response = st.session_state.chatbot.predict_intent(st.session_state.chatbot.model, user_input)
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This Khanora Mental Health Assistant can:
        - Understand multiple languages
        - Respond in the user's language
        - Provide mental health support
        - 여러 언어 이해합니다
        - 사용자의 언어로 응답합니다
        - 정신 건강 지원을 제공합니다.
        
        """)
        # Clear chat history
        if st.button("Clear Chat History"): 
            st.session_state.chat_history = [] # make chat history empty
            st.rerun() # rerun the app
    
if __name__ == "__main__":
    main()