import random
import json
import pickle
import numpy as np
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pk1' , 'rb'))
classes = pickle.load(open('classes.pk1','rb'))

model= load_model("chatbot_model.h5")

def clean_up_sentence(sentence):
    sentence_words =nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]* len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word ==w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results =[[i,r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key = lambda x:x[1], reverse =True) 
    return_list =[]
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list  

def get_response(intents_list, intents_json):
    list_of_intents= intents_json['intents']
    tag =intents_list[0]['intent']
    for intent in list_of_intents:
        if intent['tag']==tag:
            result = random.choice(intent['responses'])
            break
    return result

print("Great Bot is running!")

while True:
    message = input("You: ")  # Prompt the user for input
    if message.lower() in ['exit', 'quit']:  # Check if the user wants to exit
        print("Bot: Goodbye!")
        break  # Exit the loop if the user wants to quit
    
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)  # Print the bot's response







