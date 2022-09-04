import numpy
import tflearn
import tensorflow as tf
import random
import pickle
import requests
from datetime import datetime
import numpy as np
from data import rawDataToTraining
from gensim.utils import simple_preprocess
import gensim.downloader
import json

tf.compat.v1.reset_default_graph()

try:
    with open("data.pickle", "rb") as f:
        training, outputdata, tags, tagresponse = pickle.load(f)
except:
    training, outputdata, tags, tagresponse = rawDataToTraining('data')
    with open("data.pickle", "wb") as f:
        pickle.dump((training, outputdata, tags, tagresponse), f)

#input layer
net = tflearn.input_data(shape = [None, len(training[0])])
#hidden layers
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 8)
#output layers
net = tflearn.fully_connected(net, len(outputdata[0]), activation = 'softmax')
#regresija
net = tflearn.regression(net)
#odabir modela
chatbot = tflearn.DNN(net)

try:
    chatbot.load("ChatBotEN")
except:
# #fitanje podataka modelu
    chatbot.fit(X_inputs = training, Y_targets = outputdata, n_epoch = 500, batch_size = 10, show_metric = True)
    chatbot.save("ChatBotEN")

def Weather():
    apikey = "eaa1ff36c65a8953740ec81b7e9f4666"

    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + apikey + "&q=Osijek"

    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":

        y = x["main"]
        temp = y["temp"] - 272.15
        
        z = x["weather"]
        desc = z[0]["description"]

        pressure = y["pressure"]
        hum = y['humidity']

    else:
        print(" City Not Found ")
    
    
    return temp, desc, pressure, hum

def TestMetric(filename):
    score = 0
    total = 0
    new_model = gensim.downloader.load('glove-wiki-gigaword-300')
    trainingdata = json.loads(open(f'{filename}.json').read())
    for sets in trainingdata:
        for data in trainingdata[sets]:
            for pat in data['patterns']:
                result = [0] * 300
                for word in simple_preprocess(pat):
                    parseword = new_model[word]
                    result = np.add(result, parseword)
            
                result = chatbot.predict([result])[0]
                result_index = numpy.argmax(result)
                tag = tags[result_index]
                total += 1
                if tag == data['tag']:
                    score += 1 

    return score / total

def Chat():
    new_model = gensim.downloader.load('glove-wiki-gigaword-300')
    print("Start talking with the bot! (For quit just type 'quit')")
    while True:
        inputSentence = input("You: ")
        if inputSentence.lower() == "quit":
            break
        
        try:
            result = [0] * 300
            for word in simple_preprocess(inputSentence):
                    parseword = new_model[word]
                    result = np.add(result, parseword)
            
            result = chatbot.predict([result])[0]
            result_index = numpy.argmax(result)

            tag = tags[result_index]
            if tag == "time":
                dtime = datetime.now()
                print("ChatBot: " + random.choice(tagresponse[tag]) + str(dtime.hour), ':', str(dtime.minute))
            elif tag == "music":
                print("ChatBot: For now I can only write songs name, not sing or play them, here is your song: " + random.choice(tagresponse[tag]))
            elif tag == "weather":
                temp, desc, pressure, hum = Weather()
                print("ChatBot: " + random.choice(tagresponse[tag]) + f"temperature: {round(temp, 2)} ° C, description: {desc}, pressure : {pressure} mb, humidity: {hum}%")
            else:
                print("ChatBot: " + random.choice(tagresponse[tag]))
        except:
                print("ChatBot: Sorry I didn't understand that, try typing something else.")


Chat()

# print(TestMetric('test'))
# 0.8372093023255814