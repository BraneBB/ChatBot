import numpy
import tflearn
import tensorflow as tf
import random
import pickle
import requests
from datetime import datetime
from data import dataparser, parseInputSentence

tf.compat.v1.reset_default_graph()

try:
    with open("trainingdata.pickle", "rb") as f:
        vocabulary, tags, trainingdata, outputdata, tagresponse = pickle.load(f)
except:
    vocabulary, tags, trainingdata, outputdata, tagresponse = dataparser('trainingdata')
    with open("data.pickle", "wb") as f:
        pickle.dump((vocabulary, tags, trainingdata, outputdata, tagresponse), f)


#input layer
net = tflearn.input_data(shape = [None, len(trainingdata[0])])
#hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
#output layers
net = tflearn.fully_connected(net, len(outputdata[0]), activation = 'softmax')
#regresija
net = tflearn.regression(net)
#odabir modela
model = tflearn.DNN(net)

try:
    model.load("ChatBotEN")
except:
    model.fit(trainingdata, outputdata, n_epoch = 500, batch_size = 10, show_metric = True)
    model.save("ChatBotEN")
#fitanje podataka modelu

def BOW(tokenizedSentence):
    bag = []
    for word in vocabulary:
        if word in tokenizedSentence:
            bag.append(1)
        else:
            bag.append(0)
    
    return numpy.array(bag)

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


def Chat():
    print("Start talking with the bot! (For quit just type 'quit')")
    while True:
        inputSentence = input("You: ")
        if inputSentence.lower() == "quit":
            break

        result = model.predict([BOW(parseInputSentence(inputSentence))])[0]
        result_index = numpy.argmax(result)

        tag = tags[result_index]
        if result[result_index] > 0.75:
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
        else:
            print("ChatBot: Sorry I didn't understand that, try typing something else.")
Chat()