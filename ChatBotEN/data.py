import json
import nltk
import numpy
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
ignore_puncts = ["?", ".", ":", ";", "'", "!", "/", "_", "&"]

def parseInputSentence(sentence):
    words = nltk.word_tokenize(sentence)
    words = [stemmer.stem(word.lower()) for word in words if word not in ignore_puncts]
    return words

def dataparser(filename):

    trainingdata = json.loads(open(f'{filename}.json').read())

    vocabulary = []
    tags = []
    docx = []
    docy = []
    tagresponse = {}

    for sets in trainingdata:
        for data in trainingdata[sets]:
            for pat in data['patterns']:
                words = parseInputSentence(pat)
            
                vocabulary.extend(words)
            
                docx.append(words)
                docy.append(data['tag'])

                if data['tag'] not in tags:
                    tags.append(data['tag'])
        for resp in trainingdata[sets]:
                if resp['tag'] in tagresponse.keys():
                    tagresponse[resp['tag']].extend(resp['responses'])
                else:  
                    tagresponse[resp['tag']] = resp['responses']
    

    vocabulary = sorted(set(vocabulary))
    tags = sorted(tags)

    trainingdata = []
    outputdata = []

    out_empty = [0] * len(tags)

    data = []

    for cnt, doc in enumerate(docx):
        bag = []
        output = out_empty[:]

        for w in vocabulary:
            if w in doc:
                bag.append(1)
            else:
                bag.append(0)

        trainingdata.append(bag)
        output[tags.index(docy[cnt])] = 1
        outputdata.append(output)
     
    trainingdata = numpy.array(trainingdata)
    outputdata = numpy.array(outputdata)
    
    return vocabulary, tags, trainingdata, outputdata, tagresponse