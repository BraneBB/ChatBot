from os import cpu_count
from gensim.models import Word2Vec
import gensim.downloader as api
from gensim.utils import simple_preprocess
import numpy as np
import json
import gensim.downloader

# def word2vecCreate():
#     dataset = api.load("text8")
#     data = [d for d in dataset]
#     data_train = data[:2500]
#     model = Word2Vec(sentences = data_train, window = 10, min_count = 1, workers = cpu_count(), vector_size = 300) 
#     model.save("word2vec.model")

# word2vecCreate()

def rawDataToTraining(filename):
    new_model = gensim.downloader.load('glove-wiki-gigaword-300')
    # model = Word2Vec.load("word2vec.model").wv
    training = []
    tagresponse = {}
    tags = []
    docy = []

    trainingdata = json.loads(open(f'{filename}.json').read())
    for sets in trainingdata:
        for data in trainingdata[sets]:
            for pat in data['patterns']:
                result = [0] * 300
                for word in simple_preprocess(pat):
                    result = np.add(result, new_model[word])
                
                training.append(result)

                docy.append(data['tag'])

                if data['tag'] not in tags:
                        tags.append(data['tag'])

        for resp in trainingdata[sets]:
                if resp['tag'] in tagresponse.keys():
                    tagresponse[resp['tag']].extend(resp['responses'])
                else:  
                    tagresponse[resp['tag']] = resp['responses']
    
    tags = sorted(tags)
    outputdata = []
    out_empty = [0] * len(tags)

    for i in range(len(training)):
        
        output = out_empty[:]

        output[tags.index(docy[i])] = 1
        outputdata.append(output)
    

    training = np.array(training)
    outputdata = np.array(outputdata)

    
    return training, outputdata, tags, tagresponse

# rawDataToTraining('data')