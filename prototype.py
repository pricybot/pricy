# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import os
import sys
import random
import json
import numpy as np
import time
import datetime
import trainingset as t
from daraz import daraz_data, daraz_price, daraz_compare
from nepbay import nepbay_data, nepbay_price, nepbay_compare
from writelogs import writeToJSONFile
stemmer = LancasterStemmer()

with open("daraz.json", "r") as read_file:
    data1 = json.load(read_file)

with open("nepbay.json", "r") as read_file:
    data2 = json.load(read_file)

#use to print available models in daraz    
#daraz_data()
#use to print abailable models in nepbay
#nepbay_data()
#to print daraz and nepbay model price
#daraz_price("j7")
#nepbay_price("j7")

#neural network algorithm implementation//
words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our t.training data
for pattern in t.training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print (len(documents), "documents")
#print (len(classes), "classes", classes)
#print (len(words), "unique stemmed words", words)


# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
#print ([stemmer.stem(word.lower()) for word in w])
#print (training[i])
#print (output[i])



# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=True):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2



def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s " % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) +"   Loading.. please wait")
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error  +"   Loading.. please wait")
                break

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)





X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")


# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])





def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    #Response set

      
    try:
        
        if return_results[0][0] == "greeting":
            time.sleep(1)
            print("Pricy: Hmm.. Which phone are you looking for?")
                

        elif return_results[0][0] == "health":
            time.sleep(1)
            print("Pricy: I am fine, thank you.")
                
                
        elif return_results[0][0] == "work":
            time.sleep(1)
            print("Pricy: I can compare prices of different mobile phones from various websites.")
            time.sleep(1.5)
            print("Pricy: What do you think about samsung phones?")


        elif return_results[0][0] == "askingbrand":
            time.sleep(1)
            print("Pricy: OK! please wait...")


        elif return_results[0][0] == "brand1" or return_results[0][0] == "brand2" or return_results[0][0] == "brand3" or return_results[0][0] == "brand4":
            time.sleep(1)
            print("Pricy: Which model shall I look for?")

            
        elif return_results[0][0] == "model1" or return_results[0][0] == "model2" or return_results[0][0] == "model3":
            word_tokens = nltk.word_tokenize(sentence)
            #print(word_tokens)
            stop_words = list(set(stopwords.words('english')))
            stop_words.extend(['guys', 'please', 'want', 'buy', 'mobile', 'phone', 'hi', 'pricy', 'Pricy', 'phones', 'mobiles', 'whats', 'what\'s', 'hm', 'so', 'hmm', 'um','ok',
                               'need', '?', 'available', 'u', 'sell', 'devices', 'device', 'brands', 'nepal', 'know','price', 'market', 'provide', 'me',
                               'current', 'value', 'tell', 'hello', 'pricy', 'how', 'are', 'you', 'brand', 'give', 'show', 'samsung', 'lg', 'motorola', 'lenevo',
                               'nokia', 'huawei', 'micromax', 'Price', 'model', 'galaxy', 'Galaxy', 'about', 'compare', 'you', 'have', 'get'])
            filtered_model = [w for w in word_tokens if not w in stop_words]
            filtered_model = []
            for w in word_tokens:
                if w not in stop_words:
                    filtered_model.append(w)

            usermodel = ' '.join(map(str, filtered_model))
            print (usermodel)
            
            time.sleep(1)
            print("Pricy: Ok I will search for it...")
            

            #list of available models in daraz
            daraz_model = []
            for i in range(36):
                daraz_model.append(data1[i]['Model'])

            
            #list of available models in nepbay
            nepbay_model = []
            for i in range(38):
                nepbay_model.append(data2[i]['Model'])

            capital_usermodel = usermodel.upper()
            if usermodel in daraz_model or capital_usermodel in daraz_model or usermodel in nepbay_model or capital_usermodel in nepbay_model:

                time.sleep(1.5)

                daraz_price(usermodel)
                nepbay_price(usermodel)
                
                price1 = daraz_compare(usermodel)
                #print(price1)
                price2 = nepbay_compare(usermodel)
                #print(price2)

                if int(price1) < int(price2):
                    print("Pricy: Minimum price at Daraz in Rs. "+ price1)
                else:
                    print("Pricy: Minimum price at Nepbay in Rs. "+ price2)

            else:
                print("Pricy: Sorry this model is not available")



            


        elif return_results[0][0] == "negative":
            time.sleep(1)
            print("Pricy: What shall I do for you?")


        elif return_results[0][0] == "utter":
            time.sleep(1)
            print("Pricy: Hmm")


        elif return_results[0][0] == "exit":
            time.sleep(1)
            print("Pricy: That was nice talking to you. Bye!!")
            time.sleep(1.5)
            print("**************************************************************************")
            sys.exit()


        else:
            print("Pricy: Sorry, I cannot understand what you are trying to say...")



    except IndexError:
        print("Pricy: Index error")
        pass

    except TypeError:
        print("Pricy: Type error")
        pass

    except ValueError:
        print("Pricy: Value error")
        pass
    
    return return_results
        




#Pricy conversation
print("\n\n------------------------Pricy: The Price Comparator Bot--------------------------\n")
print("Pricy: Hello I am pricy.")
time.sleep(1.5)
print("Pricy: I am here to provide you prices of different mobile phones.")
time.sleep(0.8)
while True:
    sentence = input("You: ")

    #save user logs to user log file
    path = './'
    fileName = 'usertextlogs'

    now = datetime.datetime.now()

    data = {}
    data['sentence'] = sentence
    data['datetime'] = now.strftime("%Y-%m-%d %H:%M")

    writeToJSONFile(path, fileName, data)
    #print("Log recored to "+ fileName)

    sentence = classify(sentence)
