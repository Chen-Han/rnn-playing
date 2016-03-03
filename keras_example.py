#! /usr/bin/python
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation
from keras.layers.recurrent import SimpleRNN
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
import csv
from Bio import SeqIO
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
from random import shuffle

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encodeChar(self,c):
        v = np.zeros(len(self.chars))
        v[self.char_indices[c]] = 1
        return v

    def encode(self, sentence):
        if(self.maxlen):
            X = np.zeros((maxlen, len(self.chars)))
            for i, c in enumerate(sentence):
                X[i, self.char_indices[c]] = 1
            return X
        else:
            X = np.array([self.encodeChar(c) for _,c in enumerate(sentence)],dtype=object)
            return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

def now():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

def probToOneHotVector(prob):
    v = np.zeros(prob.shape)
    v[prob.argmax()] = 1
    return v
# vocabulary_size = _VOCABULARY_SIZE no use
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"



# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading fasta file..."
fileName = "e_coli_cdna.fa"
sentences = [[sentence_start_token] + list(record.seq) + [sentence_end_token] for record in SeqIO.parse("data/"+fileName, "fasta")]
print "Read %d sentences " % len(sentences)
shuffle(sentences)
print "shuffled sentences"

ctable = CharacterTable([sentence_start_token, 'A', 'C', 'T', 'G', 'N', sentence_end_token],None)

split_at = len(sentences) - len(sentences) / 10
sent_train = sentences[:split_at]
sent_val = sentences[split_at:]
print "sample training setence %s, %s" % (ctable.encode(sent_train[0][:-1]), ctable.encode(sent_train[0][1:]))

def evaluateOn(model, sentences,isTrain):
    totalLoss = 0.0
    for s in sentences:
        X_i = ctable.encode(s[:-1])
        y_i = ctable.encode(s[1:])
        if(isTrain): 
            totalLoss += model.train_on_batch(np.array([X_i]),np.array([y_i]))[0]
        else: 
            totalLoss += model.test_on_batch(np.array([X_i]),np.array([y_i]))[0]

        # print "totalLoss is %f " % totalLoss
    totalLoss = totalLoss / len(sentences)
    return totalLoss

def generateSentence(model,maxLength = 500):
    sentence = ctable.encode([sentence_start_token])
    assert(sentence.shape == (1,7))
    # model.predict would normally return tensor of shape 
    # (num_predictions, sentence_length,char_length)
    # but here we really only care about one example
    counter = 0
    while sentence[-1] is not  ctable.encodeChar(sentence_end_token) and counter < maxLength:
        X = np.array([sentence])
        prediction = model.predict(X, batch_size=1)[0]
        lastChar = probToOneHotVector(prediction[-1])
        sentence = np.append(sentence,[lastChar],axis=0) # append latest prediction
        counter += 1
    return ctable.decode(sentence)

HIDDEN_DIM = 20

print("building model of hidden dimension %d ..." % HIDDEN_DIM)
model = Sequential()
model.add(SimpleRNN(output_dim=HIDDEN_DIM, input_shape=(None,7),return_sequences=True,activation='sigmoid'))
model.add(TimeDistributedDense(7))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='sgd')
print("model built")
NUM_EPOCH = 20
EVAL_LOSS_PER = 1
JOB_NAME = "e_coli_cdna"
MODEL_FILE = sys.argv[1] if len(sys.argv) > 1 else None
if (MODEL_FILE): 
    model.load_weights(MODEL_FILE)
print generateSentence(model)
with open(JOB_NAME+'.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['epoch','training_loss','testing_loss','model_path_7_50','sample_sentence'])
    print "Begin training ..."
    initialModelPath = JOB_NAME + "-%s.h5" % now()
    model.save_weights(initialModelPath)
    initialLoss = evaluateOn(model,sent_train,False)
    print "Initial training set error: %f" % initialLoss
    initialTestLoss = evaluateOn(model,sent_val,False)
    print "Initial testing set error: %f" % initialTestLoss

    writer.writerow([0,initialLoss,initialTestLoss,initialModelPath])
    for epoch in range(1,NUM_EPOCH):
        testLoss = -1
        trainLoss = evaluateOn(model,sent_train,True)
        modelPath = "N/A"
        sample_sentence = "N/A"
        print "Loss after %d epochs is %0.4f" % (epoch,trainLoss)
        if epoch % EVAL_LOSS_PER == 0:
            testLoss = evaluateOn(model,sent_val,False)
            print "Test loss after %d epch is %0.4f" % (epoch,testLoss)
            modelPath = JOB_NAME + "-%s.h5" % now()
            model.save_weights(modelPath)
            print "Saved model to %s" % modelPath
            sample_sentence = generateSentence(model)
            print "Sample sentence is %s " % sample_sentence
        writer.writerow([epoch,trainLoss,testLoss,modelPath,sample_sentence])


