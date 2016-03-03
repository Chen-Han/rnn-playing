#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from Bio import SeqIO
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
from random import shuffle

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '50'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '50'))
_MODEL_FILE = os.environ.get('MODEL_FILE')
JOB_NAME = "e_coli_cdna_naive_rnn"

# vocabulary_size = _VOCABULARY_SIZE no use
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"



# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading fasta file..."
fileName = "e_coli_cdna.fa"
sentences = [list(record.seq) for record in SeqIO.parse("data/"+fileName, "fasta")]


print "Parsed %d sentences." % (len(sentences))
print sentences[1:3]


# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*sentences))
print "Found %d unique words tokens." % len(word_freq.items())
print word_freq.items()

# vocabulary_size = len(word_freq.items())+2
# index_to_word = [x[0] for x in word_freq.items()] + [sentence_start_token,sentence_end_token]
vocabulary_size = 7
index_to_word = [sentence_start_token,'A','T','C','G','N',sentence_end_token]
print index_to_word
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

def generate_sentence(model,minLength = 0,maxIter = 50):
    print "generating sentences " 
    # We start the sentence with the start token
    
    sentence_str = []
    counter = 0
    while len(sentence_str) < minLength and counter<maxIter:
        index = word_to_index[sentence_start_token]
        new_sentence = [index]
        # Repeat until we get an end token
        while not new_sentence[-1] == word_to_index[sentence_end_token]:
            next_word_probs = model.forward_propagation(new_sentence)
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
        counter += 1
    print "generated " + " ".join(sentence_str)
    return sentence_str

print "Using vocabulary size %d." % vocabulary_size

if len(sys.argv) > 1 and sys.argv[1] == "generate":
    print "loading ...."
    model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
    # losses = train_with_sgd(model, X_train, y_train, nepoch=50)
    # save_model_parameters_theano('./data/trained-model-theano.npz', model)
    load_model_parameters_theano('./data/rnn-theano-80-7-2016-02-09-21-39-28.npz', model)
    print "loaded"

    
     
    num_sentences = 10
    senten_min_length = 7
     
    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generate_sentence(model)
        print " ".join(sent)
else: # training
    
    def train_with_sgd(model, X_train, y_train, X_test, y_test, learning_rate=0.005, nepoch=1, evaluate_loss_after=3):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        with open(JOB_NAME+'.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['epoch','training_loss','testing_loss','model_path','sample_sentence'])
            for epoch in range(nepoch):
                print "begin epoch %d " % epoch
                # Optionally evaluate the loss
                if (epoch % evaluate_loss_after == 0):
                    trainLoss = model.calculate_loss(X_train, y_train)
                    losses.append((num_examples_seen, trainLoss))
                    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    print "%s: trainLoss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, trainLoss)
                    testLoss = model.calculate_loss(X_test, y_test)
                    print "Test loss: %f" % testLoss
                    # Adjust the learning rate if loss increases
                    if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                        learning_rate = learning_rate * 0.5  
                        print "Setting learning rate to %f" % learning_rate
                    sys.stdout.flush()
                    modelPath = "./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time)
                    # ADDED! Saving model oarameters
                    save_model_parameters_theano(modelPath, model)
                    sample_sentence = "".join(generate_sentence(model,minLength = 10))
                    print sample_sentence
                    writer.writerow([epoch,trainLoss,testLoss,modelPath,sample_sentence])
                # For each training example...
                print "begin training ...."
                for i in range(len(y_train)):
                    # One SGD step
                    model.sgd_step(X_train[i], y_train[i], learning_rate)
                    num_examples_seen += 1


    model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
    

    if _MODEL_FILE != None:
        load_model_parameters_theano(_MODEL_FILE, model)

    shuffle(sentences)
    split_at = len(sentences) - len(sentences) / 10

    def generateDataSet(word_to_index, sentences):
        # Create the training data
        X = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in sentences])
        y = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in sentences])
        return (X,y)
    # Create the training data
    X_train, y_train = generateDataSet(word_to_index,sentences[:split_at])
    X_test,y_test = generateDataSet(word_to_index,sentences[split_at:])

    t1 = time.time()
    model.sgd_step(X_train[100], y_train[100], _LEARNING_RATE)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
    train_with_sgd(model, X_train, y_train,X_test,y_test, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)



