#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import itertools
import numpy as np
from utils import load_model_parameters_theano, save_model_parameters_theano
from rnn_theano import RNNTheano
import sys
import os
import time
from datetime import datetime

unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

def normalizeLines(filePath):
	print "loading..."
	with open(filePath,'rb') as f:
		sentences = [[sentence_start_token] + list(line.strip().decode("utf-8")) + [sentence_end_token] for line in f if len(line.strip()) > 0]
		print "loaded"
	print "found %d sentences" % len(sentences)
	print "last sentence is %s" % ''.join([x.encode('utf-8') for x in sentences[-1]])
	return sentences

def replaceUnknown(sentences,word_to_index,unknown_token):
	# Replace all words not in our vocabulary with the unknown token
	for i, sent in enumerate(sentences):
	    sentences[i] = [w if w in word_to_index else unknown_token for w in sent]


def generateIndexToWord(sentences,vocabulary_size):
	word_freq = nltk.FreqDist(itertools.chain(*sentences))
	print "Found %d unique words tokens." % len(word_freq.items())
	# Get the most common words and build index_to_word and word_to_index vectors
	vocab = word_freq.most_common(vocabulary_size-1)
	index_to_word = [x[0] for x in vocab]
	print "Using vocabulary size %d." % vocabulary_size
	print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0].encode('utf-8'), vocab[-1][1])
	index_to_word.append(unknown_token)
	return index_to_word

def generateWordToIndex(index_to_word):
	return dict([(w,i) for i,w in enumerate(index_to_word)])

def generateTrainingExamples(sentences,word_to_index):
	X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in sentences])
	y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in sentences])
	return (X_train,y_train)

def indexToSentence(x,indexToWord):
	return [indexToWord[i].encode('utf-8') for i in x]

def train(X_train,y_train,vocabulary_size,hiddenDim,modelFiles):
	model = RNNTheano(vocabulary_size, hidden_dim=hiddenDim)
	t1 = time.time()
	model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
	t2 = time.time()
	print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

	if modelFiles != None:
	    load_model_parameters_theano(modelFiles, model)

	train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)

def generate(modelPath,hiddenDim,word_to_index,index_to_word,vocabulary_size,minLength=10,sentStart=[sentence_start_token]):
	model = RNNTheano(vocabulary_size, hidden_dim=hiddenDim)
	# losses = train_with_sgd(model, X_train, y_train, nepoch=50)
	# save_model_parameters_theano('./data/trained-model-theano.npz', model)
	load_model_parameters_theano(modelPath, model)

	def generate_sentence(model):
	    # We start the sentence with the start token
	    new_sentence = [word_to_index[x] for x in [sentence_start_token] + sentStart]
	    # Repeat until we get an end token
	    while not new_sentence[-1] == word_to_index[sentence_end_token]:
	        next_word_probs = model.forward_propagation(new_sentence)
	        sampled_word = word_to_index[unknown_token]
	        # We don't want to sample unknown words
	        while sampled_word == word_to_index[unknown_token]:
	            samples = np.random.multinomial(1, next_word_probs[-1])
	            sampled_word = np.argmax(samples)
	        new_sentence.append(sampled_word)
	    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
	    return sentence_str
	 
	num_sentences = 60
	senten_min_length = minLength
	 
	for i in range(num_sentences):
	    sent = []
	    # We want long sentences, not sentences with one or two words
	    while len(sent) < senten_min_length:
	        sent = generate_sentence(model)
	    print " ".join(sent)

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '4000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '50'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

modelPath = "data/rnn-theano-80-4000-2016-02-10-09-36-52.npz"
filePath = "data/xi_you_ji_utf.txt"
sentences = normalizeLines(filePath)
replaceUnknown(sentences,wordToIndex,unknown_token) #get rid of infrequent tokens with unknown token
indexToWord = generateIndexToWord(sentences,_VOCABULARY_SIZE)
wordToIndex = generateWordToIndex(indexToWord)
print sys.argv
if len(sys.argv) > 1 and sys.argv[1] == "generate":
	print "generating..."
	generate(modelPath,_HIDDEN_DIM,wordToIndex,indexToWord,_VOCABULARY_SIZE,6,list("第三".decode("utf-8")))
	generate(modelPath,_HIDDEN_DIM,wordToIndex,indexToWord,_VOCABULARY_SIZE,10,list("诗曰".decode("utf-8")))
else:
	trainingSet = generateTrainingExamples(sentences,wordToIndex)
	train(trainingSet[0],trainingSet[1],_VOCABULARY_SIZE,_HIDDEN_DIM,_MODEL_FILE)


