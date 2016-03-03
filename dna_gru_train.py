#! /usr/bin/env python

import sys
import os
import time
import numpy as np
from gru_utils import *
from datetime import datetime
from gru_theano import GRUTheano
from utils import CharacterTable, splitToTrainAndVal
from cdna_protein_dataset import CDNAProteinDataset

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "512"))
NEPOCH = int(os.environ.get("NEPOCH", "50"))
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/really_short_comments.csv")
# PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "11931"))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "10"))
JOB_NAME = os.environ.get("JOB_NAME","CDNA_GRU")

#getting dataset 
dataset = CDNAProteinDataset("./data/short_cdna.fa")
index_to_word = dataset.getCharSetCDNA()
word_to_index = dict((c,i) for i,c in enumerate(index_to_word))
print index_to_word
print word_to_index
ctableDNA = CharacterTable(index_to_word)


# ctableProtein = CharacterTable(dataset.getCharSetProtein())
(trainSet,valSet) = splitToTrainAndVal(ctableDNA,dataset.toCDNA(),indexOnly=True,listOnly=True)
x_train = trainSet[0][:10]
y_train = trainSet[1][:10]
x_val = valSet[0]
y_val = valSet[1]
print "Training size %d" % len(x_train)
print "sample train X ", x_train[0]
print "sample train y ", y_train[0]
print "Testing size %d" % len(x_val)

print "Type info "
print type(x_train)
print type(x_train[0])
print type(x_train[0][0])



print "Building model ..."
# Build model
model = GRUTheano(len(dataset.getCharSetCDNA()), hidden_dim=HIDDEN_DIM, bptt_truncate=-1)
print "Model built "

print "Getting sgd step time ..."
# Print SGD step time
t1 = time.time()
model.sgd_step(x_train[0], y_train[0], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()


with open(JOB_NAME+'.csv', 'wb') as csvfile:
  writer = csv.writer(csvfile, delimiter=',',
                      quotechar='"', quoting=csv.QUOTE_MINIMAL)
  writer.writerow(['epoch','num_examples_seen','training_loss','testing_loss','model_path','sample_sentence'])
  # We do this every few examples to understand what's going on
  def sgd_callback(model, num_examples_seen,epoch):
    dt = datetime.now().isoformat()
    testLoss = model.calculate_loss(x_val, y_val)
    print("\n%s (%d)" % (dt, num_examples_seen))
    print("--------------------------------------------------")
    print("Test Loss: %f" % testLoss )
    trainLoss = model.calculate_loss(x_train,y_train)
    print "train loss: %f" % trainLoss
    print ("Sample Sentence: ")
    sampleSentence = print_sentence(generate_sentence(model, index_to_word, word_to_index),index_to_word)

    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    MODEL_OUTPUT_FILE = "GRU-%s-%s-%s.dat" % (ts, EMBEDDING_DIM, HIDDEN_DIM)
    save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
    print ("Saved model to %s" % MODEL_OUTPUT_FILE)

    print("\n")
    sys.stdout.flush()
    writer.writerow([epoch,num_examples_seen,trainLoss,testLoss,MODEL_OUTPUT_FILE,sampleSentence])


  train_with_sgd(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=NEPOCH, decay=0.9, 
    callback_every=PRINT_EVERY, callback=sgd_callback)