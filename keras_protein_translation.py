#! /usr/bin/python
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
import itertools
import operator
import numpy as np
import nltk
import sys
#! /usr/bin/python
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation
from keras.layers.recurrent import SimpleRNN
import itertools
import operator
import numpy as np
import os
import time
from datetime import datetime
import csv
from datetime import datetime
from utils import *
from random import shuffle

codonTable = {"UUU":"F", "UUC":"F", "UUA":"L", "UUG":"L",
    "UCU":"S", "UCC":"S", "UCA":"S", "UCG":"S",
    "UAU":"Y", "UAC":"Y", "UAA":"STOP", "UAG":"STOP",
    "UGU":"C", "UGC":"C", "UGA":"STOP", "UGG":"W",
    "CUU":"L", "CUC":"L", "CUA":"L", "CUG":"L",
    "CCU":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAU":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "CGU":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "AUU":"I", "AUC":"I", "AUA":"I", "AUG":"M",
    "ACU":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAU":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "AGU":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GUU":"V", "GUC":"V", "GUA":"V", "GUG":"V",
    "GCU":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAU":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "GGU":"G", "GGC":"G", "GGA":"G", "GGG":"G"}
proteinCharSet = getProteinCharSet() + ["STOP"]

codonCharSet = list('AUCG')
ctable = CharacterTable(codonCharSet) #generate 4 base pair
ctableProtein = CharacterTable(proteinCharSet)
pairs = codonTable.items()
X = np.asarray([ctable.encode(list(codon)) for (codon,_) in pairs])
y = np.asarray([ctableProtein.encodeChar(protein,False) for (_,protein) in pairs])

print("X[0] is ", X[0])
print("y[0] is ", y[0])

print "Building model"
model = Sequential()
# each training sentence has max 3 characters, each character is 4 dimension vector
model.add(SimpleRNN(output_dim=128, input_shape=(3,len(codonCharSet)),return_sequences=False,activation='sigmoid',init='lecun_uniform'))
model.add(Dense(len(proteinCharSet),init='lecun_uniform'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd')

print "Model built"

model.load_weights("protein_translation_codon_level")
print "Start training"
model.fit(X,y,nb_epoch=1000,batch_size=64,verbose=1)

model.save_weights("protein_translation_codon_level",overwrite=True)


predict = model.predict(X)
resultChar = [ctableProtein.indices_char[p.argmax()] for p in predict]
print pairs
print resultChar