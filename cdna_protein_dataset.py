import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from Bio import SeqIO

sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

def withStartEndToken(sentence):
    return [sentence_start_token] + sentence + [sentence_end_token]
class CDNAProteinDataset(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, filePath):
        self.filePath = filePath
        self.cDNA = [r for r in SeqIO.parse(filePath, "fasta")]
        self.protein = None
        self.unknown_token = "N"
    def toProtein(self,cDNA=None):
    	if cDNA is None:
    		cDNA = self.cDNA
        if self.protein is None:
            self.protein = [withStartEndToken(list(record.seq.translate(to_stop=True))) for record in cDNA]
        return self.protein

    def toCDNA(self):
        return [withStartEndToken(list(record.seq)) for record in self.cDNA]

    def getCharSetCDNA(self):
        return [sentence_start_token,'A','T','C','G',self.unknown_token,sentence_end_token]

    def getCharSetProtein(self):
        return [sentence_start_token,'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X',sentence_end_token]


# # tests
# dataset = CDNAProteinDataset("data/tmp")
# print "CDNA exapmles: %d " % len(dataset.toCDNA())
# print dataset.toCDNA()[0]
# print dataset.toProtein()[0]
