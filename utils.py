import numpy as np
from random import shuffle
import math

def getProteinCharSet():
    return ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])
    
def save_model_parameters_numpy(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile
   
def load_model_parameters_numpy(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])

def now():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

def probToOneHotVector(prob):
    v = np.zeros(prob.shape)
    v[prob.argmax()] = 1
    return v

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen=None):
        self.chars = chars
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        if(maxlen is None ):
            maxlen = len(self.chars)
        self.maxlen = maxlen

    def encodeChar(self,c,indexOnly):
        index = self.char_indices[c]
        if(indexOnly): 
            return index
        else:
            v = np.zeros(len(self.chars))
            v[index] = 1
            return v

    #return a 2d numpy array with shape: (sentence_length,char_length)
    def encode(self, sentence,indexOnly=False,listOnly=False):
        X = [self.encodeChar(c,indexOnly) for _,c in enumerate(sentence)]
        if not listOnly:
            X = np.asarray(X)
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

def generateDataSet(ctable, sentences,indexOnly,listOnly):
    X = [ctable.encode(sent[:-1],indexOnly,listOnly) for sent in sentences]
    y = [ctable.encode(sent[1:],indexOnly,listOnly) for sent in sentences]
    X = np.asarray(X)
    y = np.asarray(y)
    return (X,y)

def splitToTrainAndVal(ctable,sentences,test_percent=0.1,indexOnly=False,listOnly=False):
    shuffle(sentences)
    split_at = int(math.floor(len(sentences) * (1-test_percent)))
    trainSet = generateDataSet(ctable,sentences[:split_at],indexOnly,listOnly)
    valSet = generateDataSet(ctable,sentences[split_at:],indexOnly,listOnly)
    return (trainSet,valSet)


