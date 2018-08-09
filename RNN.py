import itertools
import numpy as np
import sys
import nltk
import csv
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt

class RNN:

    def __init__(self, vocab_size, hidden_units=100, bptt_truncate=4):
        self.vocab_size = vocab_size
        self.num_hidden = hidden_units
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(-1.0/vocab_size), np.sqrt(1.0/vocab_size), (hidden_units ,vocab_size))
        self.V = np.random.uniform(-np.sqrt(-1.0/hidden_units), np.sqrt(1.0/hidden_units), (vocab_size, hidden_units))
        self.W = np.random.uniform(-np.sqrt(-1.0/hidden_units), np.sqrt(1.0/hidden_units), (hidden_units, hidden_units))

    def feed_forward(self, input):
        T = len(input)
        s = np.zeros((T + 1, self.hidden_units))
        s[-1] = np.zeros(self.hidden_units)
        o = np.zeros((T, self.vocab_size))
        for t in range(T):
            s[t] = np.tanh(self.U[:,input[t]] + self.W.dot(s[t - 1]))
            o[t] = np.softmax(self.V.dot(s[t]))

        return [o, s]

    def prediction(self, x):
        o, s = self.feed_forward(x)
        return np.argmax(0, axis=1)
