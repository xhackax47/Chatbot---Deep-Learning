import json
import nltk
import numpy as np
import random 
import string
import tensorflow as tf 
import time

from nltk.stem import WordNetLemmatizer 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:46:34 2023

@author: samy_
"""

# Variables

punkt = nltk.download("punkt")
wordnet = nltk.download("wordnet")

# Fonctions

def programme():
    print("INITIALISATION ET DEMARRAGE DU PROGRAMME DE CHATBOT PAR DEEP LEARNING")
    time.sleep(5)
    punkt
    wordnet
    #afficherDigitsImage()
    #afficherGraphImage()
    #afficherDigitsImage()
    #entrainement1000img(digitsTab1D_train, digitsTarget_train, mlp)
    #evalPerf(digitsTab1D_test, digitsTarget_test, mlp)
    print("FIN DU PROGRAMME")