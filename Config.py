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
from Intents import *

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:46:34 2023

@author: xhackax47
"""

# Variables

# Téléchargement des fichiers du Langage Naturel
nltk.download('all')
# initialisation de lemmatizer pour obtenir la racine des mots
lemmatizer = WordNetLemmatizer()
# création des listes
# vocabulaire de tous les mots utilisés dans les patterns
mots = []
# Liste des tags de chaque intention
classes = []
# Liste de tous les patterns dans le fichier des intentions
doc_X = []
# Liste des tags associés à chaque pattern dans le fichier des intentions
doc_y = []

# Fonctions

def programme():
    print("INITIALISATION ET DEMARRAGE DU PROGRAMME DE CHATBOT PAR DEEP LEARNING")
    time.sleep(5)
    separationDonnees(mots, classes)
    #afficherDigitsImage()
    #afficherGraphImage()
    #afficherDigitsImage()
    #entrainement1000img(digitsTab1D_train, digitsTarget_train, mlp)
    #evalPerf(digitsTab1D_test, digitsTarget_test, mlp)
    print("FIN DU PROGRAMME")
    
def separationDonnees(mots, classes):
    """Séparation des données et remplissage des listes (mots et classes) à partir des données des intentions"""
    print("La séparation des données va bientôt commencée..")
    time.sleep(3)
    # parcourir avec une boucle For toutes les intentions
    # tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
    # le tag associé à l'intention sont ajoutés aux listes correspondantes
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            mots.extend(tokens)
            doc_X.append(pattern)
            doc_y.append(intent["tag"])
        
        # ajouter le tag aux classes s'il n'est pas déjà là 
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
    # lemmatiser tous les mots du vocabulaire et les convertir en minuscule
    # si les mots n'apparaissent pas dans la ponctuation
    mots = [lemmatizer.lemmatize(mot.lower()) for mot in mots if mot not in string.punctuation]
    # trier le vocabulaire et les classes par ordre alphabétique et prendre le
    # set pour s'assurer qu'il n'y a pas de doublons
    mots = sorted(set(mots))
    classes = sorted(set(classes))
    print("Affichage des données : ")
    time.sleep(3)
    print("Mots : " + str(mots))
    print("Classes : " + str(classes))
    print("Questions : " + str(doc_X))
    print("Type : " + str(doc_y))