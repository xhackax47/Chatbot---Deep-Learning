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
from Intentions import dictionnaire

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
"""Création des listes"""
# vocabulaire de tous les mots utilisés dans les patterns
mots = []
# Liste des tags de chaque intention
classes = []
# Liste de tous les patterns dans le fichier des intentions
doc_X = []
# Liste des tags associés à chaque pattern dans le fichier des intentions
doc_y = []
# Listes pour les données d'entraînement
entrainement = []
entrainement_x = []
entrainement_y = []
# Modèle Deep Learning
model = Sequential()
shapes = []
epochs = 200


# Fonctions


def entrainementIA():
    """Programme d'entrainement du modèle"""
    separationDonnees(mots, classes)
    shapes = traitementDonnees(entrainement, entrainement_x, entrainement_y)
    creationModelChatbot(model, entrainement_x,
                         entrainement_y, shapes[0], shapes[1])
    entrainementModelChatbot(model, entrainement_x, entrainement_y, epochs)


def separationDonnees(mots, classes):
    """Séparation des données et remplissage des listes (mots et classes) à partir des données des intentions"""
    print("La séparation des données va bientôt commencer..")
    time.sleep(2)

    # parcourir avec une boucle For toutes les intentions
    # tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
    # le tag associé à l'intention sont ajoutés aux listes correspondantes
    for intent in dictionnaire["intentions"]:
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
    mots = [lemmatizer.lemmatize(mot.lower())
            for mot in mots if mot not in string.punctuation]
    # trier le vocabulaire et les classes par ordre alphabétique et prendre le
    # set pour s'assurer qu'il n'y a pas de doublons
    mots = sorted(set(mots))
    classes = sorted(set(classes))
    print("separationDonnees OK !!!")
    print("Affichage des données brutes : ")
    print("Mots : " + str(mots))
    print("Classes : " + str(classes))
    print("Questions : " + str(doc_X))
    print("Type : " + str(doc_y))
    time.sleep(3)


def traitementDonnees(entrainement, entrainement_x, entrainement_y):
    """Traitement des données séparées avec création des listes pour l'entraînement et conversion en valeurs numériques"""
    print("Le traitement des données va bientôt commencer..")
    time.sleep(2)

    sortie_vide = [0] * len(classes)
    # Création du modèle d'ensemble de mots
    for idx, doc in enumerate(doc_X):
        bow = []
        texte = lemmatizer.lemmatize(doc.lower())
        for mot in mots:
            bow.append(1) if mot in texte else bow.append(0)
        # Marque l'index de la classe à laquelle le pattern atguel est associé à
        ligne_sortie = list(sortie_vide)
        ligne_sortie[classes.index(doc_y[idx])] = 1
        # Ajoute le one hot encoded BoW et les classes associées à la liste training
        entrainement.append([bow, ligne_sortie])
    # Mélanger les données et les convertir en array
    random.shuffle(entrainement)
    entrainement = np.array(entrainement, dtype=object)
    # Séparer les features et les labels target
    entrainement_x = np.array(list(entrainement[:, 0]))
    entrainement_y = np.array(list(entrainement[:, 1]))
    # Définition des paramètres
    input_shape = (len(entrainement_x[0]),)
    output_shape = len(entrainement_y[0])
    shapes = [input_shape, output_shape]

    print("traitementDonnees OK !!!")
    print("Affichage des données au format numérique : ")
    print("entrainement_x : " + str(entrainement_x))
    print("entrainement_y : " + str(entrainement_y))
    print("shapes : " + str(shapes))
    time.sleep(3)
    return shapes


def creationModelChatbot(model, entrainement_x, entrainement_y, input_shape, output_shape):
    """Création et définition du modèle de Deep Learning"""
    print("La création du modèle Deep Learning va bientôt commencer..")
    time.sleep(2)

    #epochs = 200
    # Définition du modèle Deep Learning
    model.add(Dense(128, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape, activation="softmax"))
    adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=["accuracy"])
    print("creationModelChatbot OK !!!")
    print("Affichage le sommaire du modèle Deep Learning : ")
    print(model.summary())
    time.sleep(3)


def entrainementModelChatbot(model, entrainement_x, entrainement_y, epochs):
    """Entraînement du modèle de Deep Learning"""
    print("L'entraînement du modèle Deep Learning va bientôt commencer..")
    time.sleep(2)

    # Entrainement du modèle
    try:
        model.fit(x=entrainement_x, y=entrainement_y, epochs=epochs, verbose=1)
        print("entrainementModelChatbot OK !!!")
        time.sleep(3)
    except KeyError:
        print("Problème de clé dans les modèles d'entraînement ('pop from an empty set' = .pop à partir d'un ensemble vide)")
        print("Fichier à problème :   File ~\.conda\envs\spyder-env\lib\site-packages\keras\engine\data_adapter.py:261 in '__init__).pop()'")


def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intentions"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


def chatbot():
    """Programme principal"""
    print("INITIALISATION ET DEMARRAGE DU PROGRAMME DE CHATBOT PAR DEEP LEARNING")
    # lancement du chatbot
    time.sleep(5)
    while True:
        print("Ecrivez un message et attendez la réponse de l'IA : ")
        message = input("")
        intents = pred_class(message, mots, classes)
        result = get_response(intents, dictionnaire)
        print("Kardrid : " + result)
