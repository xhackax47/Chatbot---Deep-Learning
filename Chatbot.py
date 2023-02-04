from Config import *

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:45:50 2023

@author: xhackax47
"""

# Fonction principale du programme
def principal():
    print("INITIALISATION ET DEMARRAGE DU PROGRAMME D'ENTRAINEMENT DU MODELE DEEP LEARNING")
    time.sleep(5)
    IA = entrainementIA(entrainement)
    chatbot(IA)
    print("FIN DU PROGRAMME")


# Execution de la fonction principale du programmeTe
principal()
