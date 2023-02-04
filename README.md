# Chatbot - Deep Learning

Ce projet me sert à expérimenter le deep learning à travers un modèle Sequential et le package Natural Language ToolKit.

 ![Example de fonctionnement de l'IA](https://i.ibb.co/ZcDVbwK/Capture-d-cran-2023-02-04-153425.png)

## Environnement

Nous aurons besoin de Python dernière version, l'IDE Spyder, Anaconda3 et Tensorflow avec Keras pour mettre tout en place.

- [Python](https://www.python.org/downloads/)
- [Spyder IDE](https://www.spyder-ide.org/)
- [Anaconda3](https://www.anaconda.com/)
- [Tensorflow](https://anaconda.org/conda-forge/tensorflow)

## Dépendances

Afin d'installer les dépendances nécessaires, il vaut mieux utiliser Anaconda pour créer un environnement virtuel : 

- Vous devez créer un nouvel environnement Conda avec les modules que vous voulez utiliser avec Spyder et y inclure spyder-kernels. Par exemple, si vous souhaitez utiliser scikit-learn, ouvrez votre terminal ou l'invite Anaconda sous Windows et exécutez les commandes suivantes pour créer l'environnement, l'activer et installer les modules/dépendances nécessaires:

```
conda create -n spyder-env -y
conda activate spyder-env
conda install spyder-kernels scikit-learn -y
conda install nomdumoduleainstaller
```

- Enfin, vous devez connecter Spyder à cet environnement en modifiant l'interpréteur Python par défaut de Spyder. Pour ce faire, cliquez sur le nom de l'environnement actuel dans la barre d'état, puis cliquez sur Changer l'environnement par défaut dans les Préférences.

- La boîte de dialogue Préférences s'ouvre alors dans la section Interpréteur Python. Ici, sélectionnez l'option Utiliser l'interpréteur Python suivant, et utilisez la liste déroulante ci-dessous pour sélectionner votre environnement préféré. S'il n'est pas répertorié, utilisez la zone de texte ou le bouton Select file pour saisir le chemin d'accès à l'interpréteur Python que vous souhaitez utiliser.

- Votre nouvel environnement ne sera listé ici que si vous avez installé Miniconda (ou Anaconda) dans le chemin par défaut comme indiqué dans le tableau ci-dessus.

- Cliquez sur Restart kernel dans le menu Consoles pour que ce changement prenne effet.