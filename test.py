import numpy as np
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten,BatchNormalization
from keras import regularizers

import re
import os

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


# Chemin du dossier contenant les textes
dossier_textes = 'CORPUS DATATION TER'

# Récupérer la liste des noms de fichiers dans le dossier
fichiers = os.listdir(dossier_textes)

# Trier les fichiers par ordre alphabétique
fichiers.sort()

# Initialiser une liste pour stocker les textes
books = []

# Initialiser une liste pour stocker les années extraites des noms de fichiers
y = []

# Parcourir les fichiers et lire leur contenu
for fichier in fichiers[:10]:
    chemin_fichier = os.path.join(dossier_textes, fichier)
    with open(chemin_fichier, 'r', encoding='utf-8') as f:

        texte = f.read()

        # Extraire l'année du nom de fichier
        annee = re.search(r'\((\d{4})\)', fichier)  # Utilisation d'une expression régulière pour trouver l'année entre parenthèses
        # Si une année est trouvée, alors nous pouvons étudier le livre, nous l'ajoutons ainsi que son label dans les listes associées
        if annee:
            books.append(texte)
            annee_int = int(annee.group(1)) # Récupérer l'année
            y.append(annee_int)  # Ajouter l'année extraite à la liste des labels

# Division de notre bdd en deux bdd, l'une pour l'entrainement et l'autre pour les tests
X_train, X_test, y_train, y_test = train_test_split(books, y, test_size=0.1, random_state=42)

# Afficher les annéees extraites
print(y_train)
#print(len(X_train[9]))

# Paramètres du Tokenizer
max_len = 5000 # Longueur maximale des séquences


class Tokenizer2:
    def __init__(self):
        self.word_to_number = {}
        self.next_number = 0

    def tokenize(self, list_of_texts):
        total_texts = len(list_of_texts)
        token_lists = []
        for i, text in enumerate(list_of_texts):
            text = text.lower()
            tokens = word_tokenize(text)
            token_lists.append([])

            for token in tokens:
                if token not in self.word_to_number:
                    self.word_to_number[token] = self.next_number
                    self.next_number += 1

                token_lists[-1].append(self.word_to_number[token])

            percentage = (i + 1) / total_texts * 100
            print(f"Progress: {percentage:.2f}% complete")

        print("\nTokenization complete.")
        return token_lists, self.next_number
    


def diviser_liste(liste,labels, taille):
    resultats = []
    res_labels = []
    ind = 0
    # Parcours de chaque sous-liste dans la liste principale
    for sous_liste in liste:
        longueur = len(sous_liste)
        nbtour = round(longueur/taille)
        lastpos=0
        # Division de la sous-liste en morceaux de taille spécifiée
        for i in range(nbtour):
            if(lastpos + taille > longueur):
              lastpos -= (lastpos + taille)-longueur

            # Ajout du morceau à la liste de résultats
            resultat = sous_liste[lastpos:lastpos + taille]
            lastpos = lastpos + taille
            resultats.append(resultat)

            # A chaque fois que je divisie, jajoute le label au meme indice
            res_labels.append(labels[ind])
        ind+=1

    return resultats,res_labels


tokenizer = Tokenizer2()
# Transformation de chaque texte dans X_train en une séquence d'entiers
sequences, token_number = tokenizer.tokenize(X_train)


data,y_train = diviser_liste(sequences,y_train,max_len)

max_words = token_number
print(y_train)
# Convertir y_train en tableau numpy
y_train = np.array(y_train)
data = np.array(data)
print(len(data))
print("Nombre d'index dans le tokenizer:", max_words)








embedding_dim = 100

def archi_3CONV32_F_DE_DR_DE():
  model = Sequential()

#inputdim = le nombre de voc, inputlength le nombre delement dans une liste
  model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
  model.add(Conv1D(32, 5, activation='relu'))
  model.add(Conv1D(32, 5, activation='relu'))
  model.add(Conv1D(32, 5, activation='relu'))
  model.add(Flatten())
  model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-6)))
  model.add(Dropout(0.5))
  model.add(Dense(1))

  return model

def archi_3CONV16_F_DE_DR_DE():
  model = Sequential()

  model.add(Embedding(max_words, embedding_dim, input_length=max_len))
  model.add(Conv1D(16, 5, activation='relu'))
  model.add(Conv1D(16, 5, activation='relu'))
  model.add(Conv1D(16, 5, activation='relu'))
  model.add(Flatten())
  model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-6)))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  return model

def archi_3CONV64_F_DE_DR_DE():
  model = Sequential()

  model.add(Embedding(max_words, embedding_dim, input_length=max_len))
  model.add(Conv1D(64, 5, activation='relu'))
  model.add(Conv1D(64, 5, activation='relu'))
  model.add(Conv1D(64, 5, activation='relu'))
  model.add(Flatten())
  model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-6)))
  model.add(Dropout(0.5))
  model.add(Dense(1))

  return model

def archi_3CONV64_32_16_F_DE_DR_DE():
  model = Sequential()

  model.add(Embedding(max_words, embedding_dim, input_length=max_len))
  model.add(Conv1D(64, 5, activation='relu'))
  model.add(Conv1D(32, 5, activation='relu'))
  model.add(Conv1D(16, 5, activation='relu'))
  model.add(Flatten())
  model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-6)))
  model.add(Dropout(0.5))
  model.add(Dense(1))

  return model



# Transformation de chaque texte dans X_train en une séquence d'entiers
sequences_test, token_number_test = tokenizer.tokenize(X_test)

data_test,y_test = diviser_liste(sequences_test,y_test,max_len)

# Convertir y_train en tableau numpy
y_test = np.array(y_test)


model = archi_3CONV32_F_DE_DR_DE()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#model.summary()

# Entraînement du modèle
history = model.fit(data, y_train, epochs=20, batch_size=32)
# history.history