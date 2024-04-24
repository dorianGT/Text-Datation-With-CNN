import sys
import re
import os
import tensorflow as tf
import numpy as np
import nltk
import json
import statistics
from math import ceil
from keras.models import load_model
from nltk.tokenize import word_tokenize
nltk.download('punkt')

if len(sys.argv) < 2:
    print("Veuillez fournir le chemin vers le dossier contenant les nouvelles données en argument.")
    sys.exit(1)

dossier_test = sys.argv[1]
model = load_model('model_to_show_2.h5')
max_len = 30000

class WordTokenizer:
    def __init__(self, dict_word=None, dict_word_frequence=None):
        if dict_word is None:
            self.dict_word = {}
            self.word_freq = {}
            self.index = 1
        else:
            self.dict_word = dict_word
            self.index = len(dict_word) + 1
            self.word_freq = dict_word_frequence

    def fit(self, list_of_texts):
        # If dict_word_frequence is None, initialize word_freq
        if self.word_freq is None:
            self.word_freq = {}

        # Calculate word frequency
        for text in list_of_texts:
            tokens = word_tokenize(text)
            for token in tokens:
                if token in self.word_freq:
                    self.word_freq[token] += 1
                else:
                    self.word_freq[token] = 1

        # Sort words by frequency in descending order
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        # Assign indices based on frequency
        for _, (word, _) in enumerate(sorted_words):
            if word not in self.dict_word:
                self.dict_word[word] = self.index
                self.index += 1

        print("\nTokenization complete.")

    def get_local_word_freq(self, tokens):
        # Initialise un dictionnaire pour stocker la fréquence locale des mots
        local_word_freq = {}
        # Parcourt tous les tokens dans la liste
        for token in tokens:
            # Vérifie si le token n'est pas déjà présent dans le dictionnaire
            if token not in local_word_freq:
                # Si le token n'est pas présent, initialise sa fréquence à 1
                local_word_freq[token] = 1
            else:
                # Si le token est déjà présent, incrémente sa fréquence de 1
                local_word_freq[token] += 1
        # Retourne le dictionnaire de fréquence locale des mots
        return local_word_freq

    def tokenize(self, list_of_texts):
        total_texts = len(list_of_texts)
        # Initialise un tableau pour stocker les tokens de chaque texte
        token_arrays = np.empty((total_texts,), dtype=object)
        local_occs = np.empty((total_texts,), dtype=object)
        token_pos = []
        # Parcourt chaque texte dans la liste
        for i, text in enumerate(list_of_texts):
            # Tokenise le texte en mots individuels
            tokens = word_tokenize(text)

            # Initialise un tableau pour stocker les informations sur chaque token
            token_array = np.empty((len(tokens), ), dtype=np.float32)
            local_occ = np.empty((len(tokens), ), dtype=np.float32)
            token_pos_arr = np.empty((len(tokens), ), dtype=np.float32)
            # Calcule la fréquence locale des mots dans le texte
            local_word_freq = self.get_local_word_freq(tokens)

            position = 1
            # Parcourt chaque token dans le texte
            for j, token in enumerate(tokens):
                # Position du token
                if token == '.':
                    position = 0
                token_pos_arr[j] = position
                #print("token :",token," pos :",position)
                position +=1

                # Vérifie si le token n'est pas dans le dictionnaire des mots
                if token not in self.dict_word:
                    token_array[j] = 0
                    local_occ[j] = local_word_freq[token]
                else:
                    token_array[j] = self.dict_word[token]
                    local_occ[j] = local_word_freq[token]
            
            token_arrays[i] = token_array
            local_occs[i] = local_occ
            token_pos.append(token_pos_arr)
            
            percentage = (i + 1) / total_texts * 100
            print(f"Progression : {percentage:.2f}% complète")

        print("\nTokenisation terminée.")
        return token_arrays,local_occs, token_pos


def GetDataBooks(chemin):
    # Récupérer la liste des noms de fichiers dans le dossier
    fichiers = os.listdir(chemin)
    
    # Initialiser une liste pour stocker les textes
    books = []

    # Initialiser une liste pour stocker les années extraites des noms de fichiers
    y = []

    ind_file = []
    ind = 0
    # Parcourir les fichiers et lire leur contenu
    for fichier in fichiers:
        chemin_fichier = os.path.join(chemin, fichier)
        with open(chemin_fichier, 'r', encoding='utf-8') as f:

            texte = f.read()

            # Extraire l'année du nom de fichier
            annee = re.search(r'\((\d{4})\)', fichier)  # Utilisation d'une expression régulière pour trouver l'année entre parenthèses
            # Si une année est trouvée, alors nous pouvons étudier le livre, nous l'ajoutons ainsi que son label dans les listes associées
            if annee:
                books.append(texte)
                annee_int = int(annee.group(1)) # Récupérer l'année
                y.append(annee_int)  # Ajouter l'année extraite à la liste des labels
                ind_file.append(ind)
        ind += 1
    return books,y,ind_file

def diviser_liste(liste,labels, taille):
    resultats = []
    res_labels = []
    nb_sequence_per_book = []
    ind = 0
    # Parcours de chaque sous-liste dans la liste principale
    for sous_liste in liste:
        longueur = len(sous_liste)
        if(longueur<taille):
            ind+=1
            continue
        nbtour = ceil(longueur / taille)
        lastpos=0
        # Division de la sous-liste en morceaux de taille spécifiée
        nb_sequence = 0
        for i in range(nbtour):
            if(lastpos + taille > longueur):
              if((lastpos + taille)-longueur > taille/2):
                  break
              lastpos -= (lastpos + taille)-longueur

            # Ajout du morceau à la liste de résultats
            resultat = sous_liste[lastpos:lastpos + taille]
            lastpos = lastpos + taille
            resultats.append(resultat)

            # A chaque fois que je divisie, jajoute le label au meme indice
            res_labels.append(labels[ind])
            nb_sequence += 1
        ind+=1
        nb_sequence_per_book.append(nb_sequence)

    # Convertir resultats et res_labels en tableaux numpy
    resultats = np.array(resultats)
    res_labels = np.array(res_labels)

    return resultats,res_labels,nb_sequence_per_book


def diviser_liste2(liste,taille):
    resultats = []
    ind = 0
    # Parcours de chaque sous-liste dans la liste principale
    for sous_liste in liste:
        longueur = len(sous_liste)
        if(longueur<taille):
            ind+=1
            continue
        nbtour = ceil(longueur / taille)
        lastpos=0
        # Division de la sous-liste en morceaux de taille spécifiée
        for i in range(nbtour):
            if(lastpos + taille > longueur):
              if((lastpos + taille)-longueur > taille/2):
                  break
              lastpos -= (lastpos + taille)-longueur

            # Ajout du morceau à la liste de résultats
            resultat = sous_liste[lastpos:lastpos + taille]
            lastpos = lastpos + taille
            resultats.append(resultat)
        ind+=1

    # Convertir resultats et res_labels en tableaux numpy
    resultats = np.array(resultats)

    return resultats

# Chemin du fichier JSON à lire
path_dict_word = "dict_word.json"
path_dict_word_freq = "dict_word_freq.json"

ditc_word = {}
word_freq = {}

# Lecture du fichier JSON
with open(path_dict_word, "r") as json_file:
    ditc_word = json.load(json_file)
with open(path_dict_word_freq, "r") as json_file:
    word_freq = json.load(json_file)
tokenizer = WordTokenizer(dict_word=ditc_word,dict_word_frequence=word_freq)


X_test,y_test,ind_file = GetDataBooks(dossier_test)
sequences_testA,sequences_testB,sequences_testC = tokenizer.tokenize(X_test)
data_testA,y_test,sequence_count_book_test = diviser_liste(sequences_testA,y_test, max_len)
data_testB = diviser_liste2(sequences_testB, max_len)
data_testC = diviser_liste2(sequences_testC, max_len)
test = [data_testA,data_testB,data_testC]
predictions = model.predict(test)
model.evaluate(test,y_test)


predicted_book_date = np.zeros(len(sequence_count_book_test)) # Liste pour stocker les dates de prédites pour chaque livre
true_book_date = np.zeros(len(sequence_count_book_test))  # Liste pour stocker les dates réelles

ind = 0
current = 0
for nb in sequence_count_book_test: 
    dates = []  # Liste pour stocker les dates prédites pour chaque séquence
    # Parcours de séquence
    for i in range(nb):
        dates.append(predictions[current])  # Ajouter la prédiction actuelle à la liste des dates
        current += 1  # Incrémenter l'indice pour obtenir la prochaine prédiction

    # Calculer la moyenne des dates prédites dans cette séquence
    date = sum(dates) / len(dates)
    # date = statistics.median(dates)
    predicted_book_date[ind] = date
    true_book_date[ind] = y_test[current-1]
    # # Calculer la différence absolue entre la moyenne prédite et la vérité terrain
    # diff = abs(date - y_test[current-1])
    # diff_total += diff

    ind += 1

print(len(ind_file),len(y_test),len(predicted_book_date))
print('len pred:',len(predictions),' len test:',len(y_test))

fichiers = os.listdir(dossier_test)
# Comparer les valeurs prédites avec les valeurs réelles
for i in range(len(predicted_book_date)):
    # fichier = fichiers[ind_file[i]]
    # nom_complet = fichier.split(')(')[0:5]
    # print(f"Prédiction : {predicted_book_date[i]}, Réel : {true_book_date[i]}, Livre : {nom_complet}")

    print(f"Prédiction : {predicted_book_date[i]}, Réel : {true_book_date[i]}")


