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

# Chemin par défaut du modèle (peut être modifié)
model_path = 'model_to_show_2.h5'

# Chargement du modèle
model = load_model(model_path)

# Constantes
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
        # Si dict_word_frequence est None, initialiser word_freq
        if self.word_freq is None:
            self.word_freq = {}

        # Calculer la fréquence des mots
        for text in list_of_texts:
            tokens = word_tokenize(text)
            for token in tokens:
                if token in self.word_freq:
                    self.word_freq[token] += 1
                else:
                    self.word_freq[token] = 1

        # Trier les mots par fréquence dans l'ordre décroissant
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        # Attribuer des indices en fonction de la fréquence
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
        token_pos = np.empty((total_texts,), dtype=object)
        token_size = np.empty((total_texts,), dtype=object)
        # Parcourt chaque texte dans la liste
        for i, text in enumerate(list_of_texts):
            # Tokenise le texte en mots individuels
            tokens = word_tokenize(text)

            # Initialise un tableau pour stocker les informations sur chaque token
            token_array = np.empty((len(tokens), ), dtype=np.float32)
            local_occ = np.empty((len(tokens), ), dtype=np.float32)
            token_pos_arr = np.empty((len(tokens), ), dtype=np.float32)
            token_size_arr = np.empty((len(tokens), ), dtype=np.float32)
            # Calcule la fréquence locale des mots dans le texte
            local_word_freq = self.get_local_word_freq(tokens)

            position = 1
            # Parcourt chaque token dans le texte
            for j, token in enumerate(tokens):
                # Position du token
                if token == '.':
                    position = 0
                token_pos_arr[j] = position
                position +=1

                # Si le token est une ponctuation, on définit la taille à 0, sinon on prend la longueur du token
                if token in string.punctuation:
                    token_size_arr[j] = 0
                else:
                    token_size_arr[j] = len(token)

                # Vérifie si le token n'est pas dans le dictionnaire des mots
                if token not in self.dict_word:
                    token_array[j] = 0
                    local_occ[j] = local_word_freq[token]
                else:
                    token_array[j] = self.dict_word[token]
                    local_occ[j] = local_word_freq[token]
            
            token_arrays[i] = token_array
            local_occs[i] = local_occ
            token_pos[i] = token_pos_arr
            token_size[i] = token_size_arr

            percentage = (i + 1) / total_texts * 100
            print(f"Progression : {percentage:.2f}% complète")

        print("\nTokenisation terminée.")
        return token_arrays,local_occs, token_pos,token_size

def get_data_book(fichier,chemin_fichier):
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        texte = f.read()
        annee = re.search(r"\((\d{4})\)", fichier)

        if annee:
            return texte,int(annee.group(1))
        else:
            print("Erreur avec le fichier:",fichier)
            return None, None

# Fonction pour lire des livres et leurs labels
def get_data_books(chemin):
    fichiers = os.listdir(chemin)
    books = []
    y = []

    for fichier in fichiers:
        chemin_fichier = os.path.join(chemin, fichier)
        texte,annee = get_data_book(fichier,chemin_fichier)
        if(annee != None):
            books.append(texte)
            y.append(annee)

    return books, y


# Fonction pour diviser une liste en segments avec padding
def diviser_liste(liste, labels, taille):
    resultats = []
    res_labels = []
    nb_sequence_per_book = []
    ind = 0
    # Parcours de chaque sous-liste dans la liste principale
    for sous_liste in liste:
        longueur = len(sous_liste)
        if longueur < taille:
            padding = [0] * (taille - longueur) 
            sous_liste = np.concatenate((sous_liste, padding))
            longueur = taille

        nbtour = ceil(longueur / taille)
        lastpos = 0
        nb_sequence = 0
        # Division de la sous-liste en morceaux de taille spécifiée
        for i in range(nbtour):
            if lastpos + taille > longueur:
                if lastpos + taille - longueur > taille / 2:
                    break
                lastpos -= lastpos + taille - longueur

            # Ajout du morceau à la liste de résultats
            resultat = sous_liste[lastpos:lastpos + taille]
            lastpos += taille
            resultats.append(resultat)
            # A chaque fois que je divisie, jajoute le label au meme indice
            res_labels.append(labels[ind])
            nb_sequence += 1
        
        ind += 1
        nb_sequence_per_book.append(nb_sequence)

    # Convertir resultats et res_labels en tableaux numpy
    resultats = np.array(resultats)
    res_labels = np.array(res_labels)

    return resultats, res_labels, nb_sequence_per_book

# Fonction pour diviser une liste sans étiquettes
def diviser_liste2(liste,taille):
    resultats = []
    ind = 0
    # Parcours de chaque sous-liste dans la liste principale
    for sous_liste in liste:
        longueur = len(sous_liste)
        if longueur < taille:
            padding = [0] * (taille - longueur) 
            sous_liste = np.concatenate((sous_liste, padding))
            longueur = taille
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

# Lecture des dictionnaires JSON
path_dict_word = "dict_word.json"
path_dict_word_freq = "dict_word_freq.json"

with open(path_dict_word, "r") as json_file:
    dict_word = json.load(json_file)
with open(path_dict_word_freq, "r") as json_file:
    word_freq = json.load(json_file)

tokenizer = WordTokenizer(dict_word=dict_word, dict_word_frequence=word_freq)

# Fonction pour charger des données à partir d'un fichier
def charger_donnees_de_fichier(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            texte = f.read()
        return texte
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None

# Programme principal
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Veuillez fournir le chemin vers le fichier de données ou le dossier en argument.")
        sys.exit(1)

    chemin = sys.argv[1]

    if os.path.isfile(chemin) or os.path.isdir(chemin):
        # Si c'est un fichier, chargez les données
        if os.path.isfile(chemin):
            fichier = os.path.basename(chemin)
            texte,annee = get_data_book(fichier,chemin)
            X_test, y_test = [texte],[annee]

        # Si c'est un dossier
        elif os.path.isdir(chemin):
            X_test, y_test = get_data_books(chemin)
            
        sequences_testA, sequences_testB, sequences_testC = tokenizer.tokenize(X_test)
        data_testA, y_test, sequence_count_book_test = diviser_liste(sequences_testA, y_test, max_len)
        data_testB = diviser_liste2(sequences_testB, max_len)
        data_testC = diviser_liste2(sequences_testC, max_len)
        test = [data_testA, data_testB, data_testC]
        predictions = model.predict(test)
        model.evaluate(test, y_test)

        predicted_book_date = np.zeros(len(sequence_count_book_test))
        true_book_date = np.zeros(len(sequence_count_book_test))

        ind = 0
        current = 0
        for nb in sequence_count_book_test:
            dates = []
            for i in range(nb):
                dates.append(predictions[current])
                current += 1
            date = sum(dates) / len(dates)
            predicted_book_date[ind] = int(date)
            true_book_date[ind] = y_test[current - 1]
            ind += 1

        for i in range(len(predicted_book_date)):
            print(f"Prédiction {i} : {predicted_book_date[i]}, Réel : {true_book_date[i]}")
    else:
        print("Le chemin spécifié n'est ni un fichier ni un dossier.")
