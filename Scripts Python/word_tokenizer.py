import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import numpy as np
import string

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

        # Trier les mots par fréquence puis par ordre alphabétique pour les mêmes fréquences
        sorted_words = sorted(self.word_freq.items(), key=lambda x: (-x[1],x[0]))
        
        self.index = 1
        self.dict_word = {}
        # Attribuer des indices en fonction de la fréquence
        for _, (word, _) in enumerate(sorted_words):
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

        # Trier les mots par fréquence puis par ordre alphabétique pour les mêmes fréquences
        sorted_words = sorted(local_word_freq.items(), key=lambda x: (-x[1],x[0]))

        dict_local_word_freq = {}
        ind = 0
        # Attribuer des indices en fonction de la fréquence
        for _, (word, _) in enumerate(sorted_words):
            if word not in dict_local_word_freq:
                dict_local_word_freq[word] = ind
                ind += 1
                
        # Retourne le dictionnaire de fréquence locale des mots
        return dict_local_word_freq

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
            print(f"\rProgression : {percentage:.2f}% complète", end='', flush=True)

        print("\nTokenisation terminée.")
        return token_arrays,local_occs, token_pos,token_size