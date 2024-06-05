import re
import os
import numpy as np
import json
from keras.models import load_model
import word_tokenizer as wt
import util_methods as um
from keras import backend as K

# Chemin par défaut du modèle
model_path = 'model_to_show_5.h5'

# Définition de la métrique personnalisée d'accuracy avec une tolérance de 20
def accuracy_with_tolerance(tolerance=20):
    def accuracy(y_true, y_pred):
        # Calcul de la différence absolue entre la valeur prédite et la valeur réelle
        diff = K.abs(y_pred - y_true)

        # Compte le nombre de prédictions qui sont dans la tolérance
        within_tolerance = K.less_equal(diff, tolerance)

        # Retourne le pourcentage de prédictions correctes
        return K.mean(K.cast(within_tolerance, 'float32'))

    return accuracy

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
    filenames = []
    for fichier in fichiers:
        chemin_fichier = os.path.join(chemin, fichier)
        texte,annee = get_data_book(fichier,chemin_fichier)
        if(annee != None):
            books.append(texte)
            y.append(annee)
            filenames.append(fichier)
    return books, y,filenames

# Lecture des dictionnaires JSON
path_dict_word = "dict_word.json"
path_dict_word_freq = "dict_word_freq.json"

with open(path_dict_word, "r") as json_file:
    dict_word = json.load(json_file)
with open(path_dict_word_freq, "r") as json_file:
    word_freq = json.load(json_file)

tokenizer = wt.WordTokenizer(dict_word=dict_word, dict_word_frequence=word_freq)

# Fonction pour charger des données à partir d'un fichier
def charger_donnees_de_fichier(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            texte = f.read()
        return texte
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None

if __name__ == "__main__":
    model_path = ""
    # Demander le chemin du modèle jusqu'à ce qu'un chemin valide soit fourni
    while not os.path.isfile(model_path):
        model_path = input("Veuillez fournir le chemin vers le fichier du modèle : ")
        if not os.path.isfile(model_path):
            print("Le chemin spécifié n'est pas un fichier valide.")

    # Chargement du modèle
    model = load_model(model_path, custom_objects={'accuracy': accuracy_with_tolerance(25)})
    
    chemin = ""
    # Demander le chemin du fichier ou du dossier jusqu'à ce qu'un chemin valide soit fourni
    while not os.path.exists(chemin):
        chemin = input("Veuillez fournir le chemin vers le fichier texte ou le dossier : ")
        if not os.path.exists(chemin):
            print("Le chemin spécifié n'existe pas.")

    # Afficher le résumé du modèle
    model.summary()

    # Obtenir la taille de l'entrée attendue
    input_shape = model.input_shape
    
    # Constantes
    max_len = input_shape[0][1]

    # Si c'est un fichier, chargez les données
    if os.path.isfile(chemin):
        fichier = os.path.basename(chemin)
        texte,annee = get_data_book(fichier,chemin)
        X_test, y_test, filenames = [texte],[annee],[fichier]

    # Si c'est un dossier
    elif os.path.isdir(chemin):
        X_test, y_test, filenames = get_data_books(chemin)
        
    sequences_testA, sequences_testB, sequences_testC, sequences_testD = tokenizer.tokenize(X_test)
    data_testA, y_test, sequence_count_book_test = um.diviser_liste(sequences_testA, y_test, max_len)
    data_testB = um.diviser_liste2(sequences_testB, max_len)
    data_testC = um.diviser_liste2(sequences_testC, max_len)
    data_testD = um.diviser_liste2(sequences_testD, max_len)
    test = [data_testA, data_testB, data_testC, data_testD]
    predictions = model.predict(test)

    loss_test, mae_test,accuracy_test = model.evaluate(test, y_test)

    predicted_book_date = np.zeros(len(sequence_count_book_test))
    true_book_date = np.zeros(len(sequence_count_book_test))
    diff_total = 0
    ind = 0
    current = 0
    # Parcours de toutes les séquences
    for nb in sequence_count_book_test: 
        dates = []  # Liste pour stocker les dates prédites pour chaque séquence

        # Collecte des prédictions pour chaque séquence
        for i in range(nb):
            dates.append(predictions[current])  # Ajouter la prédiction actuelle à la liste des dates
            current += 1  # Incrémenter l'indice pour la prochaine prédiction
    
        # Retirer les valeurs extrêmes
        if len(dates) > 2:
            dates.remove(min(dates))  # Supprime la valeur minimale
            dates.remove(max(dates))  # Supprime la valeur maximale

        # Calculer la moyenne des dates prédites (sans les valeurs extrêmes)
        date = sum(dates) / len(dates) if dates else 0
        # date = statistics.median(dates)
        # Mettre à jour les tableaux avec la date moyenne prédite et la vérité terrain
        predicted_book_date[ind] = int(date)
        true_book_date[ind] = y_test[current - 1]

        # Calculer la différence absolue entre la moyenne prédite et la vérité terrain
        diff = abs(date - y_test[current - 1])
        diff_total += diff

        ind += 1

    for i in range(len(predicted_book_date)):
        print(f"{filenames[i]}, Prédiction : {predicted_book_date[i]}, Réel : {true_book_date[i]}")

    # Calculer la MAE pour les livres
    mae_book = diff_total / len(sequence_count_book_test)

    tolerance = 25  # Tolérance définie pour considérer les prédictions comme correctes
    within_tolerance_count = 0  # Compteur pour les prédictions dans la tolérance

    for i in range(len(predicted_book_date)):
        # Vérification si la différence entre la date réelle et la date prédite est dans la tolérance
        if abs(round(true_book_date[i]) - predicted_book_date[i]) <= tolerance:
            within_tolerance_count += 1

    # Calcul de l'exactitude dans la tolérance
    accuracy_within_tolerance_book = (within_tolerance_count / len(predicted_book_date)) * 100

    # Affichage des résultats
    print("-"*20)
    print("RESULTATS")
    print("-"*20)
    print(f"MSE (Mean Square Error): {loss_test:.4f}")
    print(f"MAE (Mean Absolute Error): {mae_test:.4f}")
    print(f"Sequence Accuracy within tolerance: {accuracy_test:.4f}")
    print(f"MAE Book (Mean Absolute Error): {mae_book[0]:.4f}")
    print(f"Book Accuracy within tolerance: {accuracy_within_tolerance_book:.2f}%")