from math import ceil
import numpy as np

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

def divide_by_seq(df,max_len):
    sequences_A = df["Sequence A"].to_list()
    sequences_B = df["Sequence B"].to_list()
    sequences_C = df["Sequence C"].to_list()
    sequences_D = df["Sequence D"].to_list()

    # Division des données en séquences de longueur maximale max_len
    data_A,y,sequence_count_book = diviser_liste(sequences_A,df["Year"].to_list(), max_len)
    data_B = diviser_liste2(sequences_B, max_len)
    data_C = diviser_liste2(sequences_C, max_len)
    data_D = diviser_liste2(sequences_D, max_len)

    # Affichage de la longueur des données
    print("Nombre de séquences de données:", len(data_A))

    return [data_A,data_B,data_C,data_D],y,sequence_count_book
    