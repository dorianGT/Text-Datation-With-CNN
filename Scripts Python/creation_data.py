import json
import re
import os
import pickle
import pandas as pd
import word_tokenizer

def GetDataBooks(chemin,df):
    # Récupérer la liste des noms de fichiers dans le dossier
    fichiers = os.listdir(chemin)
    
    # Initialiser une liste pour stocker les textes
    books = []

    # Initialiser une liste pour stocker les années extraites des noms de fichiers
    y = []
    correct_fichiers = []
    # Parcourir les fichiers et lire leur contenu
    for fichier in fichiers:
        chemin_fichier = os.path.join(chemin, fichier)
        with open(chemin_fichier, 'r', encoding='utf-8') as f:

            texte = f.read()

            # Extraire l'année du nom de fichier
            annee = re.search(r'\((\d{4})\)', fichier)  # Utilisation d'une expression régulière pour trouver l'année entre parenthèses
            
            # Si une année est trouvée, alors nous pouvons étudier le livre, nous l'ajoutons ainsi que son label dans les listes associées
            if annee:
                annee = int(annee.group(1)) # Récupérer l'année
                books.append(texte)   
                y.append(annee)  # Ajouter l'année extraite à la liste des labels
                correct_fichiers.append(fichier)
            else:
                print(f"Impossible de lire la date pour le fichier: {fichier}")
    fichiersframe = pd.DataFrame({'Filename': correct_fichiers})
    df['Filename'] = fichiersframe['Filename']
    yearframe = pd.DataFrame({'Year': y})
    df['Year'] = yearframe['Year']
    booksframe = pd.DataFrame({'Text': books})
    df['Text'] = booksframe['Text']


if __name__ == "__main__":

    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()

    # Chemin des dossier contenant les textes
    dossier_train = 'train'
    dossier_test = 'test'
    dossier_val = 'validation'

    GetDataBooks(dossier_train,df_train)
    GetDataBooks(dossier_val,df_val)
    GetDataBooks(dossier_test,df_test)

    tokenizer = word_tokenizer.WordTokenizer()
    print("Tokenizer Fit pout DS Train")
    tokenizer.fit(df_train["Text"].to_list())
    print("Tokenizer Fit pout DS Validation")
    tokenizer.fit(df_val["Text"].to_list())

    # Chemin du fichier de sauvegarde
    path_dict_word = "dict_word.json"
    path_dict_word_freq = "dict_word_freq.json"

    # Sauvegarde du dictionnaire en JSON dans un fichier
    with open(path_dict_word, "w") as json_file:
        json.dump(tokenizer.dict_word, json_file)
    with open(path_dict_word_freq, "w") as json_file:
        json.dump(tokenizer.word_freq, json_file)


    print("Tokenization pout DS Train")
    # Transformation de chaque texte dans train en séquence
    sequences_trainA,sequences_trainB,sequences_trainC,sequences_trainD = tokenizer.tokenize(df_train["Text"].tolist())

    seqA = pd.DataFrame({'Sequence A': sequences_trainA})
    df_train['Sequence A'] = seqA['Sequence A']
    seqB = pd.DataFrame({'Sequence B': sequences_trainB})
    df_train['Sequence B'] = seqB['Sequence B']
    seqC = pd.DataFrame({'Sequence C': sequences_trainC})
    df_train['Sequence C'] = seqC['Sequence C']
    seqD = pd.DataFrame({'Sequence D': sequences_trainD})
    df_train['Sequence D'] = seqD['Sequence D']
    df_train['Form_Count'] = df_train['Sequence A'].apply(lambda seq: len(seq))

    os.makedirs('csv', exist_ok=True)

    # Save df
    with open('csv/df_train.pkl', 'wb') as f:
        pickle.dump(df_train, f)

    print("Tokenization pout DS Val")
    # Transformation de chaque texte dans val en séquence
    sequences_valA,sequences_valB,sequences_valC,sequences_valD = tokenizer.tokenize(df_val["Text"].to_list())

    seqA = pd.DataFrame({'Sequence A': sequences_valA})
    df_val['Sequence A'] = seqA['Sequence A']
    seqB = pd.DataFrame({'Sequence B': sequences_valB})
    df_val['Sequence B'] = seqB['Sequence B']
    seqC = pd.DataFrame({'Sequence C': sequences_valC})
    df_val['Sequence C'] = seqC['Sequence C']
    seqD = pd.DataFrame({'Sequence D': sequences_valD})
    df_val['Sequence D'] = seqD['Sequence D']
    df_val['Form_Count'] = df_val['Sequence A'].apply(lambda seq: len(seq))

    # Save df
    with open('csv/df_val.pkl', 'wb') as f:
        pickle.dump(df_val, f)

    print("Tokenization pout DS Test")
    # Transformation de chaque texte dans test en séquence
    sequences_testA,sequences_testB,sequences_testC,sequences_testD = tokenizer.tokenize(df_test["Text"].to_list())

    seqA = pd.DataFrame({'Sequence A': sequences_testA})
    df_test['Sequence A'] = seqA['Sequence A']
    seqB = pd.DataFrame({'Sequence B': sequences_testB})
    df_test['Sequence B'] = seqB['Sequence B']
    seqC = pd.DataFrame({'Sequence C': sequences_testC})
    df_test['Sequence C'] = seqC['Sequence C']
    seqD = pd.DataFrame({'Sequence D': sequences_testD})
    df_test['Sequence D'] = seqD['Sequence D']
    df_test['Form_Count'] = df_test['Sequence A'].apply(lambda seq: len(seq))

    # Save df
    with open('csv/df_test.pkl', 'wb') as f:
        pickle.dump(df_test, f)


    print("Fichiers créés")