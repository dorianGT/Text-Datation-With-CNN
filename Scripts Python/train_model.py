import json
import word_tokenizer
import model_func as mf
import util_methods as um
import pickle
import os
import ast
import pandas as pd
from keras import utils

if __name__ == "__main__":

    test_name = input("Entrez le nom du test: ")
    year_min = int(input("Entrez l'année minimale des textes: "))

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
    tokenizer = word_tokenizer.WordTokenizer(dict_word=ditc_word,dict_word_frequence=word_freq)

    token_number = tokenizer.index  

    # Load df
    with open('csv/df_train.pkl', 'rb') as f:
        df_train = pickle.load(f)

        # Filtrez les lignes où la colonne 'Year' est inférieure à 1800
        filtered_df = df_train[df_train['Year'] < year_min]

        # Obtenez le nombre de lignes
        count = len(filtered_df)

        print(f"Nombre de lignes où 'Year' est inférieur à {year_min}: {count}")
            # Créer un masque qui retient seulement les valeurs de 'Year' entre 1800 et 2020
        mask = (df_train['Year'] >= year_min)

        # Appliquer le masque pour filtrer le DataFrame
        df_train = df_train.loc[mask]

        df_train.reset_index(drop=True, inplace=True)

    # Load df
    with open('csv/df_val.pkl', 'rb') as f:
        df_val = pickle.load(f)
        # Filtrez les lignes où la colonne 'Year' est inférieure à 1800
        filtered_df = df_val[df_val['Year'] < year_min]

        # Obtenez le nombre de lignes
        count = len(filtered_df)

        print(f"Nombre de lignes où 'Year' est inférieur à {year_min}: {count}")

        # Créer un masque qui retient seulement les valeurs de 'Year' entre 1800 et 2020
        mask = (df_val['Year'] >= year_min)

        # Appliquer le masque pour filtrer le DataFrame
        df_val = df_val.loc[mask]

        df_val.reset_index(drop=True, inplace=True)

    # Load df
    with open('csv/df_test.pkl', 'rb') as f:
        df_test = pickle.load(f)
        
        # Filtrez les lignes où la colonne 'Year' est inférieure à 1800
        filtered_df = df_test[df_test['Year'] < year_min]

        # Obtenez le nombre de lignes
        count = len(filtered_df)

        print(f"Nombre de lignes où 'Year' est inférieur à {year_min}: {count}")

        # Créer un masque qui retient seulement les valeurs de 'Year' entre 1800 et 2020
        mask = (df_test['Year'] >= year_min)

        # Appliquer le masque pour filtrer le DataFrame
        df_test = df_test.loc[mask]
        df_test.reset_index(drop=True, inplace=True)

    utils.set_random_seed(420)

    
    csv_path_model_params = 'model_params.csv'
    df_model_params = pd.read_csv(csv_path_model_params)
    csv_results_name = f'csv_results_{test_name}'
    mf.create_csv_data(csv_results_name)

    ind = 0
    # Pour chaque ensemble de paramètres, entraînez le modèle
    for idx, row in df_model_params.iterrows():
        max_len = int(row['max_len'])

        train,y_train,sequence_count_book_train = um.divide_by_seq(df_train,max_len)
        val,y_val,sequence_count_book_val = um.divide_by_seq(df_val,max_len)
        test,y_test,sequence_count_book_test = um.divide_by_seq(df_test,max_len)

        # Créez le dictionnaire des paramètres à partir de la ligne de CSV
        params = {
            'included_branches': row['included_branches'],
            'use_embedding': row['use_embedding'],
            'embedding_dim': int(row['embedding_dim']),
            'count_conv1D': int(row['count_conv1D']),
            'conv1D_1_filters': ast.literal_eval(row['conv1D_1_filters']),
            'conv1D_1_kernel': ast.literal_eval(row['conv1D_1_kernel']),
            'pool_size': int(row['pool_size']),
            'dense_units_branch': int(row['dense_units_branch']),
            'dense_units': int(row['dense_units']),
            'dropout': float(row['dropout'].replace(',', '.'))
        }

        print(params)

        # Entraîner le modèle
        model, history = mf.train_model(params,train,y_train,val,y_val,max_len,token_number)

        #Evaluer le modèle
        loss_test, mae_test,loss_train, mae_train,loss_val, mae_val = mf.evaluate_model_loss_mae(model,test,y_test,train,y_train,val,y_val)
        accuracy_within_tolerance_sequence = mf.evaluate_model_sequence_accuracy(model,test,y_test)
        accuracy_within_tolerance_book,predicted_book_date,true_book_date = mf.evaluate_model_book_accuracy(model,test,y_test,sequence_count_book_test)

        model_name = "Model "+str(ind)
        save_dir = f'automatic_save/{test_name}/{model_name}'

        os.makedirs(save_dir, exist_ok=True)

        # Colonnes à sauvegarder
        col_to_save = ['Filename', 'Year']
        # Créer une copie profonde du sous-dataframe avec les colonnes souhaitées
        df_to_save = df_test[col_to_save].copy()
        # Sauvegarder ce sous-dataframe en fichier CSV
        seq_book = pd.DataFrame({'Predicted': predicted_book_date})

        df_to_save['Predicted'] = seq_book['Predicted']
        # Calculer l'écart entre les valeurs réelles et prédictes
        df_to_save['Difference'] = df_to_save['Year'] - df_to_save['Predicted']

        df_to_save.to_csv(f'{save_dir}/df_test.csv', index=False)

        mf.add_data_to_csv(max_len,mae_test,loss_test,accuracy_within_tolerance_sequence,accuracy_within_tolerance_book,
                            params,model_name,csv_results_name)
        
        #Sauvegarder le modèle complet au format HDF5
        model.save(f'{save_dir}/model.h5')

        with open(f'{save_dir}/history.json', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        mf.save_model_plot(history,model,test,y_test,save_dir,predicted_book_date,true_book_date,df_to_save)

        ind+=1