import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Reshape, Dense, Dropout, Flatten, BatchNormalization, MaxPooling1D, SpatialDropout1D, Input, concatenate
from keras import utils
from keras.callbacks import EarlyStopping
from keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def define_multi_model(max_len, vocab_size,model_params):

    inputA = Input(shape=(max_len,), name='input_tokens')  # Entrée pour les mots tokenisés
    inputB = Input(shape=(max_len,), name='input_word_freq')  # Entrée pour les fréquences des mots dans le texte
    inputC = Input(shape=(max_len,), name='input_word_pos')  # Entrée pour la position des mots
    inputD = Input(shape=(max_len,), name='input_word_size')  # Entrée pour la taille des mots

    # Branche A
    # Couche pour l'entrée des mots tokenisés
    if(model_params["use_embedding"]):
        x1 = Embedding(input_dim=vocab_size, output_dim=model_params['embedding_dim'], input_length=max_len, name='embedding_tokens_A')(inputA)
    else:
        x1 = tf.keras.layers.Reshape((max_len, 1))(inputA)
    for i in range(model_params['count_conv1D']):
        x1 = Conv1D(filters=model_params['conv1D_1_filters'][i], kernel_size=model_params['conv1D_1_kernel'][i], activation='relu', name='conv1d_A_'+str(i))(x1)
        if(model_params['pool_size'] != 0):
            x1 = MaxPooling1D(pool_size=model_params['pool_size'], name='maxpooling_A_'+str(i))(x1)
    x1 = Flatten(name='flatten_A')(x1)
    if(model_params['dropout'] != 0):
        x1 = Dropout(model_params['dropout'], name='dropout_A')(x1)
    if(model_params['dense_units_branch'] != 0):
        x1 = Dense(model_params['dense_units_branch'], activation="relu", name='dense_A')(x1)

    # Branche A2
    # Couche pour l'entrée des mots tokenisés
    if(model_params["use_embedding"]):
        x2 = Embedding(input_dim=vocab_size, output_dim=model_params['embedding_dim'], input_length=max_len, name='embedding_tokens_A2')(inputA)
    else:
        x2 = tf.keras.layers.Reshape((max_len, 1))(inputA)
    for i in range(model_params['count_conv1D']):
        x2 = Conv1D(filters=model_params['conv1D_1_filters'][i], kernel_size=model_params['conv1D_1_kernel'][i], activation='relu', name='conv1d_A2_'+str(i))(x2)
        if(model_params['pool_size'] != 0):
            x2 = MaxPooling1D(pool_size=model_params['pool_size'], name='maxpooling_A2_'+str(i))(x2)
    x2 = Flatten(name='flatten_A2')(x2)
    if(model_params['dropout'] != 0):
        x2 = Dropout(model_params['dropout'], name='dropout_A2')(x2)
    if(model_params['dense_units_branch'] != 0):
        x2 = Dense(model_params['dense_units_branch'], activation="relu", name='dense_A2')(x2)

    # Branche B
    # Couches pour l'entrée des fréquences de mots
    if(model_params["use_embedding"]):
        y1 = Embedding(input_dim=vocab_size, output_dim=model_params['embedding_dim'], input_length=max_len, name='embedding_tokens_B')(inputB)
    else:
        y1 = tf.keras.layers.Reshape((max_len, 1))(inputB)
    for i in range(model_params['count_conv1D']):
        y1 = Conv1D(filters=model_params['conv1D_1_filters'][i], kernel_size=model_params['conv1D_1_kernel'][i], activation='relu', name='conv1d_B_'+str(i))(y1)
        if(model_params['pool_size'] != 0):
            y1 = MaxPooling1D(pool_size=model_params['pool_size'], name='maxpooling_B_'+str(i))(y1)
    y1 = Flatten(name='flatten_B')(y1)
    if(model_params['dropout'] != 0):
        y1 = Dropout(model_params['dropout'], name='dropout_B')(y1)
    if(model_params['dense_units_branch'] != 0):
        y1 = Dense(model_params['dense_units_branch'], activation="relu", name='dense_B')(y1)

    # Branche C
    # Couches pour l'entrée de la position des mots
    if(model_params["use_embedding"]):
        y2 = Embedding(input_dim=vocab_size, output_dim=model_params['embedding_dim'], input_length=max_len, name='embedding_tokens_C')(inputC)
    else:
        y2 = tf.keras.layers.Reshape((max_len, 1))(inputC)
    for i in range(model_params['count_conv1D']):
        y2 = Conv1D(filters=model_params['conv1D_1_filters'][i], kernel_size=model_params['conv1D_1_kernel'][i], activation='relu', name='conv1d_C_'+str(i))(y2)
        if(model_params['pool_size'] != 0):    
            y2 = MaxPooling1D(pool_size=model_params['pool_size'], name='maxpooling_C_'+str(i))(y2)
    y2 = Flatten(name='flatten_C')(y2)
    if(model_params['dropout'] != 0):
        y2 = Dropout(model_params['dropout'], name='dropout_C')(y2)
    if(model_params['dense_units_branch'] != 0):
        y2 = Dense(model_params['dense_units_branch'], activation="relu", name='dense_C')(y2)

    # Branche D
    # Couches pour l'entrée de la taille des mots
    if(model_params["use_embedding"]):
        y3 = Embedding(input_dim=vocab_size, output_dim=model_params['embedding_dim'], input_length=max_len, name='embedding_tokens_D')(inputD)
    else:
        y3 = tf.keras.layers.Reshape((max_len, 1))(inputD)
    for i in range(model_params['count_conv1D']):
        y3 = Conv1D(filters=model_params['conv1D_1_filters'][i], kernel_size=model_params['conv1D_1_kernel'][i], activation='relu', name='conv1d_D_'+str(i))(y3)
        if(model_params['pool_size'] != 0):
            y3 = MaxPooling1D(pool_size=model_params['pool_size'], name='maxpooling_D_'+str(i))(y3)
    y3 = Flatten(name='flatten_D')(y3)
    if(model_params['dropout'] != 0):
        y3 = Dropout(model_params['dropout'], name='dropout_D')(y3)
    if(model_params['dense_units_branch'] != 0):
        y3 = Dense(model_params['dense_units_branch'], activation="relu", name='dense_D')(y3)
    
    branches = []
    if 'A' in model_params['included_branches']:
        branches.append(x1)
    if 'A2' in model_params['included_branches']:
        branches.append(x2)
    if 'B' in model_params['included_branches']:
        branches.append(y1)
    if 'C' in model_params['included_branches']:
        branches.append(y2)
    if 'D' in model_params['included_branches']:
        branches.append(y3)

    if len(branches) > 1:
        # Concaténer les sorties de toutes les branches
        z = concatenate(branches, name='concatenated_features')
    else:
        z = branches[0]

    # Couches denses pour les sorties combinées
    if(model_params['dense_units'] != 0):
        z = Dense(model_params['dense_units'], activation="relu", name='dense_combined')(z)
        
    z = Dense(1, activation="linear", name='output')(z)

    # Combinez les entrées et les sorties en un modèle
    model = Model(inputs=[inputA, inputB, inputC, inputD], outputs=z)
    
    return model

from keras import backend as K

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

from keras.callbacks import LearningRateScheduler

# Fonction pour réduire le learning rate après un certain nombre d'époques
def lr_schedule(epoch):
    initial_lr = 0.001
    drop_rate = 1
    epochs_drop = 5
    return initial_lr * (drop_rate ** (epoch // epochs_drop))


from keras.callbacks import ModelCheckpoint
from keras.models import load_model
def train_model(model_params,train,y_train,val,y_val,max_len,token_number):
    utils.set_random_seed(420)

    # Création du modèle en utilisant l'architecture spécifiée
    model = define_multi_model(max_len, token_number,model_params)

    # Compilation du modèle avec l'optimiseur Adam, la fonction de perte 'mse' et la métrique 'mae'
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae',accuracy_with_tolerance(25)])
    # Affichage de la structure du modèle
    model.summary()

    # Création du callback EarlyStopping pour surveiller la perte d'entraînement
    early_stopping = EarlyStopping(monitor='loss', patience=5)

    # Création du callback EarlyStopping pour surveiller la perte de validation
    early_stopping_val = EarlyStopping(monitor='val_loss', patience=5)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

    # Entraînement du modèle avec les données d'entraînement et de validation, en utilisant EarlyStopping pour éviter le surapprentissage
    history = model.fit(train, y_train, epochs=100, batch_size=64, validation_data=(val, y_val), callbacks=[early_stopping, early_stopping_val, lr_scheduler, checkpoint])

    return load_model('best_model.h5'),history

def add_data_to_csv(max_len,mae_test,loss_test,accuracy_within_tolerance_sequence,accuracy_within_tolerance_book,
                    model_params,model_name,csv_results_name):
    # Charger les données à partir du fichier CSV
    df_models = pd.read_csv(f'csv/{csv_results_name}.csv')
    nouvelle_ligne = [model_name,
                    max_len,
                    mae_test,
                    loss_test,
                    accuracy_within_tolerance_sequence,
                    accuracy_within_tolerance_book,
                    model_params['included_branches'],
                    model_params['embedding_dim'],
                    model_params['count_conv1D'],
                    model_params['conv1D_1_filters'],
                    model_params['conv1D_1_kernel'],
                    model_params['pool_size'],
                    model_params['dense_units_branch'],
                    model_params['dense_units'],
                    model_params['dropout']]

    # Ajouter la nouvelle ligne à la DataFrame
    df_models.loc[len(df_models)] = nouvelle_ligne

    df_models.to_csv(f'csv/{csv_results_name}.csv', index=False)

    # Afficher les données mises à jour
    print(df_models)

def evaluate_model_loss_mae(model,test,y_test,train,y_train,val,y_val):
    # Évaluer le modèle sur les données de test
    print("Évaluation sur les données de test :")
    loss_test, mae_test,accuracy_test = model.evaluate(test, y_test)
    print()

    # Évaluer le modèle sur les données d'entraînement
    print("Évaluation sur les données d'entraînement :")
    loss_train, mae_train,accuracy_train = model.evaluate(train, y_train)
    print()

    # Évaluer le modèle sur les données de validation
    print("Évaluation sur les données de validation :")
    loss_val, mae_val,accuracy_validation = model.evaluate(val, y_val)
    print()

    return loss_test, mae_test,loss_train, mae_train,loss_val, mae_val

def evaluate_model_sequence_accuracy(model,test,y_test):
    # Prédiction des dates
    y_pred = model.predict(test)

    error_samples = []  # Liste pour stocker les échantillons en dehors de la tolérance
    tolerance = 25  # Tolérance définie pour considérer les prédictions comme correctes
    within_tolerance_count = 0  # Compteur pour les prédictions dans la tolérance

    # Parcours de chaque échantillon dans les données de test
    for i in range(len(y_test)):
        # Vérification si la différence entre la valeur prédite et la valeur réelle est dans la tolérance
        if abs(round(y_pred[i][0]) - y_test[i]) <= tolerance:
            within_tolerance_count += 1

    # Calcul de l'exactitude dans la tolérance
    accuracy_within_tolerance_sequence = (within_tolerance_count / len(y_test)) * 100

    print("Accuracy within tolerance:", accuracy_within_tolerance_sequence, "%")

    return accuracy_within_tolerance_sequence

import statistics
def evaluate_model_book_accuracy(model,test,y_test,sequence_count_book_test):

    # Prédiction des dates
    y_pred = model.predict(test)  
    y_pred = y_pred.flatten()  

    current = 0
    diff_total = 0

    predicted_book_date = np.zeros(len(sequence_count_book_test)) # Liste pour stocker les dates de prédites pour chaque livre
    true_book_date = np.zeros(len(sequence_count_book_test))  # Liste pour stocker les dates réelles

    ind = 0
    # Parcours de toutes les séquences
    for nb in sequence_count_book_test: 
        dates = []  # Liste pour stocker les dates prédites pour chaque séquence

        # Collecte des prédictions pour chaque séquence
        for i in range(nb):
            dates.append(y_pred[current])  # Ajouter la prédiction actuelle à la liste des dates
            current += 1  # Incrémenter l'indice pour la prochaine prédiction

        # Retirer les valeurs extrêmes
        if len(dates) > 2:
            dates.remove(min(dates))  # Supprime la valeur minimale
            dates.remove(max(dates))  # Supprime la valeur maximale

        # Calculer la moyenne des dates prédites (sans les valeurs extrêmes)
        date = sum(dates) / len(dates) if dates else 0
        # date = statistics.median(dates)
        # Mettre à jour les tableaux avec la date moyenne prédite et la vérité terrain
        predicted_book_date[ind] = date
        true_book_date[ind] = y_test[current - 1]

        # Calculer la différence absolue entre la moyenne prédite et la vérité terrain
        diff = abs(date - y_test[current - 1])
        diff_total += diff

        ind += 1

    # Calculer la MAE pour les livres
    mae_book = diff_total / len(sequence_count_book_test)

    print("Evaluation on Test Data, MAE (Mean Absolute Error):", mae_book)

    # Prédiction des dates
    y_pred = model.predict(test)

    tolerance = 25  # Tolérance définie pour considérer les prédictions comme correctes
    within_tolerance_count = 0  # Compteur pour les prédictions dans la tolérance

    for i in range(len(predicted_book_date)):
        # Vérification si la différence entre la date réelle et la date prédite est dans la tolérance
        if abs(round(true_book_date[i]) - predicted_book_date[i]) <= tolerance:
            within_tolerance_count += 1

    # Calcul de l'exactitude dans la tolérance
    accuracy_within_tolerance_book = (within_tolerance_count / len(predicted_book_date)) * 100

    print("Accuracy within tolerance:", accuracy_within_tolerance_book, "%")
    
    return accuracy_within_tolerance_book,predicted_book_date,true_book_date

def save_model_plot(history,model,test,y_test,save_dir,predicted_book_date,true_book_date,df):
    # Plot training & validation accuracy with tolerance 25
    plt.plot(history.history['accuracy'][0:])
    plt.plot(history.history['val_accuracy'][0:])
    plt.title('Model Accuracy Tolerance 25')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f'{save_dir}/accuracy_plot.png')  # Sauvegarde le graphique au lieu de l'afficher
    plt.close()  # Ferme le graphique pour éviter des conflits avec des tracés ultérieurs

    # Plot training & validation loss values
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('Évolution de la loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f'{save_dir}/loss_plot.png')  # Sauvegarde le graphique
    plt.close()

    # Plot Mean Absolute Error (MAE)
    plt.plot(history.history['mae'][1:]) 
    plt.plot(history.history['val_mae'][1:]) 
    plt.title('Évolution du MAE') 
    plt.ylabel('MAE') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Validation'], loc='upper right') 
    plt.savefig(f'{save_dir}/mae_plot.png')  # Sauvegarde le graphique
    plt.close()

    #Calcul de la différence entre les années prédites et les années réelles
    y_pred = model.predict(test) 
    diff = np.abs(y_test - y_pred.flatten())
    diff2 = np.abs(predicted_book_date - true_book_date)

    # Calcul des limites maximales des différences pour les deux jeux de données
    max_diff = max(np.max(diff), np.max(diff2))

    # Plot de l'histogramme des différences entre années prédites et réelles pour la séquence
    plt.hist(diff, bins=20, color='skyblue', edgecolor='black') 
    plt.xlabel('Difference entre années prédites et réelles') 
    plt.ylabel('Nombre de prédictions') 
    plt.title('Répartition des différences entre les années prédites et réelles (séquence)') 
    plt.xlim(0, max_diff)
    plt.savefig(f'{save_dir}/diff_histogram_sequence.png')  # Sauvegarde le graphique
    plt.close()

    # Plot de l'histogramme des différences entre années prédites et réelles pour le livre
    plt.hist(diff2, bins=20, color='orange', edgecolor='black') 
    plt.xlabel('Difference entre années prédites et réelles') 
    plt.ylabel('Nombre de prédictions') 
    plt.title('Répartition des différences entre les années prédites et réelles (livre)') 
    plt.xlim(0, max_diff)
    plt.savefig(f'{save_dir}/diff_histogram_book.png')  # Sauvegarde le graphique
    plt.close()

    #------------------------voir quelle année est la plus dure (tolerance de 15)---------------------------#
    error_samples = []
    tolerance = 25

    for i in range(len(y_test)):
        if abs(round(y_pred[i][0]) - y_test[i]) > tolerance:
            error_samples.append(y_test[i])

    # Histogramme des années d'erreur
    plt.hist(error_samples, bins=max(y_test)-min(y_test)+1, color='skyblue')
    plt.xlabel('Année')
    plt.ylabel('Nombre d\'échantillons mal prédits')
    plt.title('Histogramme des années d\'erreur')
    plt.savefig(f'{save_dir}/error_year_histogram.png')  # Sauvegarde le graphique
    plt.close()

    #------------------------Utilisation des dates reels et des dates prédites par livre---------------------------#
    # Scatter plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='Year', y='Predicted')
    plt.xlabel("Année Réelle")
    plt.ylabel("Année Prédite")
    plt.title("Comparaison des Années Réelles et Prédites")
    plt.savefig(f'{save_dir}/scatter_plot_book.png')  # Sauvegarde le graphique
    plt.close()

    # Création d'un DataFrame séparé pour les lignes avec 25 de plus et de moins
    df_extra = df.copy()
    df_extra['Year_plus_25'] = df_extra['Year'] + 25
    df_extra['Year_minus_25'] = df_extra['Year'] - 25

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x='Year', y='Predicted', label='Prédite')
    sns.lineplot(data=df, x='Year', y='Year', label='Réelle', linestyle='--', color='red')
    # Lignes avec Year + 25
    sns.lineplot(data=df_extra, x='Year_plus_25', y='Year', label='Réelle + 25', linestyle=':', color='green')
    # Lignes avec Year - 25
    sns.lineplot(data=df_extra, x='Year_minus_25', y='Year', label='Réelle - 25', linestyle=':', color='green')
    plt.xlabel("Année")
    plt.ylabel("Valeurs")
    plt.legend()
    plt.title("Tendance des Années Réelles et Prédites")
    plt.savefig(f'{save_dir}/line_plot_book.png')  # Sauvegarde le graphique
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='Difference', kde=True)
    plt.xlabel("Écart")
    plt.ylabel("Fréquence")
    plt.title("Distribution des Écarts entre Années Réelles et Prédites")
    plt.savefig(f'{save_dir}/histogram_of_errors_book.png')  # Sauvegarde le graphique
    plt.close()

    # Définir une tolérance de 25
    tolerance = 25

    # Identifier si les prédictions sont correctes ou incorrectes
    df['Correct'] = abs(df['Difference']) <= tolerance

    # Compter les prédictions correctes et incorrectes
    correction_counts = df['Correct'].value_counts()

    # Graphique en barres montrant les prédictions correctes et incorrectes
    plt.figure(figsize=(6, 4))
    sns.barplot(x=correction_counts.index, y=correction_counts.values)
    plt.xticks(ticks=[0, 1], labels=["Incorrecte", "Correcte"])
    plt.ylabel("Nombre de Prédictions")
    plt.title("Prédictions Correctes et Incorrectes avec une Tolérance de 25")
    plt.savefig(f'{save_dir}/correct_and_incorrect_predictions_book.png')  # Sauvegarde le graphique
    plt.close()


    # Identifiez les prédictions incorrectes
    df['Incorrect'] = abs(df['Difference']) > tolerance

    # Comptez le nombre de livres incorrects par année
    mal_predits_par_annee = df[df['Incorrect']].groupby('Year').size()

    # Créez un graphique montrant les livres mal prédits par année
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mal_predits_par_annee.index, y=mal_predits_par_annee.values)
    plt.xlabel("Année")
    plt.ylabel("Nombre de Livres Mal Prédits")
    plt.title("Nombre de Livres Mal Prédits par Année (Tolérance de 25)")
    plt.savefig(f'{save_dir}/mis_predicted_books_by_year.png')  # Sauvegarde le graphique
    plt.close()

def create_csv_data(csv_name):
    # Créez un DataFrame vide avec les noms des colonnes, mais sans lignes initiales
    df_models = pd.DataFrame(columns=[
        'Model Name',
        'Sequence Size',
        'MAE',
        'MSE',
        'Sequence Accuracy (toler. 25)',
        'Book Accuracy (toler. 25)'
    ])

    # Ajoutez des colonnes correspondant aux paramètres de votre modèle
    model_params = {
        'included_branches': 'A,C,D',
        'embedding_dim': 40,
        'count_conv1D': 5,
        'conv1D_1_filters': [67, 33, 17],
        'conv1D_1_kernel': 3,
        'pool_size': 2,
        'dense_units_branch': 32,
        'dense_units': 64,
        'dropout': 0
    }

    # Ajoutez ces colonnes au DataFrame
    for key in model_params.keys():
        if key not in df_models.columns:
            df_models[key] = None  # Créez des colonnes sans valeurs initiales

    # Sauvegardez le DataFrame en CSV (sans index)
    df_models.to_csv(f'csv/{csv_name}.csv', index=False)

    print(df_models)