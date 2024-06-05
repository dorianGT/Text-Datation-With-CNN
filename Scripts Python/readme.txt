Étapes d'utilisation

----------------Création des données----------------

Créez les données avant l'entraînement du modèle :

Utilisez la commande :
python creation_data.py

Organisation des dossiers :
Assurez-vous d'avoir les trois dossiers train, validation, et test au même endroit. 
Si vous avez d'autres livres (ou si vous souhaitez réaliser une séparation des données différente),
ajoutez-les manuellement dans les dossiers appropriés.

Cette opération prend en moyenne 15 minutes mais n'est pas à répéter si les données ne change pas.
Si les données ont déjà été créées, cette commande va écraser les anciennes données par les nouvelles.

----------------Entraînement des modèles----------------

Après avoir créé les dictionnaires, entraînez les modèles :

Utilisez la commande :
python train_model.py

Cela entraînera tous les modèles présents dans le fichier model_params.csv.

Définition des paramètres :
Dans le fichier model_params.csv, vous pouvez définir les paramètres des modèles (avec ou sans embedding, taille d'embedding, etc.). 
Il n'y a qu'un seul modèle (notre meilleur modèle) dans le fichier par défaut,
mais vous pouvez ajouter d'autres lignes pour entraîner plusieurs modèles à la suite (sans interruption).

Saisie des informations :
Après avoir entré la commande python train_model.py, 
il vous sera demandé de renseigner le nom du test et l'année minimale à considérer (les textes avec une année inférieure ne seront pas considérés).

Structure des résultats :

Un fichier avec le nom que vous avez renseigné sera créé dans automatic_save, contenant tout le nécessaire pour chaque modèle : le modèle, l'architecture, l'history, les différents graphiques d'analyses, et les résultats dans un fichier CSV.
Un fichier CSV de résultats général pour tous les modèles entraînés pour ce test (avec l'accuracy, MAE, etc., pour chaque modèle) sera créé dans le dossier csv sous le nom csv_results_nomdutest.csv.

----------------Test des modèles obtenus----------------

Essayez les modèles obtenus avec d'autres livres :

Utilisez la commande :
python script_predict.py

Vous devrez renseigner l'emplacement du modèle .h5 et le fichier/dossier des textes.



