#--------------------------------------------Sauvegarde et charger modele-------------------------------------------------#

from keras.models import load_model

# Enregistrer le modèle CNN
model.save('mon_modele_cnn.h5')

# Pour charger le modèle sauvegardé plus tard
#loaded_model = load_model('mon_modele_cnn.h5')

#------------------------------------------------Affichage des courbes------------------------------------------------------------#

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

#Plot MAE
plt.plot(history.history['mae']) 
plt.plot(history.history['val_mae']) 
plt.title('Model Mean Absolute Error') 
plt.ylabel('MAE') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Validation'], loc='upper right') 
plt.show()

#Calcul de la différence entre les années prédites et les années réelles
y_pred = model.predict(data_test) 
diff = np.abs(y_test - y_pred.flatten())

#le dernier bin
max_diff = np.max(diff)

#Plot de l'histogramme des différences entre les années prédites et réelles
plt.hist(diff, bins=np.arange(0,max_diff,1)) 
plt.xlabel('Difference entre années prédites et réelles') 
plt.ylabel('Nombre de prédictions') 
plt.title('Répartition des différences entre les années prédites et réelles') 
plt.show()


#-----------------------------------voir quelle année est la plus dure----------------------------------#

y_pred = model.predict(data_test)
error_samples = []
for i in range(len(y_test)):
    if round(y_pred[i][0]) != y_test[i]:
        error_samples.append(y_test[i])

#------------------------voir quelle année est la plus dure (tolerance de 15)---------------------------#
error_samples = []
tolerance = 15
for i in range(len(y_test)):
    if abs(round(y_pred[i][0]) - y_test[i]) > tolerance:
        error_samples.append(y_test[i])

# Créer un histogramme des années d'erreur
plt.hist(error_samples, bins=max(y_test)-min(y_test)+1, color='skyblue')
plt.xlabel('Année')
plt.ylabel('Nombre d\'échantillons mal prédits')
plt.title('Histogramme des années d\'erreur')
plt.show()


#------------------- Créer un histogramme pour visualiser la répartition des textes par année---------------#

plt.hist(y, bins=range(min(y), max(y) + 1), alpha=0.5,color='skyblue')
plt.grid(axis='y', linestyle='--')
plt.xlabel('Année')
plt.ylabel('Nombre de textes')
plt.title('Répartition des textes par année')
plt.show()

