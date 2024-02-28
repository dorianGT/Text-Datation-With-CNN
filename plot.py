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
