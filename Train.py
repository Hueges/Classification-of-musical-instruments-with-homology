import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from gtda.plotting import plot_point_cloud
from gtda.homology import VietorisRipsPersistence
from ripser import ripser
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler



Heat_diagrams=np.load("Heat_diagrams2.npy")
labele=np.load("Labele.npy2.npy")

# for heat in Heat_diagrams:
#     plt.imshow(heat[1],cmap="jet")
#     plt.show()
#     plt.close()
#     plt.imshow(heat[2],cmap="jet")
#     plt.show()
#     plt.close()

X=Heat_diagrams
Y=labele
#  Heat_diagrams ima (200, 3, 100, 100)
#uzimamo prvu homologiju
X = X[:, 1, :, :]
Y=Y[:]


# Verify the new shape

model = Sequential()
model.add(Flatten(input_shape=(100, 100)))
#model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
scaler=StandardScaler()

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y)

X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_validation = scaler.transform(X_validation.reshape(-1, X_validation.shape[-1])).reshape(X_validation.shape)

# print('here',X_train, X_validation, Y_train, Y_validation)
model.fit(X_train,Y_train,epochs=10)
Y_pred = model.predict(X_validation)
Y_pred = (Y_pred > 0.5).astype(int)
model.evaluate(X_validation,Y_validation)
cm = confusion_matrix(Y_validation, Y_pred)

print(f"f1 score: {f1_score(Y_validation, Y_pred)}")
print(cm)