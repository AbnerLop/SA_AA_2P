import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle
import random
import keras
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import glorot_normal

DIRECTORIO = "Train"
CATEGORIAS = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
for categoria in CATEGORIAS:
    path = os.path.join(DIRECTORIO, categoria)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()

        break
    break

datos_entrenamiento = []
IMG_SIZE = 70
def crea_datos_entrenamiento():
    for categoria in CATEGORIAS:
        path = os.path.join(DIRECTORIO ,categoria)
        class_num = CATEGORIAS.index(categoria)
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path ,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                datos_entrenamiento.append([new_array, class_num])
            except Exception as e:
                pass


crea_datos_entrenamiento()
#print(len(datos_entrenamiento))

X = []
y = []

random.shuffle(datos_entrenamiento)

for m, clase in datos_entrenamiento:
    X.append(m)
    y.append(clase)

salida = open("X.pickle", "wb")
pickle.dump(X, salida)
salida.close()
salida = open("y.pickle", "wb")
pickle.dump(y, salida)
salida.close()

X = np.array(X)
X = X / 255
X = X.reshape(-1,70,70,1)

modelo = Sequential()
modelo.add(Conv2D(32, (3,3), input_shape=X.shape[1:]))
modelo.add(Activation('relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))
modelo.add(BatchNormalization())
modelo.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
modelo.add(Activation('relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))
modelo.add(Flatten())
modelo.add(Dense(50, activation = 'elu', kernel_initializer=glorot_normal(seed=0)))
modelo.add(Dense(50, activation = 'elu', kernel_initializer=glorot_normal(seed=0)))
modelo.add(Dense(50, activation = 'elu', kernel_initializer=glorot_normal(seed=0)))
modelo.add(Dense(50, activation = 'elu', kernel_initializer=glorot_normal(seed=0)))
modelo.add(Dense(10))
modelo.add(Dense(10))
modelo.add(Dense(10))
modelo.add(Dense(10))
modelo.add(Dense(1))
modelo.add(Activation('sigmoid'))
modelo.summary()

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo.fit(X, y, batch_size=32, epochs=10, validation_split=.25)
modelo.save('modelo.model')


def carga(rutaImg):
    imgSize = 70
    imgArray = cv2.imread(rutaImg, cv2.IMREAD_GRAYSCALE)
    nImgArray = cv2.resize(imgArray, (imgSize, imgSize))
    return nImgArray.reshape(-1, imgSize, imgSize, 1)

modelo = keras.models.load_model('modelo.model')

DIRECTORIOTEST = 'Test'
for categoria in CATEGORIAS:
    path = os.path.join(DIRECTORIOTEST,categoria)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        p = modelo.predict([carga(os.path.join(path,img))])
        print(p)
        print(CATEGORIAS[int(p[0][0])])
