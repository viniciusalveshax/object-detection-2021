#!/usr/bin/env python
# coding: utf-8

# Baseado em https://www.pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/
# 
# 

# # Importa os pré-requisitos

# In[ ]:

import os
from bs4 import BeautifulSoup
from imutils import paths
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from datetime import datetime


# In[ ]:



# # Define variáveis do ambiente

# In[ ]:


#Limite das probabilidades - se for menor então considera que o objeto não está na imagem
THRESHOLD = 0.9

# Limite acima do qual considera que as caixas sobrepostas são o mesmo objeto
NMS_THRESHOLD = 0.15

#Número de classes
NRCLASSES = 15

#Dimensões da imagem que deve ser passada para a rede
INPUT_DIMS = (224, 224)
# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99

#Variáveis para o treimanento
#Learning rate, qtdade de épocas e batch size
INIT_LR = 1e-4
EPOCHS = 100
BS = 16

MAX_PROPOSALS_INFER = 20

#Nome da pasta
DATA_PATH="objetos-mesa"
COLLAB=False
DEBUG=True

# Prefixo do arquivo do modelo
PREFIX = "object_detector.h5"
PREFIX = PREFIX + datetime.today().strftime('%Y-%m-%d')

# # Monta o google drive

# In[ ]:

# Se está usando o collab então monta o drive
if COLLAB:
	#Mount gdrive
	from google.colab import drive
	drive.mount("/content/gdrive", force_remount=True)


# In[ ]:


get_ipython().system("ln -s gdrive/'My Drive'/'Object Detection Dataset'/'objetos-mesa' .")



#Inicializa os vetores que vão ser usados no treinamento
data = []
labels = []
image_count = 0

#Lê o arquivo index.txt que tem os prefixos das imagens e das anotações
f = open(DATA_PATH + os.path.sep + 'index.txt')
for linha in f:
  #Tira o \n no final de cada linha
	linha = linha[:-1]  
	print("[INFO] processing image ", linha)
	# extract the filename from the file path and use it to derive
	# the path to the XML annotation file

  #nome do arquivo da imagem
	imgFilename = DATA_PATH + os.path.sep + linha + '.jpg'

  # para fazer testes em um conjunto menor de imagens 
  # descomente as linhas a seguir pois isso irá carregar
  # um conjunto menor de imagens (6 somente)
	#image_count = image_count + 1
	#if (image_count > 5):
	#	break

  #nome do arquivo da anotação
	annotFilename = DATA_PATH + os.path.sep + linha + '.xml'
	
	# load the annotation file, build the soup, and initialize our
	# list of ground-truth bounding boxes
	contents = open(annotFilename).read()
	soup = BeautifulSoup(contents, "html.parser")

	# extract the image dimensions
	w = int(soup.find("width").string)
	h = int(soup.find("height").string)
	
	#continue

	# loop over all 'object' elements
	for o in soup.find_all("object"):
		# extract the label and bounding box coordinates
		label = o.find("name").string
		xMin = int(o.find("xmin").string)
		yMin = int(o.find("ymin").string)
		xMax = int(o.find("xmax").string)
		yMax = int(o.find("ymax").string)
		# truncate any bounding box coordinates that may fall
		# outside the boundaries of the image
		xMin = max(0, xMin)
		yMin = max(0, yMin)
		xMax = min(w, xMax)
		yMax = min(h, yMax)
		
  	# load the input image (224x224) and preprocess it
		image = load_img(imgFilename, target_size=INPUT_DIMS)
		image = img_to_array(image)
		image = preprocess_input(image) # 
  
    # a imagem é lida em 224x224 então é preciso calcular as dimensões
    # dos objetos anotados dentro dessa nova proporção de imagem
		ratio_w0 = round(w / INPUT_DIMS[0])
		ratio_h0 = round(h / INPUT_DIMS[1])
		xMin0 = round(xMin / ratio_w0)
		xMax0 = round(xMax / ratio_w0)
		yMin0 = round(yMin / ratio_h0)
		yMax0 = round(yMax / ratio_h0) 

    # seleciona a parte da imagem onde o objeto anotado está
		obj_image = image[yMin0:yMax0, xMin0:xMax0]

    # a entrada da rede espera uma imagem 224x224 então redimensiona o objeto
		obj_resized = resize(obj_image, (INPUT_DIMS[0], INPUT_DIMS[1]))
		
		data.append(obj_resized)
		labels.append(label)


# In[ ]:

if DEBUG:
	print(labels[0])
	data[0].shape
	plt.imshow(data[0])
	for i in range(1,8):
	  #print(i)
	  print(labels[i])
	  plt.imshow(data[i])



# convert the data and labels to NumPy arrays
data_np = np.array(data, dtype="float32")
data_np.shape
labels_np = np.array(labels)
labels_np.shape
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels_b = lb.fit_transform(labels_np)

if DEBUG:
	data_np.shape
	labels_b.shape

# # Prepara o modelo


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data_np, labels_b,
	test_size=0.20, stratify=labels, random_state=42)
# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


# In[ ]:


# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(NRCLASSES, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
# TODO Testar com outras camadas não congeladas
for layer in baseModel.layers:
	layer.trainable = False


# # Treina o modelo

#usar variavel de cima
EPOCHS=100
# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)


# # Mostra as informações do treinamento

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
#plt.savefig(args["plot"])
plt.savefig('saida')


# # Salvo o modelo

# In[ ]:


# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(PREFIX+"modelo", save_format="h5")
# serialize the label encoder to disk
print("[INFO] saving label encoder...")
f = open(PREFIX+"encoder", "wb")
f.write(pickle.dumps(lb))
f.close()


