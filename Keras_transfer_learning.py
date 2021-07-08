#!/usr/bin/env python
# coding: utf-8

# Baseado em https://www.pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/
# 
# 

# # Importa os pré-requisitos

# In[ ]:


# import the necessary packages
import os
# import the necessary packages
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


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# # Define variáveis do ambiente

# In[ ]:


#Limite das probabilidades - se for menor então considera que o objeto não está na imagem
THRESHOLD = 0.9

# Limite acima do qual considera que as caixas sobrepostas são o mesmo objeto
NMS_THRESHOLD = 0.15

#Número de classes
NRCLASSES = 15

#Nome da pasta
DATA_PATH="objetos-mesa"

#Dimensões da imagem que deve ser passada para a rede
INPUT_DIMS = (224, 224)
# define the path to the output model and label binarizer
MODEL_PATH = "object_detector.h5"
ENCODER_PATH = "label_encoder.pickle"
# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99

#Variáveis para o treimanento
#Learning rate, qtdade de épocas e batch size
INIT_LR = 1e-4
EPOCHS = 100
BS = 16

MAX_PROPOSALS_INFER = 20


# # Monta o google drive

# In[ ]:


#Mount gdrive
from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)


# In[ ]:


get_ipython().system("ln -s gdrive/'My Drive'/'Object Detection Dataset'/'objetos-mesa' .")


# In[ ]:


get_ipython().system('ls')


# # Carrega as imagens

# In[ ]:


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


print(labels[0])


# In[ ]:


data[0].shape


# In[ ]:


plt.imshow(data[0])


# In[ ]:


for i in range(1,8):
  #print(i)
  print(labels[i])
  plt.imshow(data[i])


# In[ ]:


# convert the data and labels to NumPy arrays
data_np = np.array(data, dtype="float32")
data_np.shape
labels_np = np.array(labels)
labels_np.shape
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels_b = lb.fit_transform(labels_np)
#Descomente a linha abaixo se o seu modelo só tem dois tipos de saída
#labels_b = to_categorical(labels_b)


# In[ ]:


data_np.shape


# In[ ]:


labels_b.shape


# # Prepara o modelo

# In[ ]:



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

# In[ ]:



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

# In[ ]:


# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


# In[ ]:


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
model.save("modelo", save_format="h5")
# serialize the label encoder to disk
print("[INFO] saving label encoder...")
f = open("encoder", "wb")
f.write(pickle.dumps(lb))
f.close()


# In[ ]:


get_ipython().system('ls')


# In[ ]:


#TODO salvar o modelo com o nome do dia


# In[ ]:


get_ipython().system('cp modelo objetos-mesa/modelo-27abr2021')
get_ipython().system('cp encoder objetos-mesa/encoder-27abr2021')


# In[ ]:


get_ipython().system('ls objetos-mesa/')


# # Carrega o modelo
# 
# Essa parte e a seguinte podem ser separadas em um programa a parte quando for usado em produção e/ou uma competição.
# Como o treinamento está sendo feito logo acima não existe realmente necessidade de carregar o modelo.

# Execute a célula abaixo se quiser carregar um modelo antigo

# In[ ]:


get_ipython().system('cp objetos-mesa/modelo-30mar2021 modelo')
get_ipython().system('cp objetos-mesa/encoder-30mar2021 encoder')


# In[ ]:


get_ipython().system('cat encoder')


# In[ ]:


model = load_model("modelo")
lb = pickle.loads(open("encoder", "rb").read())


# In[ ]:


print(type(lb))


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls -lah objetos-mesa/*')


# In[ ]:


get_ipython().system('md5sum objetos-mesa/encoder*')


# # Carrega uma imagem como exemplo e faz a predição

# In[ ]:


#banana
#test_filename = 'IMG_20201109_195321'

test_filename = 'IMG_20201024_195842'

#pao e detergente
#test_filename = 'IMG_20201024_200717'

#test_filename = 'IMG_20201024_201140'
test_img = load_img('/content/objetos-mesa/'+test_filename+'.jpg', target_size=INPUT_DIMS)


# In[ ]:


test_img


# In[ ]:


# Gera um vetor de entradas no qual será realizada a predição
test_img = img_to_array(test_img)
test_img = preprocess_input(test_img)
input_test = []

# TODO o ideal é preencher o vetor de entrada com porções da imagem
# assim podemos saber melhor onde estão localizados os objetos
# Sugestão: implementar o algoritmo de janela deslizante sobre a imagem
input_test.append(test_img)
input_test_np = np.array(input_test, dtype="float32")

# Salva as probabilidades para essa entrada em proba
proba = model.predict(input_test_np)


# In[ ]:


proba


# In[ ]:


lb.classes_


# In[ ]:


#Para o primeira imagem (posição 0) do vetor de entrada mostra
#as probabilidades caso elas sejam maiores do que o limite.
for prob_i, label_i in zip(proba[0], lb.classes_):
  if (prob_i > THRESHOLD):
    print(label_i, " ", prob_i)


# Até aqui ele está tentando usar a imagem inteira como se tivesse um único objeto. Aparentemente ele dá o resultado referente ao maior objeto da imagem.
# 
# Entretanto na verdade é preciso adaptar o código para detectar vários objetos.

# # Até aqui parece estar funcionando 
# 
# A partir daqui parece que tem um ou mais bugs no código. Proceda com cautela :P

# # Testando com pedaços da imagem
# Testando a detecção usando as partes da imagem que foram usadas no treinamento. Não é o ideal mas serve como um teste.
# Parece que passando um pedaço da imagem ele detecta corretamente o objeto. O problema então seria descobrir onde estão os objetos na imagem.

# In[ ]:


#Dado um vetor de probabilidades e os rótulos disponíveis retorna dois valores:
# a maior probabilidade e o respectivo rótulo
def max_prob_label(proba, labels):
  max_prob = 0
  max_label = ''

  for prob_i, label_i in zip(proba, labels):
    if (prob_i > max_prob):
      max_prob = prob_i
      max_label = label_i
  return [max_prob, max_label]


# In[ ]:


data_np = np.array(data, dtype="float32")
data_np.shape


# In[ ]:





# In[ ]:


prob_test = model.predict(data_np)


# In[ ]:


i = 4


# In[ ]:


plt.imshow(data[i])


# In[ ]:


lb.classes_


# In[ ]:


prob_test[i]


# In[ ]:


max_prob_label(prob_test[i], lb.classes_)


# # Seletive search

# O objetivo do algoritmo de SeletiveSearch é tentar detectar elementos de uma imagem que pode ser que sejam objetos, para isso ele usar mudanças de gradiante, variações de cores, etc. Isso é interessante pois é mais eficiente do que percorrer a imagem e tentar detectar objetos em toda a imagem (algoritmo de janela deslizante). Ao final então ele gera um conjunto de retângulos que possivelmente englobam objetos.

# In[ ]:


ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()


# In[ ]:


rects.shape


# In[ ]:


rects[0]


# In[ ]:


type(rects)


# In[ ]:


#Mostra  a imagem original
plt.imshow(image)


# In[ ]:


#Pega um dos retângulos candidatos e mostra como ficou essa parte da imagem
# O tamanho máximo de i é dado na célula acima --> rects.shape
i = 200
[x,y,w,h]=rects[i]
roi = image[y:y + h, x:x + w]
#roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
roi = cv2.resize(roi, INPUT_DIMS,	interpolation=cv2.INTER_CUBIC)
plt.imshow(roi)


# In[ ]:


# initialize the list of region proposals that we'll be classifying
# along with their associated bounding boxes
proposals = []
boxes = []
count = 0
# loop over the region proposal bounding box coordinates generated by
# running selective search
for rect in rects:
	count = count + 1
	# extract the region from the input image, convert it from BGR to
	# RGB channel ordering, and then resize it to the required input
	# dimensions of our trained CNN
	

	[x, y, w, h] = rect
	roi = image[y:y + h, x:x + w]
	#roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	roi = cv2.resize(roi, INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
	# further preprocess the ROI
	#roi = img_to_array(roi)
	#roi = preprocess_input(roi)
	# update our proposals and bounding boxes lists
	proposals.append(roi)
	boxes.append((x, y, x + w, y + h))


# In[ ]:


proposals


# Pega as partes da imagem geradas na etapa anterior e para cada uma gera um vetor de probabilidades de quais objetos estão naquela parte da imagem.

# In[ ]:



proposals_np = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")
print("[INFO] proposal shape: {}".format(proposals_np.shape))
# classify each of the proposal ROIs using fine-tuned model
print("[INFO] classifying proposals...")
proba = model.predict(proposals_np)


# In[ ]:





# Pega um dos objetos e tenta ver qual objeto tem mais chance de estar aparecendo na parte da imagem

# In[ ]:


i=1
plt.imshow(proposals[i])


# In[ ]:


[tmp_prob, tmp_label] = max_prob_label(proba[i], lb.classes_)


# In[ ]:


[tmp_prob, tmp_label]


# In[ ]:


proba


# In[ ]:


scores = []
labels = []
for prob_i in proba:
  [tmp_prob, tmp_label] = max_prob_label(prob_i, lb.classes_)
  scores.append(tmp_prob)
  labels.append(tmp_label)


# In[ ]:


scores


# In[ ]:


from google.colab.patches import cv2_imshow
# clone the original image so that we can draw on it
clone = image.copy()
i = 0
# loop over the bounding boxes and associated probabilities
for (box, prob) in zip(boxes, proba):
	# draw the bounding box, label, and probability on the image
	(startX, startY, endX, endY) = box
	i = i + 1
	
	y = startY - 10 if startY - 10 > 10 else startY + 10

	area = (endY - startY) * (endX - startX)
	#text= "Raccoon: {:.2f}%".format(prob * 100)
	#text="teste"
	[tmp_prob, tmp_label] = max_prob_label(prob, lb.classes_)
	text = tmp_label
	#print(tmp_prob)
	if (tmp_prob > 0.5 and area < 500):
	  print("index: ", i, ", prob: ", tmp_prob, ", label: ", tmp_label, ", area: ", area)
	  cv2.rectangle(clone, (startX, startY), (endX, endY),
		(0, 255, 0), 2)
	  cv2.putText(clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

#clone = img_to_array(clone)
#clone = preprocess_input(clone)


# show the output after *before* running NMS
#imshow(clone)
plt.imshow(clone)


# In[ ]:





# In[ ]:


#Código original dessa célula por Coutinho
def NMS(B, S, L, Nt, soft=False):
  D  = []
  Sd = []
  Ld = []

  B = list(B)
  S = list(S)
  L = list(L)

  while len(B) != 0:
    #B_S = list(zip(B, S))
    max_index = S.index(max(S))
    m = S[max_index] #sorted(B_S, key= lambda x: x[1],reverse=True)[0][1]
    M = B[max_index] #sorted(B_S, key= lambda x: x[1],reverse=True)[0][0]
    l = L[max_index]
    D.append(M)
    B.pop(max_index)
    Sd.append(m)
    S.pop(max_index)
    Ld.append(l)
    L.pop(max_index)
    index = -1
    for b in B:
      #index = B.index(b)
      if len(B) == 0:
        break;
      index = index + 1
      #print("index=",index)
      #print("M=",M)
      #print("b=",b)
      if soft:
        S[index] -= IOU(M['box'],b)['box']
      else:
        #if IOU(M['box'],b['box']) > Nt:
        if IOU(M, b) > Nt:
          B.pop(index)
          S.pop(index)
          L.pop(index)

  return D, Sd, Ld

def IOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
  # boxA= (boxA[0],boxA[1], boxA[0] + width, boxA[1] + height)
  # boxB= (boxB[0],boxB[1], boxB[0] + width, boxB[1] + height)
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
  # compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)
  # return the intersection over union value
  return iou
     


# In[ ]:


#NMS_THRESHOLD = 0.15
#THRESHOLD = 0.9


# In[ ]:


selected_boxes, selected_scores, selected_labels = NMS(boxes, scores, labels, NMS_THRESHOLD)


# In[ ]:


print(len(selected_boxes))


# In[ ]:


clone2 = image.copy()
#selected_boxes = non_max_suppression_slow(boxes, 0.8)
# loop over the bounding box indexes
for tmp_box, tmp_score, tmp_label in zip(selected_boxes, selected_scores, selected_labels):
	#print(tmp_box, tmp_score)
	#continue
	# draw the bounding box, label, and probability on the image
	(startX, startY, endX, endY) = tmp_box
	if tmp_score < THRESHOLD:
		continue
	cv2.rectangle(clone2, (startX, startY), (endX, endY),
		(0, 255, 0), 1)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text= tmp_label + " {:.0f}%".format(tmp_score * 100)
	print(text)
	cv2.putText(clone2, text, (startX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
# show the output image *after* running NMS
#cv2.imshow("After NMS", image)
plt.imshow(clone2)


# In[ ]:


clone_resized = cv2.resize(clone2, (800,800))


# In[ ]:


cv2.imwrite('saida.jpg', clone_resized, [cv2.IMWRITE_JPEG_QUALITY, 100])


# In[ ]:


get_ipython().system('ls')


# In[ ]:


plt.imshow(clone_resized)


# Não execute código abaixo dessa linha ... a princípio não está funcionando

# In[ ]:


def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
    		# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]
			#print(j)
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
	# return only the bounding boxes that were picked
	return boxes[pick]
	#return idxs


# In[ ]:


boxes.shape


# In[ ]:


boxIdxs = non_max_suppression_slow(boxes, 0.3)


# In[ ]:





# In[ ]:


print(boxIdxs)


# In[ ]:


def find_labels(boxIdxs, boxes, proba):
  tmp_labels = []
  for b_i in boxIdxs:
    #print(b_i)
    for b_j, prob_j in zip(boxes, proba):
      if (b_i[0] == b_j[0] and b_i[1] == b_j[1]  and b_i[2] == b_j[2]  and b_i[3] == b_j[3]):
        tmp_label = max_prob_label(prob_j, lb.classes_)
        #print(tmp_label)
        tmp_labels.append(tmp_label)
        break
  return tmp_labels


# In[ ]:


print(lb.classes_)


# In[ ]:


print(proba[3])


# In[ ]:


labels_after_nms = find_labels(boxIdxs, boxes, proba)


# In[ ]:


labels_after_nms

