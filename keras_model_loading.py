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

def show_image(image):
	cv2.imshow('ImageWindow', image)
	cv2.waitKey()

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

#Código original dessa função por Coutinho
# https://github.com/lucas-coutinho/
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

#Código original dessa função por Coutinho
# https://github.com/lucas-coutinho/
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


def color_from_label(label):
	COLOR_FROM_LABEL = {'Alcohol bottle': (0,0,0),
						'Apple': (50,0,0),
						'Banana': (100,0,0),
						'Beer can': (150,0,0),
						'Carrot': (200,0,0),
						'Chocolate milk': (255,0,0),
						'Chocolate': (0,50,0),
						'Chocolate Powder': (0,100,0),
						'Coconut milk': (0,150,0),
						'Cracker': (0,200,0),
						'Gelatine': (0,255,0),
						'Grape juice': (0,0,50),
						'Ketchup': (0,0,100),
						'Mayonnaise': (0,0,150),
						'Napkin': (0,0,200),
						'Popcorn': (0,0,255),
						'Soap': (50,50,50),
						'Soda can': (100,100,100),
						'Toothpaste': (150,150,150),
						'Yeast': (200,200,200)}

	color = COLOR_FROM_LABEL[label]

	return color

def avaliacao(model, lb, filename, interactive):
	test_img = load_img(DATA_INPUT_PATH + os.path.sep + filename + '.jpg', target_size=INPUT_DIMS)

	#test_img = cv2.imread(DATA_INPUT_PATH + os.path.sep + filename + '.jpg', cv2.COLOR_BGR2RGB)

	# Gera um vetor de entradas no qual será realizada a predição
	test_img = img_to_array(test_img)
	test_img = preprocess_input(test_img)

	original_size_img = load_img(DATA_INPUT_PATH + os.path.sep + filename + '.jpg', target_size=(1000, 1000))
	original_size_img = img_to_array(original_size_img)
	original_size_img = preprocess_input(original_size_img)
	print("ORIGINAL_SIZE: ", original_size_img.shape)
	orig_size_X, orig_size_Y, orig_size_colors = original_size_img.shape
	#orig_size_X = original_size_img[0]
	#orig_size_Y = original_size_img[1]
	print("origx:", orig_size_X, " origy:", orig_size_Y)

	input_test = []

	# TODO o ideal é preencher o vetor de entrada com porções da imagem
	# assim podemos saber melhor onde estão localizados os objetos
	# Sugestão: implementar o algoritmo de janela deslizante sobre a imagem
	input_test.append(test_img)
	input_test_np = np.array(input_test, dtype="float32")

	# Salva as probabilidades para essa entrada em proba
	proba = model.predict(input_test_np)

	#proba
	#lb.classes_


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



	# In[ ]:

	# Era Debug mas troquei por false só pra testar se funcionava sem essa parte do código
	if False:
		data_np = np.array(data, dtype="float32")
		data_np.shape
		prob_test = model.predict(data_np)
		i = 4
		plt.imshow(data[i])
		lb.classes_
		prob_test[i]
		max_prob_label(prob_test[i], lb.classes_)

	# # Seletive search
	# O objetivo do algoritmo de SeletiveSearch é tentar detectar elementos
	# de uma imagem que pode ser que sejam objetos, para isso ele usar mudanças de gradiante,
	# variações de cores, etc.
	# Isso é interessante pois é mais eficiente do que percorrer
	# a imagem e tentar detectar objetos em toda a imagem (algoritmo de janela deslizante).
	# Ao final então ele gera um conjunto de retângulos que possivelmente englobam objetos.

	# In[ ]:


	image = test_img
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)
	ss.switchToSelectiveSearchFast()
	rects = ss.process()


	# In[ ]:
	if DEBUG:
		rects.shape
		rects[0]
		type(rects)
		#Mostra  a imagem original
		if interactive:
			plt.imshow(image)

		#Pega um dos retângulos candidatos e mostra como ficou essa parte da imagem
		# O tamanho máximo de i é dado na célula acima --> rects.shape
		#i = 200
		#[x,y,w,h]=rects[i]
		#roi = image[y:y + h, x:x + w]
		#roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
		#roi = cv2.resize(roi, INPUT_DIMS,	interpolation=cv2.INTER_CUBIC)
		#plt.imshow(roi)


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


	if DEBUG:
		proposals


	# Pega as partes da imagem geradas na etapa anterior e para cada uma gera um vetor de probabilidades de quais objetos estão naquela parte da imagem.

	proposals_np = np.array(proposals, dtype="float32")
	boxes = np.array(boxes, dtype="int32")

	if DEBUG:
		print("[INFO] proposal shape: {}".format(proposals_np.shape))
		print("[INFO] classifying proposals...")
	# classify each of the proposal ROIs using fine-tuned model
	proba = model.predict(proposals_np)



	# Pega um dos objetos e tenta ver qual objeto tem mais chance de estar aparecendo na parte da imagem

	if DEBUG:
		i=1
		if interactive:
			plt.imshow(proposals[i])
		[tmp_prob, tmp_label] = max_prob_label(proba[i], lb.classes_)
		[tmp_prob, tmp_label]

		print("Prob a")
		print(proba)


	# In[ ]:


	scores = []
	labels = []
	for prob_i in proba:
	  [tmp_prob, tmp_label] = max_prob_label(prob_i, lb.classes_)
	  scores.append(tmp_prob)
	  labels.append(tmp_label)

	if DEBUG:
		print("Scores")
		print(scores)

	if COLLAB:
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

		if DEBUG:
			print("Prob: ", tmp_prob, ", Label: ", tmp_label)
		#print(tmp_prob)
		if (tmp_prob > THRESHOLD and area < 500):
			if DEBUG:
				print("index: ", i, ", prob: ", tmp_prob, ", label: ", tmp_label, ", area: ", area)
			cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
			cv2.putText(clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	#clone = img_to_array(clone)
	#clone = preprocess_input(clone)


	# show the output before running NMS
	#imshow(clone)
	if interactive:
		plt.imshow(clone)

	#Aplica o algoritmo de NMS para remover bounding boxes redundantes
	selected_boxes, selected_scores, selected_labels = NMS(boxes, scores, labels, NMS_THRESHOLD)

	if DEBUG:
		print("Length selected_boxes")
		print(len(selected_boxes))


	# In[ ]:

    # Salva um resumo dos resultados para comparar
	resultados = {}

	clone2 = image.copy()
	#selected_boxes = non_max_suppression_slow(boxes, 0.8)
	# loop over the bounding box indexes
	for tmp_box, tmp_score, tmp_label in zip(selected_boxes, selected_scores, selected_labels):
		#print(tmp_box, tmp_score)
		#continue

		bb_color = color_from_label(tmp_label)

		# draw the bounding box, label, and probability on the image
		#print("Print score", tmp_score)
		(startX, startY, endX, endY) = tmp_box
		if tmp_score < THRESHOLD:
			continue
		if DEBUG:
			#print("Mais um retângulo")
			print("Orig size x:", orig_size_X, " y:", orig_size_Y)
			print("Retângulo x: ", startX, "-", endX, "; y: ", startY, "-", endY)
			startX_orig_size = (round) ((startX / 224) * orig_size_X)
			endX_orig_size   = (round) ((endX / 224) * orig_size_X)
			startY_orig_size = (round) ((startY / 224) * orig_size_Y)
			endY_orig_size   = (round) ((endY / 224) * orig_size_Y)
			print("Novo Retângulo x: ", startX_orig_size, "-", endX_orig_size, "; y: ", startY_orig_size, "-", endY_orig_size)

		cv2.rectangle(clone2, (startX, startY), (endX, endY),
			bb_color, 1)
		cv2.rectangle(original_size_img, (startX_orig_size, startY_orig_size), (endX_orig_size, endY_orig_size),
					  bb_color, 10)

		y = startY - 10 if startY - 10 > 10 else startY + 10
		text = tmp_label + " {:.0f}%".format(tmp_score * 100)

		# Salva uma contagem resumida dos resultados
		if resultados.get(tmp_label):
			resultados[tmp_label] = resultados[tmp_label] + 1
		else:
			resultados[tmp_label] = 1

		if DEBUG:
			print(text)
		cv2.putText(clone2, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.3, bb_color, 1)
		cv2.putText(original_size_img, text, (startX_orig_size, startY_orig_size),
			cv2.FONT_HERSHEY_SIMPLEX, 1, bb_color, 5)
	# show the output image *after* running NMS
	#cv2.imshow("After NMS", image)

	clone_resized = cv2.resize(clone2, (800,800))
	#original_size_img_resized = cv2.resize(original_size_img, (800,800))

	original_size_rgb = cv2.cvtColor(original_size_img, cv2.COLOR_BGR2RGB)

	if interactive:
		show_image(clone_resized)
		show_image(original_size_rgb)

#	cv2.imshow('ImageWindow', original_size_img)
#	cv2.waitKey()

	# imwrite precisa receber valores entre 0 e 255 e os valores até o momento estão entre 0 e 1
	cv2.imwrite('output-data'+os.path.sep+filename+'.jpg', 255*original_size_rgb)

	return resultados

def dict_from_xml(test_filename):
	annotFilename = DATA_INPUT_PATH + os.path.sep + test_filename + ".xml"
	contents = open(annotFilename).read()
	soup = BeautifulSoup(contents, "html.parser")

	result_dict = {}

	# loop over all 'object' elements
	for o in soup.find_all("object"):
		# extract the label
		label = o.find("name").string
		if result_dict.get(label):
			result_dict[label] = result_dict[label] + 1
		else:
			result_dict[label] = 1

	return result_dict

def calculate_score(predict_results, xml_labels):
	true_positives = 0
	for key in predict_results.keys():
		if xml_labels.get(key):
			if xml_labels[key] <= predict_results[key]:
				true_positives += xml_labels[key]
				predict_results[key] -= xml_labels[key]
				del xml_labels[key]
				#if predict_results[key] == 0:
				#	del predict_results[key]
			else:
				true_positives += predict_results[key]
				xml_labels[key] -= predict_results[key]
				predict_results[key] = 0
				#del predict_results[key]

	false_positives = 0
	for key in predict_results.keys():
		false_positives += predict_results[key]

	undetecteds = 0
	for key in xml_labels.keys():
		undetecteds += xml_labels[key]

	if DEBUG:
		print("Xml labels", xml_labels)
		print("predição", predict_results)
		print("Positivos verdadeiros", true_positives)
		print("Falsos positivos", false_positives)
		print("Não detectados", undetecteds)

	return true_positives, false_positives, undetecteds


#Limite das probabilidades - se for menor então considera que o objeto não está na imagem
THRESHOLD = 0.90

# Limite acima do qual considera que as caixas sobrepostas são o mesmo objeto
# Quanto menor esse valor menos a tolerância, logo menos sobreposições
NMS_THRESHOLD = 0.2
#Número de classes
NRCLASSES = 20
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
# Prefixo do arquivo do modelo
PREFIX = "object_detector.h52021-09-29"
#PREFIX = PREFIX + datetime.today().strftime('%Y-%m-%d')

#True para rodar no collab
COLLAB=False

#Debug mostra ou não variáveis intermediárias
DEBUG=True

# Se está usando o collab então monta o drive
if COLLAB:
	#Mount gdrive
	from google.colab import drive
	drive.mount("/content/gdrive", force_remount=True)
	get_ipython().system("ln -s gdrive/'My Drive'/'Object Detection Dataset'/'objetos-mesa' .")
else:
	DATA_INPUT_PATH="input-data/imagens-competicao-2020/"
	DATA_MODEL_PATH="model-data/"

LOAD_ALL_IMAGES=False

if LOAD_ALL_IMAGES == True:
	#Inicializa os vetores que vão ser usados no treinamento
	data = []
	labels = []
	image_count = 0

	#Lê o arquivo index.txt que tem os prefixos das imagens e das anotações
	f = open(DATA_INPUT_PATH + os.path.sep + 'index.txt')
	for linha in f:
	  #Tira o \n no final de cada linha
		linha = linha[:-1]
		print("[INFO] processing image ", linha)
		# extract the filename from the file path and use it to derive
		# the path to the XML annotation file

	  #nome do arquivo da imagem
		imgFilename = DATA_INPUT_PATH + os.path.sep + linha + '.jpg'

	  # para fazer testes em um conjunto menor de imagens
	  # descomente as linhas a seguir pois isso irá carregar
	  # um conjunto menor de imagens (6 somente)
		#image_count = image_count + 1
		#if (image_count > 5):
		#	break

	  #nome do arquivo da anotação
		annotFilename = DATA_INPUT_PATH + os.path.sep + linha + '.xml'

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

	#quit()

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

	#quit()

	if DEBUG:
		data_np.shape
		labels_b.shape

	if COLLAB:
		get_ipython().system('ls')
		get_ipython().system('cp modelo objetos-mesa/modelo-27abr2021')
		get_ipython().system('cp encoder objetos-mesa/encoder-27abr2021')
		get_ipython().system('ls objetos-mesa/')

#print(PREFIX+"modelo")

# # Carrega o modelo
# Essa parte e a seguinte podem ser separadas em um programa a parte quando for usado em produção e/ou uma competição.
# Como o treinamento está sendo feito logo acima não existe realmente necessidade de carregar o modelo.
model = load_model(DATA_MODEL_PATH + os.path.sep + PREFIX + "modelo")
lb = pickle.loads(open(DATA_MODEL_PATH + os.path.sep + PREFIX + "encoder", "rb").read())

if DEBUG:
	print(type(lb))

LOAD_ALL_IMAGES=False

if LOAD_ALL_IMAGES == False:

	IMAGE_HAVE_XML=False

	#Arquivo que será testado
	filenames = ['bonus_13','normal_6', 'normal_7', 'normal_13', 'normal_22', 'normal_26', 'normal_31']

	for test_filename in filenames:

		print("Gerando resultado para ",test_filename)

		#Resumo do resultado fica armazenado em um dicionário
		predict_results = avaliacao(model, lb, test_filename, interactive=True)

		#Se a imagem possui um arquivo de anotações
		if IMAGE_HAVE_XML:
			#Rótulos do arquivo xml originais
			xml_labels = dict_from_xml(test_filename)
			true_positives, false_positives, undetecteds = calculate_score(predict_results, xml_labels)
			print("General Score: ", true_positives-false_positives-undetecteds)
		else:
			print(predict_results)

else:

	#Inicializa os vetores que vão ser usados no treinamento
	data = []
	labels = []
	image_count = 0
	sum_score = 0

	#Lê o arquivo index.txt que tem os prefixos das imagens e das anotações
	f = open(DATA_INPUT_PATH + os.path.sep + 'index.txt')
	for linha in f:
		#Tira o \n no final de cada linha
		linha = linha[:-1]
		print("[INFO] processing image ", linha)
		# extract the filename from the file path and use it to derive
		# the path to the XML annotation file

		# Resumo do resultado armazenado em um dicionário
		predict_results = avaliacao(model, lb, linha, interactive=False)
		# Rótulos do arquivo xml originais
		xml_labels = dict_from_xml(linha)
		true_positives, false_positives, undetecteds = calculate_score(predict_results, xml_labels)
		score = true_positives - false_positives - undetecteds
		print("[INFO] General Score: ", score)

		sum_score += score

	print("nr imagens: ", f.count(), " score acumulado: ", sum_score)
