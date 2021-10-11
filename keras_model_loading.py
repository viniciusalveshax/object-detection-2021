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
from IOU import IOU

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
# params B = bounding boxes, S = scores, L = labels, Nt = NMS_THRESHOLD
def NMS(B, S, L, Nt, soft=False):
  Bd  = []
  Sd = []
  Ld = []

  B = list(B)
  S = list(S)
  L = list(L)

  #Enquanto a lista de bboxes não for vazia ... do
  while len(B) != 0:

    if DEBUG:
      print("Tamanho de B", len(B))
      print("Tamanho de S", len(S))


    #B_S = list(zip(B, S))
	#Pega o índice do maior score
    max_index = S.index(max(S))
    if DEBUG:
      print("max_index: ", max_index)
    max_score = S[max_index] #sorted(B_S, key= lambda x: x[1],reverse=True)[0][1]
    max_bbox = B[max_index] #sorted(B_S, key= lambda x: x[1],reverse=True)[0][0]
    max_label = L[max_index]

    if DEBUG:
      print("Selecionando M:", max_bbox)

	# Seleciona a bbox dada por max_index e salva na lista das bboxes que vão continuar
    Bd.append(max_bbox)
    B.pop(max_index)
    Sd.append(max_score)
    S.pop(max_index)
    Ld.append(max_label)
    L.pop(max_index)

    index = -1


    if DEBUG:
      print("Tamanho de B após remover M", len(B))
      print("Tamanho de S após remover M", len(S))
      print("B=", B)

    #comparasions=0
    #B_copy = B.copy()
    novo_B = []
    novo_S = []
    novo_L = []

	# Para cada bbox ainda não selecionada ... faz
	#while(len(B) != 0):
    for tmpbbox,tmps,tmpl in zip(B,S,L):

      if soft:
        S[index] -= IOU(max_bbox['box'],tmpbbox)['box']
      else:
        if IOU(max_bbox, tmpbbox) < Nt:
          novo_B.append(tmpbbox)
          novo_S.append(tmps)
          novo_L.append(tmpl)

    B = novo_B
    S = novo_S
    L = novo_L

  return Bd, Sd, Ld

def class_from_label(label):
	CLASS_FROM_LABEL = {'Alcohol bottle': 'Hygiene Product',
						'Apple': 'Organics',
						'Banana': 'Organics',
						'Beer can': 'Drinks',
						'Carrot': 'Organics',
						'Chocolate milk': 'Drinks',
						'Chocolate': 'Groceries',
						'Chocolate Powder': 'Derivative',
						'Coconut milk': 'Derivative',
						'Cracker': 'Groceries',
						'Gelatine': 'Groceries',
						'Grape juice': 'Drinks',
						'Ketchup': 'Condiments',
						'Mayonnaise': 'Condiments',
						'Napkin': 'Hygiene Product',
						'Popcorn': 'Groceries',
						'Soap': 'Hygiene Product',
						'Soda can': 'Drinks',
						'Toothpaste': 'Hygiene Product',
						'Yeast': 'Derivative'}

	klass = CLASS_FROM_LABEL[label]

	return klass


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
	orig_size_X, orig_size_Y, orig_size_colors = original_size_img.shape
	#orig_size_X = original_size_img[0]
	#orig_size_Y = original_size_img[1]

	if DEBUG:
		print("ORIGINAL_SIZE: ", original_size_img.shape)
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
		# Teste da função max_prob_label
		[tmp_prob, tmp_label] = max_prob_label(proba[i], lb.classes_)
		[tmp_prob, tmp_label]

		print("Prob a")
		print(proba)


	# In[ ]:


	scores = []
	labels = []

	#Pra cada uma das propostas calcula o rótulo mais provável
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
	detected_objects = 0

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
			detected_objects += 1


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

	rectangle_index = 0

	# loop over the bounding box indexes
	for tmp_box, tmp_score, tmp_label in zip(selected_boxes, selected_scores, selected_labels):
		#print(tmp_box, tmp_score)
		#continue

		# Se o score for muito pequeno não faz nada
		if tmp_score < THRESHOLD:
			continue

		# draw the bounding box, label, and probability on the image
		# print("Print score", tmp_score)
		(startX, startY, endX, endY) = tmp_box

		tmp_area = (endX - startX) * (endY - startY)
		#print("area ", tmp_area)

		# Se a área da bbox for muito pequena ou muito grande
		# possivelmente é um falso positivo
		if (tmp_area < 700) or (tmp_area > 5000):
			continue

		if (resultados.get(tmp_label)):
			resultados[tmp_label] = resultados[tmp_label] + 1
		else:
			resultados[tmp_label] = 1

		if (resultados[tmp_label] > 2):
			continue

		if (tmp_label == "Chocolate") or (tmp_label == "Grape juice"):
			continue

		#print("\nResultados ", resultados[tmp_label])


		#if rectangle_index > MAX_PROPOSALS_INFER:
		#	break

		bb_color = color_from_label(tmp_label)
		klass    = class_from_label(tmp_label)

		# descobre os valores para o tamanho da imagem original
		startX_orig_size = (round)((startX / 224) * orig_size_X)
		endX_orig_size = (round)((endX / 224) * orig_size_X)
		startY_orig_size = (round)((startY / 224) * orig_size_Y)
		endY_orig_size = (round)((endY / 224) * orig_size_Y)


		if DEBUG:
			#print("Mais um retângulo")
			print("-----------------")
			print("index: ", rectangle_index)
			print("Orig size x:", orig_size_X, " y:", orig_size_Y)
			print("Retângulo x: ", startX, "-", endX, "; y: ", startY, "-", endY)
			print("Novo Retângulo x: ", startX_orig_size, "-", endX_orig_size, "; y: ", startY_orig_size, "-", endY_orig_size)
			print("-----------------")

		rectangle_index = rectangle_index + 1

		cv2.rectangle(clone2, (startX, startY), (endX, endY),
			bb_color, 1)
		cv2.rectangle(original_size_img, (startX_orig_size, startY_orig_size), (endX_orig_size, endY_orig_size),
					  bb_color, 5)

		y = startY - 10 if startY - 10 > 10 else startY + 10
		new_y = startY_orig_size - 10 if startY_orig_size - 10 > 10 else startY_orig_size + 10
		text = "Obj" + str(rectangle_index) + ' ' + tmp_label + " {:.0f}%".format(tmp_score * 100)
		#text = tmp_label + " {:.0f}%".format(tmp_area)

		print('\n'+text)
		print("Class: ", klass)

		if DEBUG:
			print(text)
		cv2.putText(clone2, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.3, bb_color, 2)
		cv2.putText(original_size_img, text, (startX_orig_size, new_y),
			cv2.FONT_HERSHEY_SIMPLEX, 1, bb_color, 2)

		if DEBUG:
			#Mostra a imagem a medida que os retângulos forem sendo adicionados
			show_image(clone2)

	# show the output image *after* running NMS
	#cv2.imshow("After NMS", image)

	clone_resized = cv2.resize(clone2, (800,800))
	original_size_img_resized = cv2.resize(original_size_img, (600,600))

	original_size_rgb = cv2.cvtColor(original_size_img_resized, cv2.COLOR_BGR2RGB)

	if interactive:
		if DEBUG:
		    show_image(clone_resized)
		show_image(original_size_rgb)

#	cv2.imshow('ImageWindow', original_size_img)
#	cv2.waitKey()

	print("Salvando resultado em: ", 'output-data'+os.path.sep+filename+'.jpg')
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
THRESHOLD = 0.72

# Limite acima do qual considera que as caixas sobrepostas são o mesmo objeto
# Quanto menor esse valor menos a tolerância, logo menos sobreposições
NMS_THRESHOLD = 0.10
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

MAX_PROPOSALS_INFER = 7
# Prefixo do arquivo do modelo
PREFIX = "object_detector.h52021-09-29"
#PREFIX = PREFIX + datetime.today().strftime('%Y-%m-%d')

#True para rodar no collab
COLLAB=False

#Debug mostra ou não variáveis intermediárias
DEBUG=False

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

print("Tentando carregar o modelo ", DATA_MODEL_PATH+os.path.sep+PREFIX+"modelo")

# # Carrega o modelo
# Essa parte e a seguinte podem ser separadas em um programa a parte quando for usado em produção e/ou uma competição.
# Como o treinamento está sendo feito logo acima não existe realmente necessidade de carregar o modelo.
model = load_model(DATA_MODEL_PATH + os.path.sep + PREFIX + "modelo")
lb = pickle.loads(open(DATA_MODEL_PATH + os.path.sep + PREFIX + "encoder", "rb").read())

print("Modelo carregado.")
print("Digite uma opção")
print("1 - Executar interativamente para testes")
print("2 - Executar de maneira não interativa para testes")
print("3 - Competição")
print("4 - Sair")
opcao = input()

if DEBUG:
	print(type(lb))

if (opcao == '1'):

	if LOAD_ALL_IMAGES == False:

		IMAGE_HAVE_XML=False

		#Arquivo que será testado
		filenames = ['bonus_13','normal_6', 'normal_7', 'normal_13', 'normal_22', 'normal_26', 'normal_31']

		for test_filename in filenames:

			print("Gerando resultado para ",test_filename)

			#Modificando o nms
			#avaliacao(model, lb, test_filename, interactive=True)

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

	if (opcao == '2'):

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
			print("\n[INFO] processando imagem ", linha)
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
			image_count += 1

		print("\nnr imagens: ", image_count, " score acumulado: ", sum_score)

	else:

		if (opcao == '3'):
			DEBUG=False
			DATA_INPUT_PATH = "input-data/imagens-competicao-2021/"
			#filename = "teste"
			while (True):
				print("\nDigite a imagem na qual os objetos serão detectados. Digite 4 para sair.")
				filename = input()
				if (filename == "4"):
					break
				print("Tentando ler arquivo ", DATA_INPUT_PATH+filename+".jpg")
				avaliacao(model, lb, filename, interactive=True)

print("\nEncerrando o programa")
