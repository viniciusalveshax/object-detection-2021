# object-detection-2021
Object Detection Code for 2021

Detecção de objetos:

Uma vez treinado o modelo coloque o arquivo do modelo na pasta model-data/
e as imagens de entrada dentro da pasta input-data

Ajuste os parâmetros das pastas acima no arquivo keras_model_loading.py

Rodando a detecção:

No Linux:
$ python keras_model_loading 2> /dev/null
O envio da saída de erro para /dev/null é para fazer com que algumas mensagens de aviso
não sejam mostradas
