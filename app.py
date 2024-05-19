# Parte 1 - Pacotes Para Processamento de Dados
import cv2
import time
import imutils
import argparse
import pafy
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import datetime

# Parte 2 - Customização e Leitura dos Argumentos

# Cria o objeto
argumento = argparse.ArgumentParser()

# Adiciona os argumentos à lista de argumentos
argumento.add_argument("-p", "--prototxt", 
                    required = False, 
                    default = 'modelo/MobileNetSSD_deploy.prototxt.txt', 
                    help = "Caminho para o arquivo pronto com a especificação do modelo")

argumento.add_argument("-m", "--modelo",   
                    required = False, 
                    default = 'modelo/MobileNetSSD_deploy.caffemodel', 
                    help = "Caminho para o modelo caffe pré-treinado")

argumento.add_argument("-c", "--confidence", 
                    type = float,  
                    default = 0.5, 
                    help = "Probabilidade mínima para filtrar previsões fracas")

# Faz o parse dos argumentos
args = vars(argumento.parse_args())

# Parte 3 - Classes Que Serão Previstas

# Lista de 21 classes usadas no treinamento do modelo MobileNetSSD
# Cada previsão terá o nome da classe e a caixa delimitadora do objeto representado pela classe
# O modelo faz a previsão de probabilidade das 21 classes para cada objeto detectado e nós consideramos a maior probabilidade
CLASSES = ["avião", "fundo da imagem", "bicicleta", "pássaro", "barco", "garrafa", "ônibus", "carro", "gato", "cadeira", "vaca", "mesa de jantar",
           "cachorro", "cavalo", "motocicleta", "pessoa", "vaso de planta", "ovelha", "sofa", "trem", "tv/monitor"]

# Atribuímos cores randômicas para cada classe, sendo 21 cores RGB
COLORS = np.random.uniform(0, 255, size = (len(CLASSES), 3))

# Parte 4 - Carregamento do Modelo de Detecção de Objetos
# Download em: https://github.com/chuanqi305/MobileNet-SSD
# Obs: O download também pode ser feito a partir do Titan
print("\nCarregando o Modelo...")
modelo = cv2.dnn.readNetFromCaffe(args["prototxt"], args["modelo"])

# Parte 5 - Leitura do Streaming de Vídeo
# Altere o item abaixo para fazer a captura de um arquivo de vídeo
print("\nIniciando a Captura do Stream de Vídeo...")

#vs = VideoStream(src = "gado.mp4").start() # Dados de arquivo de video
vs = VideoStream(src = 0).start()         # Dados da webcam

# Vamos aguardar alguns segundos até a câmera ficar pronta
time.sleep(2.0)

# Vamos calcular um valor aproximado do FPS (Frames per second)
fps = FPS().start()

# Parte 6 - Processamento do Sinal de Vídeo

# Loop enquanto for verdadeiro (enquanto houver sinal de vídeo ou o usuário pressionar uma tecla)
ultimo_obj = ""

while True:
	
    # Fazemos a leitura de cada frame
	frame = vs.read()

    # Redimensionamos cada frame para uma largura máxima de 400 pixels
	frame = imutils.resize(frame, width = 460)

    # Imprimimos o shape
	#print('Shape do frame:', frame.shape)

    # Capturamos altura e largura do frame
	(h, w) = frame.shape[:2]

    # Convertemos o frame em imagem fazendo o resize para 300x300 pixels
	resized_image = cv2.resize(frame, (300, 300))
	
    # Criando o blob
    # A função blobFromImage():
    # blob = cv2.dnn.blobFromImage (imagem, fator de escala = 1,0, tamanho, média, swapRB = Verdadeiro)
    # imagem: a imagem de entrada que queremos pré-processar antes de passá-la por nossa rede neural profunda para classificação
    # fator de escala: Depois de realizar a subtração média, podemos opcionalmente dimensionar nossas imagens por algum fator. Padrão = 1.0
    # fator de escala deve ser 1 / sigma, pois estamos, na verdade, multiplicando os canais de entrada (após a subtração média) pelo fator de escala (aqui 1 / 127,5)
    # swapRB: OpenCV assume que as imagens estão na ordem do canal BGR; entretanto, o valor 'médio' assume que estamos usando a ordem RGB.
    # Para resolver esta discrepância, podemos trocar os canais R e B na imagem, definindo este valor para 'True'
    # Por padrão, o OpenCV realiza essa troca de canais para nós.
	blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB = True)

    # Usamos o blob para alimentar o modelo de Deep Learning
	modelo.setInput(blob) 

    # Fazemos previsões com o modelo
	previsoes = modelo.forward()

	# Loop pelas previsões
	for i in np.arange(0, previsoes.shape[2]):

        # Extraímos o nível de confiança (a probabilidade) das previsões
		confidence = previsoes[0, 0, i, 2]

        # Condicional a fim de verificar se a confiança é maior que o argumento do usuário
        # Se for menor, não consideramos
		if confidence > args["confidence"]:

            # Índices das previsões (índice das classes)
			idx = int(previsoes[0, 0, i, 1])

            # Previsões das caixas delimitadoras
			box = previsoes[0, 0, i, 3:7] * np.array([w, h, w, h])

            # Coordenadas das previsões (apenas convertemos as coordenadas anteriores para o tipo inteiro)
			(startX, startY, endX, endY) = box.astype("int")

            # Preparamos os labels com os respectivos níveis de confiança
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

			if ultimo_obj != label:
				print("Data/Hora:" ,datetime.datetime.now(),"- Objeto Detectado:", label)
				ultimo_obj = label

            # Desenhamos as caixas delimitadoras a partir das coordenadas
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

            # Incluímos texto com a classe prevista
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# Visualizamos o frame
	cv2.imshow("Frame", frame)

    # A captura não para até o sinal de vídeo ser encerrado ou pressionarmos uma tecla
	key = cv2.waitKey(1) & 0xFF

	# Pressionamos 'q' para encerrar a detecção
	if key == ord("q"):
		break

	# Atuualiza o contador de FPS
	fps.update()

# Interrompe o timer
fps.stop()

# Parte 7 - Mostrar em Tempo Real o Resultado da Detecção

# Mostra os detalhes para o usuário
print("\nTempo Total de Captura e Detecção: {:.2f}".format(fps.elapsed()))
print("FPS Aproximado: {:.2f}".format(fps.fps()))

# Encerra a janela
cv2.destroyAllWindows()

# Encerra a captura de vídeo
vs.stop()