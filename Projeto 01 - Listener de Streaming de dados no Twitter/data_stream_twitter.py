#------------------------------------------
#--- Author....: Jaime Leite
#--- Objective.: Stream de Dados do Twitter com MongoDB, Pandas e Scikit Learn
#--- Date......: 7th June 2020
#--- Version...: 1.0
#--- Python Ver: 3.7.6
#--- Details At: 
#------------------------------------------

# Importando os módulos Tweepy, Datetime e Json
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from datetime import datetime
import json
import os
os.system('cls')  # on windows

#------------------------------------------
# Preparando a Conexão com o Twitter
#------------------------------------------
# usuário: falconylistener
# e-mail.: jaime@falcony.com.br
#------------------------------------------


# Adicione aqui sua Consumer Key
consumer_key = "VQ1lzuj9vXYzxXS7kscZIMQDX"
# Adicione aqui sua Consumer Secret 
consumer_secret = "kzD16LStIOVRZKeUGsKKsCAeFGEchkmloWHHNU9EGA0cYxwzNp"
# Adicione aqui seu Access Token
access_token = "1269796100116041730-7pVPiPzKFQAJXhDzvqgvsZMNVGIGvZ"
# Adicione aqui seu Access Token Secret
access_token_secret = "L44qHFoKmoiKbRQMlrmCHLrG6CT0KJfgIBPG9WV8X3oci"


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

nqtde = 0
ntwet = int(input("Quantos tweets você gostaria de analisar ? "))

# Criando uma classe para capturar os stream de dados do Twitter e 
# armazenar no MongoDB

class MyListener(StreamListener):
    def on_data(self, dados):
        global nqtde
        global ntwet
        tweet = json.loads(dados)
        created_at = tweet["created_at"]
        id_str = tweet["id_str"]
        text = tweet["text"]
        obj = {"created_at":created_at,"id_str":id_str,"text":text,}
        tweetind = col.insert_one(obj).inserted_id
        print (obj)
        print (nqtde)
        nqtde += 1      
        if nqtde == ntwet:
            os.system('cls')  # on windows
            return False
        else:
            return True
                

# Criando o objeto mylistener

mylistener = MyListener()

# Criando o objeto mystream
mystream = Stream(auth, listener = mylistener)

#------------------------------------------
# Preparando a Conexão com o MongoDB
#------------------------------------------

# Importando do PyMongo o módulo MongoClient
from pymongo import MongoClient   

# Criando a conexão ao MongoDB
client = MongoClient('localhost', 27017)

# Criando o banco de dados twitterdb
db = client.twitterdb

# Criando a collection "col"
col = db.tweets 

# Limpa o banco para o caso dele estar com dados
db.drop_collection(col)

# Criando uma lista de palavras chave para buscar nos Tweets
keywords = ['Will Smith']

#------------------------------------------
# Coletando os Tweets
#------------------------------------------

# Iniciando o filtro e gravando os tweets no MongoDB
mystream.filter(track=keywords)

#------------------------------------------
# Consultando os Dados no MongoDB
#------------------------------------------

mystream.disconnect()

# Verificando um documento no collection
col.find_one()

#------------------------------------------
# Análise de Dados com Pandas e Scikit-Learn
#------------------------------------------

# criando um dataset com dados retornados do MongoDB
dataset = [{"created_at": item["created_at"], "text": item["text"],} for item in col.find()]

# Importando o módulo Pandas para trabalhar com datasets em Python
import pandas as pd
#pd.__version__

# Criando um dataframe a partir do dataset 
df = pd.DataFrame(dataset)

# Imprimindo o dataframe
#print(df)

# Importando o módulo Scikit Learn
from sklearn.feature_extraction.text import CountVectorizer

import sklearn
#sklearn.__version__

# Usando o método CountVectorizer para criar uma matriz de documentos
cv = CountVectorizer()
count_matrix = cv.fit_transform(df.text)

# Contando o número de ocorrências das principais palavras em nosso dataset
word_count = pd.DataFrame(cv.get_feature_names(), columns=["word"])
word_count["count"] = count_matrix.sum(axis=0).tolist()[0]
word_count = word_count.sort_values("count", ascending=False).reset_index(drop=True)
word_count[:35]

print(word_count)

import csv

with open('result.csv', 'w', newline='') as csvfile:
    fieldnames = ['word', 'count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for n in len(word_count):
	    writer.writerow({'word': word_count["word"], 'count': word_count[count]})

# import xlsxwriter module 
#import xlsxwriter 
  
# Workbook() takes one, non-optional, argument  
# which is the filename that we want to create. 
#workbook = xlsxwriter.Workbook('result.xlsx') 
  
# The workbook object is then used to add new  
# worksheet via the add_worksheet() method. 
#worksheet = workbook.add_worksheet() 
  
# Use the worksheet object to write 
# data via the write() method. 
#worksheet.write('A1', word_count["word"] ) 
#worksheet.write('B1', word_count["count"] ) 
  
# Finally, close the Excel file 
# via the close() method. 
workbook.close() 

