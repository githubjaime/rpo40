#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Data Science Academy - Python Fundamentos - Capítulo 12</font>
# 
# ## Download: http://github.com/dsacademybr

# In[ ]:


# Versão da Linguagem Python
from platform import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())


# ## Detecção de Emoções em Imagens com Inteligência Artificial

# ## Teste

# In[ ]:


#get_ipython().system('pip install -q tensorflow==1.15.2')


# In[ ]:


from scipy import misc
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
import os, sys, inspect
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib as mat
import matplotlib.image as mpimg
from modulos import utils
from modulos.utils import testResult
from tensorflow.python.framework import ops
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import sklearn as sk
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


np.__version__


# In[ ]:


tf.__version__


# In[ ]:


mat.__version__


# In[ ]:


sk.__version__


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()


# In[ ]:


emotion = {0:'anger', 
           1:'disgust',
           2:'fear',
           3:'happy',
           4:'sad',
           5:'surprise',
           6:'neutral'}


# In[ ]:


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# In[ ]:


sess = tf.InteractiveSession()


# In[ ]:


new_saver = tf.train.import_meta_graph('modelo/model.ckpt-900.meta')
new_saver.restore(sess, 'modelo/model.ckpt-900')
tf.get_default_graph().as_graph_def()


# In[ ]:


x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")


# In[ ]:


img = mpimg.imread('images_teste/image05.jpg')     
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()


# In[ ]:


image_0 = np.resize(gray,(1,48,48,1))
tResult = testResult()
num_evaluations = 50


# In[ ]:


for i in range(0, num_evaluations):
    result = sess.run(y_conv, feed_dict={x:image_0})
    label = sess.run(tf.argmax(result, 1))
    label = label[0]
    label = int(label)
    tResult.evaluate(label)
tResult.display_result(num_evaluations)


# Para adquirir conhecimento técnico sólido e especializado em Deep Learning, Visão Computacional, Processamento de Linguagem Natural e outros temas relacionados à Inteligência Artificial, confira nosso programa completo: <a href="https://www.datascienceacademy.com.br/pages/formacao-inteligencia-artificial">Formação Inteligência Artificial</a>.

# # Fim

# ### Obrigado - Data Science Academy - <a href="http://facebook.com/dsacademybr">facebook.com/dsacademybr</a>
