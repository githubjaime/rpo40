#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Data Science Academy - Python Fundamentos - Capítulo 14</font>
# 
# ## Download: http://github.com/dsacademybr

# In[1]:


# Versão da Linguagem Python
from platform import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())


# ## Web Scraping

# In[2]:


# Biblioteca usada para requisitar uma página de um web site
import urllib.request


# In[3]:


# Definimos a url
# Verifique as permissões em https://www.python.org/robots.txt
with urllib.request.urlopen("https://www.python.org") as url:
    page = url.read()


# In[4]:


# Imprime o conteúdo
print(page)


# In[5]:


from bs4 import BeautifulSoup


# In[6]:


# Analise o html na variável 'page' e armazene-o no formato Beautiful Soup
soup = BeautifulSoup(page, "html.parser")


# In[7]:


soup.title


# In[8]:


soup.title.string


# In[9]:


soup.a 


# In[10]:


soup.find_all("a")


# In[11]:


tables = soup.find('table')


# In[12]:


print(tables)


# # Fim

# ### Obrigado - Data Science Academy - <a href="http://facebook.com/dsacademybr">facebook.com/dsacademybr</a>
