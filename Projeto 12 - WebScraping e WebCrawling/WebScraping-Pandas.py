#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Data Science Academy - Python Fundamentos - Capítulo 14</font>
# 
# ## Download: http://github.com/dsacademybr

# In[ ]:


# Versão da Linguagem Python
from platform import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())


# ## Web Scraping e Pandas

# In[ ]:


get_ipython().system('pip install -q tabulate')


# In[ ]:


# Imports
import pandas as pd
import requests
from bs4 import BeautifulSoup 
from tabulate import tabulate


# In[ ]:


pd.__version__


# In[ ]:


# URL
res = requests.get("http://www.nationmaster.com/country-info/stats/Media/Internet-users")


# In[ ]:


# Parser
soup = BeautifulSoup(res.content,'lxml')


# In[ ]:


# Extrai a tabela do código HTML
table = soup.find_all('table')[0] 


# In[ ]:


print(table)


# In[ ]:


# Conversão da tabela HTML em um dataframe do Pandas
df = pd.read_html(str(table))


# In[ ]:


print(df)


# In[ ]:


# Conversão do dataframe para o formato JSON
print(df[0].to_json(orient='records'))


# In[ ]:


res = requests.get("http://www.nationmaster.com/country-info/stats/Media/Internet-users")
soup = BeautifulSoup(res.content,'lxml')
table = soup.find_all('table')[0] 
df = pd.read_html(str(table))
print( tabulate(df[0], headers='keys', tablefmt='psql') )


# # Fim

# ### Obrigado - Data Science Academy - <a href="http://facebook.com/dsacademybr">facebook.com/dsacademybr</a>
