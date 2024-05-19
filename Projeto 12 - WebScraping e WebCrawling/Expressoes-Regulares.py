#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Data Science Academy - Python Fundamentos - Capítulo 14</font>
# 
# ## Download: http://github.com/dsacademybr

# In[ ]:


# Versão da Linguagem Python
from platform import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())


# ## Expressões Regulares
# 
# Uma expressão regular é um método formal de se especificar um padrão de texto.
# 
# Mais detalhadamente, é uma composição de símbolos, caracteres com funções especiais, que, agrupados entre si e com caracteres literais, formam uma sequência, uma expressão. Essa expressão é interpretada como uma regra, que indicará sucesso se uma entrada de dados qualquer casar com essa regra, ou seja, obedecer exatamente a todas as suas condições.

# In[ ]:


# Importando o módulo re (regular expression)
# Esse módulo fornece operações com expressões regulares (ER)
import re


# In[ ]:


# Lista de termos para busca
lista_pesquisa = ['informações', 'Negócios']


# In[ ]:


# Texto para o parse
texto = 'Existem muitos desafios para o Big Data. O primeiro deles é a coleta dos dados, pois fala-se aqui de ''enormes quantidades sendo geradas em uma taxa maior do que um servidor comum seria capaz de processar e armazenar. ''O segundo desafio é justamente o de processar essas informações. Com elas então distribuídas, a aplicação deve ser ''capaz de consumir partes das informações e gerar pequenas quantidades de dados processados, que serão calculados em ''conjunto depois para criar o resultado final. Outro desafio é a exibição dos resultados, de forma que as informações ''estejam disponíveis de forma clara para os tomadores de decisão.'


# In[ ]:


# Exemplo básico de Data Mining
for item in lista_pesquisa:
    print ('Buscando por "%s" em: \n\n"%s"' % (item, texto))
    
    # Verificando se o termo de pesquisa existe no texto
    if re.search(item,  texto):
        print ('\n')
        print ('Palavra encontrada. \n')
        print ('\n')
    else:
        print ('\n')
        print ('Palavra não encontrada.')
        print ('\n')


# In[ ]:


# Termo usado para dividir uma string
split_term = '@'


# In[ ]:


frase = 'Qual o domínio de alguém com o e-mail: aluno@gmail.com'


# In[ ]:


# Dividindo a frase
re.split(split_term, frase)


# In[ ]:


def encontra_padrao(lista, frase):
    
    for item in lista:
        print ('Pesquisando na frase: %r' %item)
        print (re.findall(item, frase))
        print ('\n')


# In[ ]:


frase_padrao = 'zLzL..zzzLLL...zLLLzLLL...LzLz...dzzzzz...zLLLL'

lista_padroes = [ 'zL*',       # z seguido de zero ou mais L
                  'zL+',       # z seguido por um ou mais L
                  'zL?',       # z seguido por zero ou um L
                  'zL{3}',     # z seguido por três L
                  'zL{2,3}',   # z seguido por dois a três L
                ]


# In[ ]:


encontra_padrao(lista_padroes, frase_padrao)


# In[ ]:


frase = 'Esta é uma string com pontuação. Isso pode ser um problema quando fazemos mineração de dados em busca '        'de padrões! Não seria melhor retirar os sinais ao fim de cada frase?'


# In[ ]:


# A expressão [^!.? ] verifica por valores que não sejam pontuação 
# (!, ., ?) e o sinal de adição (+) verifica se o item aparece pelo menos 
# uma vez. Traduzindo: esta expressão diz: traga apenas as palavras na 
# frase.
re.findall('[^!.? ]+', frase)


# In[ ]:


frase = 'Esta é uma frase de exemplo. Vamos verificar quais padrões serão encontrados.'

lista_padroes = [ '[a-z]+',      # sequência de letras minúsculas
                  '[A-Z]+',      # sequência de letras maiúsculas
                  '[a-zA-Z]+',   # sequência de letras minúsculas e maiúsculas
                  '[A-Z][a-z]+'] # uma letra maiúscula, seguida de uma letra minúscula


# In[ ]:


encontra_padrao(lista_padroes, frase)


# ### Escape Codes

# É possível usar códigos específicos para enocntrar padrões nos dados, tais como dígitos, não dígitos, espaços, etc..
# 
# <table border="1" class="docutils">
# <colgroup>
# <col width="14%" />
# <col width="86%" />
# </colgroup>
# <thead valign="bottom">
# <tr class="row-odd"><th class="head">Código</th>
# <th class="head">Significado</th>
# </tr>
# </thead>
# <tbody valign="top">
# <tr class="row-even"><td><tt class="docutils literal"><span class="pre">\d</span></tt></td>
# <td>um dígito</td>
# </tr>
# <tr class="row-odd"><td><tt class="docutils literal"><span class="pre">\D</span></tt></td>
# <td>um não-dígito</td>
# </tr>
# <tr class="row-even"><td><tt class="docutils literal"><span class="pre">\s</span></tt></td>
# <td>espaço (tab, espaço, nova linha, etc.)</td>
# </tr>
# <tr class="row-odd"><td><tt class="docutils literal"><span class="pre">\S</span></tt></td>
# <td>não-espaço</td>
# </tr>
# <tr class="row-even"><td><tt class="docutils literal"><span class="pre">\w</span></tt></td>
# <td>alfanumérico</td>
# </tr>
# <tr class="row-odd"><td><tt class="docutils literal"><span class="pre">\W</span></tt></td>
# <td>não-alfanumérico</td>
# </tr>
# </tbody>
# </table>
# 

# In[ ]:


# O prefixo r antes da expressão regular evita o pré-processamento da ER 
# pela linguagem. Colocamos o modificador r (do inglês "raw", crú) 
# imediatamente antes das aspas
r'\b'


# In[ ]:


'\b'


# In[ ]:


frase = 'Esta é uma string com alguns números, como 1287 e um símbolo #hashtag'

lista_padroes = [ r'\d+', # sequência de dígitos
                  r'\D+', # sequência de não-dígitos
                  r'\s+', # sequência de espaços
                  r'\S+', # sequência de não-espaços
                  r'\w+', # caracteres alfanuméricos
                  r'\W+', # não-alfanumérico
                ]


# In[ ]:


encontra_padrao(lista_padroes, frase)


# # FIM

# ### Obrigado - Data Science Academy - <a href="http://facebook.com/dsacademybr">facebook.com/dsacademybr</a>
