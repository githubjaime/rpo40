# Módulo de relatório

# Imports
import pandas as pd
from ai.ml import data

# Dados iniciais para os gráficos
Toyota = data[data['make'] == 'Toyota']
Land_Cruiser = Toyota[Toyota['model'] == 'Land cruiser']

# Dicionário inicial de dados de veículos
info = {'year': [year for year in Land_Cruiser['year']],
        'odometer': [odometer for odometer in Land_Cruiser['odometer']],
        'price': [price for price in Land_Cruiser['price']],}
