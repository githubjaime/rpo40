# Módulo de formatação dos dados

# Imports
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Carrega o modelo
LinearRegressor = load('modelos/model.joblib')

# Nomes das colunas
col_names = load('modelos/col_names.joblib')

# Carrega os dados
data = pd.read_csv('static/dataset_carros.csv', low_memory=False)

# Método para obter a lista de fabricantes (nomes únicos)
def get_makes(df):
    makes = df['make'].unique().tolist()
    makes.sort()
    return makes

# Método para obter os modelos de carros (nomes únicos)
def get_cars(df):
    makes = df['make'].unique().tolist()
    makes.sort()
    cars = {}
    for make in makes:
        models = df.loc[df.make == make, 'model'].unique().tolist()
        models.sort()
        cars[make] = models
    return cars

# Função que converte o input em um dataframe
def input_process(year, odometer, make, model, transmission):
    cols = col_names
    dic = {}

    for col in cols:
        if col == 'make_' + make:
            dic[col] = [1]

        elif col == 'model_' + model:
            dic[col] = [1]

        elif col == 'transmission_' + transmission:
            dic[col] = [1]

        else:
            dic[col] = [0]

    df = pd.DataFrame(dic)
    df['year'] = year
    df['odometer'] = odometer
    return df

# Faz as previsões
def make_predict(check):
    df = check.copy()
    columns_names = ['year', 'odometer']
    features = df[columns_names]
    predictions = {}

    min_max = MinMaxScaler()
    df[columns_names] = min_max.fit_transform(features.values)
    mms = LinearRegressor.predict(df)
    predictions['Valor Previsto Aplicando Normalização'] = mms[0]

    std = StandardScaler()
    df[columns_names] = std.fit_transform(features.values)
    sts = LinearRegressor.predict(df)
    predictions['Valor Previsto Aplicando Padronização'] = sts[0]

    return predictions
