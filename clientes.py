import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import datetime as dt
import random
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim

st.set_page_config(
    page_title="Dashboard",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #d8e4e4; /* Cor verde desejada */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .title-text {
        color: #283c54; /* Substitua pela cor desejada */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("21.AI Dashboard - Clientes")
placeholder = st.empty()

#===========================================================================
#Criacao do dataframe de clientes fake

# Initialize Faker to a constant seed for reproducibility
fake = Faker()
Faker.seed(0)
np.random.seed(0)

num_linhas = 5000

# Creating the imoveis DataFrame
imoveis = pd.DataFrame({
    'imovel_id': range(1, num_linhas + 1),
    'localizacao': [fake.address() for _ in range(num_linhas)],
    'tipo': np.random.choice(['Casa', 'Apartamento'], num_linhas),
    'tamanho': np.random.normal(120, 40, num_linhas).clip(20),  # Minimum size of 20m²
    'quartos': np.random.randint(1, 5, num_linhas),  # Between 1 and 4 bedrooms
    'suites': np.random.randint(0, 5, num_linhas),  # Between 0 and 4 suites
    'vendido': np.random.choice([0, 1], num_linhas),  # 0 for not sold, 1 for sold
    'alugado': np.random.choice([0, 1], num_linhas),  # 0 for not rented, 1 for rented
    'vagaGaragem': np.where(np.random.choice(['Casa', 'Apartamento'], num_linhas) == 'Apartamento', np.random.choice([0, 1], num_linhas), 0),
    'precoVenda': np.random.normal(500000, 150000, num_linhas).clip(100000),  # Minimum price of R$100.000
    'precoAluguel': np.random.normal(2000, 500, num_linhas).clip(500)  # Minimum rent of R$500
})

# Adjusting suites to not exceed the number of bedrooms
imoveis['suites'] = imoveis.apply(lambda row: min(row['suites'], row['quartos']), axis=1)

# Adjusting the logic for sold/rented
for index, row in imoveis.iterrows():
    if row['vendido'] == 1:
        imoveis.at[index, 'alugado'] = 0
    elif row['alugado'] == 1:
        imoveis.at[index, 'vendido'] = 0

# Function to determine the purpose of the property
def determinar_finalidade(row):
    if row['vendido'] == 1:
        return 'Compra'
    elif row['alugado'] == 1:
        return 'Aluguel'
    else:
        return np.random.choice(['Compra', 'Aluguel'])

# Applying the function to the DataFrame
imoveis['Finalidade'] = imoveis.apply(determinar_finalidade, axis=1)

# Define the start and end date for lead generation as one year before the current date
start_date = dt.datetime.now() - pd.DateOffset(years=1)
end_date = dt.datetime.now()

# Function to generate a random date between the start date and the end date
def random_date(start, end):
    return start + pd.Timedelta(seconds=np.random.randint(0, int((end - start).total_seconds())))

# Apply the function to generate 'sold_date'
imoveis['sold_date'] = np.where(imoveis['vendido'] == 1, 
                                imoveis.apply(lambda row: random_date(start_date, end_date), axis=1),
                                pd.NaT)
imoveis['sold_date'] = pd.to_datetime(imoveis['sold_date']).dt.normalize()

# Setting a fixed number of days (e.g., 90 days) before the sold_date or the current date for the announcement date
fixed_days_before = 90

# Function to set the 'data_anuncio' as a fixed period before 'sold_date' or the current date
def set_fixed_data_anuncio(row):
    if row['vendido'] == 1 and pd.notna(row['sold_date']):
        return row['sold_date'] - pd.DateOffset(days=fixed_days_before)
    else:
        return end_date - pd.DateOffset(days=fixed_days_before)

# Atualizando 'data_anuncio' para ser sempre 17 dias antes de 'sold_date'
imoveis['data_anuncio'] = imoveis['sold_date'] - pd.DateOffset(days=17)

# Adjusting the 'precoAluguel' field to be NaN for properties that are not for rent
imoveis.loc[imoveis['alugado'] == 0, 'precoAluguel'] = np.nan

# Ajuste o número de linhas para 5000
num_linhas = 5000

# Exemplo para gerar dados de clientes
clientes = pd.DataFrame({
    'cliente_id': range(1, num_linhas + 1),
    'faixaEtaria': np.random.choice(['20-30', '30-40', '40-50', '50+'], num_linhas),
    'Sexo': np.random.choice(['Masculino', 'Feminino'], num_linhas),
    # Parâmetros de Funil de Marketing
    'Origem': np.random.choice(['Instagram', 'Facebook', 'Email', 'Site', 'Outros'], num_linhas),
    'LP': np.random.choice([0, 1], num_linhas),  # 0 para não, 1 para sim
    'Formulario': np.random.choice([0, 1], num_linhas)  # 0 para não, 1 para sim
})

# Se LP = 0 & Formulario = 1, então LP = 1
for index, row in clientes.iterrows():
    if row['Formulario'] == 1 and row['LP'] == 0:
        clientes.at[index, 'LP'] = 1

# Define the start date for lead generation as one year before the current date
lead_start_date = dt.datetime.now() - pd.DateOffset(years=1)
lead_end_date = dt.datetime.now()

# Function to generate a random date between the start date and the end date
def random_lead_date(start, end):
    return start + pd.Timedelta(seconds=np.random.randint(0, int((end - start).total_seconds())))

# Apply the function to generate lead dates
clientes['lead_date'] = clientes.apply(lambda row: random_lead_date(lead_start_date, lead_end_date), axis=1)
clientes['lead_date'] = clientes['lead_date'].dt.normalize()

# Gerando coordenadas aleatórias dentro do intervalo geográfico de São Paulo
latitudes = np.random.uniform(-23.7, -23.4, num_linhas)
longitudes = np.random.uniform(-46.9, -46.3, num_linhas)

# Adicionando as coordenadas ao DataFrame
clientes['latitude'] = latitudes
clientes['longitude'] = longitudes

# Função para classificar os leads
def classificar_lead(row):
    if row['LP'] == 0 and row['Formulario'] == 0:
        return 'Frio'
    elif row['LP'] == 1 and row['Formulario'] == 0:
        return 'Morno'
    elif row['LP'] == 1 and row['Formulario'] == 1:
        return 'Quente'
    else:
        return 'Indefinido'  # Para qualquer outra combinação

# Aplicar a função para criar a coluna condicional
clientes['Classificacao_Lead'] = clientes.apply(classificar_lead, axis=1)

#Fim da criação do dataframe de clientes fake
#===========================================================================
