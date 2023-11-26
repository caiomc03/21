import time  # to simulate a real time data, time loop
import matplotlib.pyplot as plt  # for bar plot
import matplotlib.dates as mdates
import datetime as dt
from random import randint  # to generate random numbers

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  #  data web app development
from faker import Faker
import plotly.graph_objects as go



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
st.title(":grey[21.AI Dashboard - Imóveis]")
placeholder = st.empty()

#===============================================================================================
#Criacao do Fake Dataframe

# Creating a fake dataset
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
        # Random choice between 'Compra' and 'Aluguel' for properties not sold or rented
        return np.random.choice(['Compra', 'Aluguel'])

# Function to generate a random date between the start date and the end date
def random_date():
    start= dt.datetime.now() - pd.DateOffset(years=1)
    end = dt.datetime.now()

    return start + pd.Timedelta(seconds=np.random.randint(0, int((end - start).total_seconds())))

# Applying the function to the DataFrame
imoveis['Finalidade'] = imoveis.apply(determinar_finalidade, axis=1)

# Add 'sold_date' column with random dates for when the property was sold
# For properties not sold, we leave the date as NaT (Not a Time)
imoveis['sold_date'] = np.nan
imoveis.loc[imoveis['vendido'] == 1, 'sold_date'] = imoveis[imoveis['vendido'] == 1].apply(lambda row: random_date(), axis=1)
imoveis['sold_date'] = pd.to_datetime(imoveis['sold_date'])

# To remove the time from the 'sold_date' and keep only the date, we can normalize the datetime.
imoveis['sold_date'] = imoveis['sold_date'].dt.normalize()

# Adjusting the 'precoAluguel' field to be NaN for properties that are not for rent
imoveis.loc[imoveis['alugado'] == 0, 'precoAluguel'] = np.nan

#Fim da criação do Fake Dataframe
#===============================================================================================


#===============================================================================================
#Determinacao dos novos dataframes para o dashboard (camada prata)

imoveis_nao_vendidos_nem_alugados = imoveis[(imoveis['vendido'] == 0) & (imoveis['alugado'] == 0)]
imoveis_para_venda = imoveis[(imoveis['Finalidade'] == "Compra")]
imoveis_vendidos = imoveis_para_venda[(imoveis_para_venda['vendido'] == 1)].shape[0]
imoveis_parados = imoveis_para_venda[(imoveis_para_venda['vendido'] == 0)].shape[0]
imoveis_para_alugar = imoveis[(imoveis['Finalidade'] == "Aluguel")]
imoveis_alugados = imoveis_para_alugar[(imoveis_para_alugar['alugado'] == 1)]
imoveis_para_alugar_parados = imoveis_para_alugar[(imoveis_para_alugar['alugado'] == 0)]

# Função para simplificar o número
def simplify_number(num):
    if num >= 1e9:
        return f'R$ {num / 1e9:.2f}B'
    elif num >= 1e6:
        return f'R$ {num / 1e6:.2f}M'
    elif num >= 1e3:
        return f'R$ {num / 1e3:.2f}K'
    else:
        return f'R$ {num:.2f}'

# # read csv from a URL
# @st.cache_data
# def get_imoveis_nao_vendidos_nem_alugados() -> pd.DataFrame:
#     return imoveis_nao_vendidos_nem_alugados

# df = get_data()



fig_col1, fig_col2, fig_col3= st.columns(3)
#=================================================================================================
#Primeira linha do dashboard

with fig_col1:
    st.markdown('### :grey[Imóveis à Venda]')
   
    # Calculating the total number of properties and the percentage of those that are sold
    total_properties = len(imoveis_para_venda)
    sold_properties = imoveis['vendido'].sum()
    percent_sold = (sold_properties / total_properties) * 100

   # Data for the plot
    labels = ['Vendido', 'Disponível']
    values = [percent_sold, 100-percent_sold]
    colors = ['#283c54', '#9ba8a8']

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.7, marker_colors=colors)])

    # Update the layout
    fig.update_layout(
        # title_text='Total de Imóveis: ' + str(total_properties),
        annotations=[dict(text=f'{percent_sold:.2f}%', x=0.5, y=0.5, font_size=16, showarrow=False)],
        legend=dict(title='Legenda', itemsizing='constant'),
        showlegend=True,
        paper_bgcolor='#d8e4e4',
        plot_bgcolor='#d8e4e4',
        hoverlabel=dict(
            bgcolor='#283c54',
            font=dict(
                color='#ffffff'
            ),
            bordercolor='#289c84'
        ),

    )
    #### Alterar Legenda
    fig.update_traces(hovertemplate='%{value}')

    st.plotly_chart(fig, theme= 'streamlit', use_container_width= True)


with fig_col2:
    st.markdown("### :grey[Imoveis Alugados]")

    # Calculating the total number of properties and the percentage of those that are sold
    total_properties = len(imoveis_para_alugar)
    rent_properties = imoveis['alugado'].sum()
    percent_rent = (rent_properties / total_properties) * 100

    # Data to plot
    labels = ['Alugado', 'Vago']
    data = [percent_rent, 100-percent_rent]
    colors = ['#289c84', '#9ba8a8']   # Gold for sold, Light blue for available

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.7, marker_colors=colors)])

    # Update the layout
    fig.update_layout(
        # title_text='Total de Imóveis: ' + str(total_properties),
        annotations=[dict(text=f'{percent_sold:.2f}%', x=0.5, y=0.5, font_size=16, showarrow=False)],
        legend=dict(title='Legenda', itemsizing='constant'),
        showlegend=True,
        paper_bgcolor='#d8e4e4',
        plot_bgcolor='#d8e4e4',
        hoverlabel=dict(
            bgcolor='#283c54',
            font=dict(
                color='#ffffff'
            ),
            bordercolor='#289c84'
        ),

    )
    #### Alterar Legenda
    fig.update_traces(hovertemplate='%{x}')

    st.plotly_chart(fig, theme= 'streamlit', use_container_width= True) 

with fig_col3:
   
    # Then, we calculate the counts for each category.
    disponiveis_para_venda = len(imoveis_nao_vendidos_nem_alugados[imoveis_nao_vendidos_nem_alugados['tipo'] == 'Casa'])
    disponiveis_para_alugar = len(imoveis_nao_vendidos_nem_alugados[imoveis_nao_vendidos_nem_alugados['tipo'] == 'Apartamento'])
    disponiveis_total = disponiveis_para_venda + disponiveis_para_alugar
    # Dados para o gráfico
    labels = ['Disponíveis para Venda', 'Disponíveis para Alugar']
    values = [disponiveis_para_venda, disponiveis_para_alugar]
    colors = ['#283c54', '#289c84']  # Azul escuro para venda, Verde para alugar

    # Criando o gráfico de pizza
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors)])

    # Atualizando o layout
    fig.update_layout(
        annotations=[
            dict(text=f'{disponiveis_para_venda/disponiveis_total*100:.2f}%', x=0.85, y=0.8, font_size=16, showarrow=False),
            dict(text=f'{disponiveis_para_alugar/disponiveis_total*100:.2f}%', x=0.15, y=0.2, font_size=16, showarrow=False)
        ],
        legend=dict(title='Legenda', itemsizing='constant'),
        showlegend=True,
        paper_bgcolor='#d8e4e4',
        plot_bgcolor='#d8e4e4'
    )

    # Exibindo o gráfico no Streamlit
    with st.container():
        st.markdown("### :grey[Imóveis Disponíveis em Anúncio]")
        st.plotly_chart(fig, use_container_width=True)

#=================================================================================================
#Segunda linha do dashboard

st.markdown("### :grey[Vendas ao longo do ano]")
monthly_sales = imoveis[imoveis['vendido'] == 1].groupby(imoveis['sold_date'].dt.to_period('M'))['precoVenda'].sum().reset_index()
monthly_sales['sold_date'] = monthly_sales['sold_date'].dt.to_timestamp()

# Ajustar o DataFrame de metas para começar a partir do primeiro mês das vendas
first_month_of_sales = monthly_sales['sold_date'].min()
target_months = pd.date_range(start=first_month_of_sales, periods=len(monthly_sales), freq='MS')
target_values = [1000000 + 500000 * i for i in range(len(target_months))]
target_df = pd.DataFrame({'Month': target_months, 'Target': target_values})

    # Adjusting the target data to start from the first month of the sales data
# and setting the y-axis to show integer values

# Get the first month of sales from the sales data
first_month_of_sales = monthly_sales['sold_date'].min()

# Generate the target months starting from the first month of sales
target_months = pd.date_range(start=first_month_of_sales, periods=len(target_months), freq='MS')

# Generate target values starting from 1 million
target_values = [10000000 + 10000000 * randint(0,10) for i in range(len(target_months))]

# Create the target DataFrame
target_df = pd.DataFrame({'Month': target_months, 'Target': target_values})

# Plot the bar chart for monthly sales and line chart for monthly targets
fig, ax =plt.subplots()
fig.patch.set_facecolor('#d8e4e4')
fig.set_linewidth(4)
fig.set_edgecolor('#283c54')
fig.set_figwidth(15)

# Bar chart for monthly sales
ax.bar(monthly_sales['sold_date'], monthly_sales['precoVenda'], width=20, label='Soma das Vendas', color='#283c54')

# Line chart for the targets
plt.plot(target_df['Month'], target_df['Target'], marker='o', linestyle='-', label='Metas Mensais', color='#289c84')

# Set title and labels
# plt.title('Soma das Vendas Mensais x Metas Mensais')
plt.ylabel('Soma das Vendas (em Milhoes de R$)')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Set y-axis to have integer values only
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x/1000000))))

# Show legend
plt.legend(labels=[f'Metas Mensais',f'Soma das Vendas'], 
        loc='lower right')

# Show grid on y axes only
plt.grid(axis='y', alpha=0.5, linestyle='--')

st.pyplot(fig)

#=================================================================================================
#Terceira linha do dashboard
fig_col1, fig_col2= st.columns(2)
with fig_col1:
    st.markdown("### :grey[Vendas do Quarter Atual]")

    # Calcular as vendas totais por quarter
    vendas_por_quarter = imoveis[imoveis['vendido'] == 1].groupby(imoveis['sold_date'].dt.to_period('Q'))['precoVenda'].sum()

    # Obter as vendas totais do quarter atual e do quarter anterior
    current_quarter = pd.Timestamp.now().to_period('Q')
    previous_quarter = (current_quarter - 1).asfreq('Q', 'end')

    total_current_quarter_sales = vendas_por_quarter.get(current_quarter, 0)
    total_previous_quarter_sales = vendas_por_quarter.get(previous_quarter, 0)

    # Simplificar o valor das vendas atuais para exibição
    simplified_current_sales = f'R$ {total_current_quarter_sales / 1000000:.1f} Milhões'

    # Calcular a diferença proporcional entre o quarter atual e o anterior
    if total_previous_quarter_sales > 0:
        proportional_difference = ((total_current_quarter_sales - total_previous_quarter_sales) / total_previous_quarter_sales) * 100
    else:
        proportional_difference = np.inf  # Aumento infinito se não houve vendas no quarter anterior

    # Formatar a diferença proporcional
    if proportional_difference == np.inf:
        proportional_difference_str = "Infinity"  # Sem vendas no quarter anterior
    else:
        proportional_difference_str = f'{proportional_difference:.1f}%'
        if proportional_difference > 0:
            proportional_difference_str = '+' + proportional_difference_str  # Adicionando sinal de '+' para diferenças positivas

    # Plotando o gráfico
    fig, ax = plt.subplots()
    plt.text(0.5, 0.6, simplified_current_sales, ha='center', va='center', fontsize=18, color='#283c54', fontweight='bold')
    plt.text(0.5, 0.4, f'{proportional_difference_str} com Relação ao Quarter Passado', ha='center', va='center', fontsize=14, color='red')
    plt.axis('off')
    fig.patch.set_facecolor('#d8e4e4')
    fig.set_linewidth(4)
    fig.set_edgecolor('#283c54')
    fig.set_figheight(3)

    fig.savefig('quarter_sales.png')  # Save the figure as an image
    st.image('quarter_sales.png')  # Display the saved image

with fig_col2:

    st.markdown("### :grey[Renda Acumulada dos Imóveis Alugados]")
    total_rental_income = imoveis[imoveis['alugado'] == 1]['precoAluguel'].sum()



    # Simplificando a renda total de aluguel
    simplified_rental_income = simplify_number(total_rental_income)

    # Plotando a renda total de aluguel como um único número
    fig, ax = plt.subplots()
    plt.text(0.5, 0.5, simplified_rental_income, ha='center', va='center', fontsize=20, color='#283c54', fontweight='bold')
    plt.axis('off')  # Desligar o eixo
    fig.patch.set_facecolor('#d8e4e4')
    fig.set_linewidth(4)
    fig.set_edgecolor('#283c54')
    fig.set_figheight(3)
    fig.savefig('quarter_renting_sales.png')  # Save the figure as an image
    st.image('quarter_renting_sales.png') 