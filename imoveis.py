import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  #  data web app development
import matplotlib.pyplot as plt  # for bar plot
from faker import Faker
import datetime as dt


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
st.title("21.AI Dashboard - Imóveis")
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
    st.markdown("### Imoveis a venda")
   
    # Calculating the total number of properties and the percentage of those that are sold
    total_properties = len(imoveis_para_venda)
    sold_properties = imoveis['vendido'].sum()
    percent_sold = (sold_properties / total_properties) * 100

    # Data to plot
    labels = ['Vendido', 'Disponível']
    data = [percent_sold, 100-percent_sold]
    colors = ['#283c54', '#D3D3D3']
    

    # Plotting the data
    fig, ax = plt.subplots()
    ax.set_facecolor('#d8e4e4')
    ax.pie(x=data, explode=None, labels=labels, colors=colors,
    autopct=None, shadow=True,startangle=140, )
 
    # Creating a doughnut chart by setting a circle at the center again
    circle = plt.Circle((0,0), 0.70, color='#d8e4e4')

    fig.gca().add_artist(circle)  # Adding the white circle in the middle
    fig.patch.set_facecolor('#d8e4e4')
    
    # Moving the legend to the bottom right corner
    plt.legend(labels=['Vendido: ' + str(sold_properties), 'Disponível: ' + str(total_properties - sold_properties)], 
            title='Imóveis à venda: ' + str(total_properties),
            loc='lower right')

    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
    plt.annotate('{:.2f}%'.format(percent_sold), (-0.2,0), fontsize=16, fontweight='bold')


    st.pyplot(fig)    

with fig_col2:
    st.markdown("### Imoveis Alugados")

    # Calculating the total number of properties and the percentage of those that are sold
    total_properties = len(imoveis_para_alugar)
    rent_properties = imoveis['alugado'].sum()
    percent_rent = (rent_properties / total_properties) * 100

    # Data to plot
    labels = ['Alugado', 'Vago']
    data = [percent_rent, 100-percent_rent]
    colors = ['#283c54', '#D3D3D3']   # Gold for sold, Light blue for available

    # Plotting the data
    fig, ax = plt.subplots()
    ax.pie(x=data, explode=None, labels=labels, colors=colors,
    autopct=None, shadow=True,startangle=140, )
    fig.patch.set_facecolor('#d8e4e4')
    # Creating a doughnut chart by setting a circle at the center again
    circle = plt.Circle((0,0), 0.70, color='#d8e4e4')

    fig.gca().add_artist(circle)  # Adding the white circle in the middle

    # Moving the legend to the bottom right corner
    plt.legend(labels=['Alugado: ' + str(rent_properties), 'Vago: ' + str(total_properties - rent_properties)], 
            title='Imóveis para aluguel : ' + str(total_properties),
            loc='lower right')

    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
    plt.annotate('{:.2f}%'.format(percent_rent), (-0.2,0), fontsize=16, fontweight='bold')

    st.pyplot(fig)  

with fig_col3:
    st.markdown("### Disponibilidade de Imóveis")
   
    # Then, we calculate the counts for each category.
    disponiveis_para_venda = len(imoveis_nao_vendidos_nem_alugados[imoveis_nao_vendidos_nem_alugados['tipo'] == 'Casa'])
    disponiveis_para_alugar = len(imoveis_nao_vendidos_nem_alugados[imoveis_nao_vendidos_nem_alugados['tipo'] == 'Apartamento'])

    # Now we create the pie chart data.
    labels = ['Disponíveis para Venda', 'Disponíveis para Alugar']
    sizes = [disponiveis_para_venda, disponiveis_para_alugar]
    colors = ['#283c54', '#ADD8E6']  # Gold for sale, Sky blue for rent

    # Plotting the pie chart.
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#d8e4e4')
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Moving the legend to the bottom right corner
    plt.legend(labels=['Alugado: ' + str(rent_properties), 'Disponível: ' + str(total_properties - rent_properties)], 
            title='Imóveis para Alugar: ' + str(total_properties),
            loc='lower right')

    st.pyplot(fig)


fig_col1, fig_col2, fig_col3= st.columns(3)
#=================================================================================================
#Segunda linha do dashboard

with fig_col1:
    st.markdown("### Vendas X Metas Mensais")
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
    target_values = [1000000 + 500000 * i for i in range(len(target_months))]

    # Create the target DataFrame
    target_df = pd.DataFrame({'Month': target_months, 'Target': target_values})

    # Plot the bar chart for monthly sales and line chart for monthly targets
    fig, ax =plt.subplots()
    fig.patch.set_facecolor('#d8e4e4')

    # Bar chart for monthly sales
    ax.bar(monthly_sales['sold_date'], monthly_sales['precoVenda'], width=20, label='Soma das Vendas', color='#283c54')

    # Line chart for the targets
    plt.plot(target_df['Month'], target_df['Target'], marker='o', linestyle='-', label='Metas Mensais', color='#289c84')

    # Set title and labels
    plt.title('Soma das Vendas Mensais x Metas Mensais')
    plt.xlabel('Meses')
    plt.ylabel('Soma das Vendas (em Milhões)')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Set y-axis to have integer values only
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)

    st.pyplot(fig)

with fig_col2:

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
    plt.text(0.5, 0.6, simplified_current_sales, ha='center', va='center', fontsize=18, color='blue')
    plt.text(0.5, 0.4, f'{proportional_difference_str} com Relação ao Quarter Passado', ha='center', va='center', fontsize=14, color='red')
    plt.title('Vendas do Quarter Atual')
    plt.axis('off')
    fig.patch.set_facecolor('#d8e4e4')
    fig.savefig('quarter_sales.png')  # Save the figure as an image
    st.image('quarter_sales.png')  # Display the saved image

with fig_col3:
    total_rental_income = imoveis[imoveis['alugado'] == 1]['precoAluguel'].sum()



    # Simplificando a renda total de aluguel
    simplified_rental_income = simplify_number(total_rental_income)

    # Plotando a renda total de aluguel como um único número
    fig, ax = plt.subplots()
    plt.text(0.5, 0.5, simplified_rental_income, ha='center', va='center', fontsize=20, color='blue')
    plt.title('Renda Acumulada Mensal dos Imóveis Alugados (acumulados ao longo do ano)')
    plt.axis('off')  # Desligar o eixo
    fig.patch.set_facecolor('#d8e4e4')
    fig.savefig('quarter_renting_sales.png')  # Save the figure as an image
    st.image('quarter_renting_sales.png') 