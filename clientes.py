import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import datetime as dt
import random
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
st.title(":grey[21.AI Dashboard - Clientes]")
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


fig_col1, fig_col2 = st.columns(2)
#===========================================================================
#Primeira linha do dashboard

with fig_col1:
    st.markdown('### :grey[Proporção Leads e Vendas Mensais]')
    # Supondo que 'clientes' e 'imoveis' sejam seus DataFrames

    

    # Contar leads por mês
    leads_per_month = clientes['lead_date'].groupby([clientes['lead_date'].dt.year, clientes['lead_date'].dt.month]).count()

    # Contar vendas por mês
    sales_per_month = imoveis[imoveis['vendido'] == 1]['sold_date'].groupby([imoveis['sold_date'].dt.year, imoveis['sold_date'].dt.month]).count()

    # Create a DataFrame for the graph
    leads = pd.DataFrame({
    'count': leads_per_month,
    'type': 'Leads'
    })

    sales = pd.DataFrame({
        'count': sales_per_month,
        'type': 'Vendas'
    })

    plot_data_final = pd.concat([leads, sales])

    plot_data_final['date'] = plot_data_final.index

    plot_data_final['date'] = plot_data_final['date'].apply(lambda x: f"{int(x[0])}-{int(x[1])}")

    plot_data_final['date'] = pd.to_datetime(plot_data_final['date'])

    df = pd.DataFrame({
    'Leads': leads_per_month,
    'Vendas': sales_per_month
    })

    df['proporcao'] = np.where(df['Leads'] > 0, (df['Vendas'] / df['Leads']) * 100,0)

    df['proporcao'] = df['proporcao'].round(2)

    df['date'] = df.index

    df['date'] = df['date'].apply(lambda x: f"{int(x[0])}-{int(x[1])}")

    df['date'] = pd.to_datetime(df['date'])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar( y=leads_per_month,
                x=plot_data_final['date'],
                base='relative',
                orientation='v',
                name='Leads',
                marker=dict(
                    color='#283c54',
                    )   
                ),

        secondary_y=False
        )
    
    fig.add_trace(        
        go.Bar( y=sales_per_month,
                x=plot_data_final['date'],
                base='relative',
                name='Vendas',
                orientation='v',
                marker=dict(
                    color='#289c84',
                ) 
            ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['proporcao'],
            mode='lines+markers',
            name='Razão(%)',
            marker=dict(
                color='#FF0000'
           ),
            line=dict(
                color='#FF0000'
            )
        ),
        secondary_y=True
    )

    fig.update_layout(
        
        font=dict(
            color='#283c54',
            family='Roboto'
        ),
        xaxis=dict(
        tickfont=dict(
        color='#283c54',
        size=15
        ),
        title=dict(
            text='Meses',
            font=dict(
                color='#283c54',
                size=25
                )
            )
        ),
        yaxis=dict(
            tickfont=dict(
                size=15,
                color='#283c54'
            ),
            title=dict(
                text='Quantidade de Imóveis',
                font=dict(
                    size=25,
                    color='#283c54'
                    )
            )
        ),
        yaxis2=dict(
            tickfont=dict(
                size=15,
                color='#283c54'
            ),
            title=dict(
                text='Razão(%)',
                font=dict(
                    size=20,
                    color='#283c54'
                    )
            ),
            range=[0, 100]
        ),
        legend=dict(
            font=dict(
                size=20,
                color='#283c54',
            ),
            
        ),
        bargap=0.1,
        showlegend=True,
        height=500,
        plot_bgcolor='#d8e4e4',
        paper_bgcolor='#d8e4e4',
        
        hoverlabel=dict(
            bgcolor='#283c54',
            font=dict(
                color='#ffffff'
            ),
            bordercolor='#289c84'
        )
    )

    fig.update_yaxes(showgrid=False, secondary_y=True)
    fig.update_traces(hovertemplate='%{y}<extra></extra>')

    fig.add_shape(
        # Rectangle with reference to the plot
            type="rect",
            xref="paper",
            yref="paper",
            x0=-0.145,
            y0=-0.23,
            x1=1.277,
            y1=1.28,
            line=dict(
                color="black",
                 width=1,
             )
         )


    # Exibir o gráfico
    st.plotly_chart(fig)

with fig_col2:
    st.markdown('### :grey[Distribuição dos Preços dos Imóveis Anunciados]')


    # Filtrando imóveis que estão à venda e ainda não foram vendidos
    imoveis_a_venda = imoveis[(imoveis['vendido'] == 0) & (imoveis['Finalidade'] == 'Compra')]

    fig = px.histogram(imoveis_a_venda, x='precoVenda', nbins=30, color_discrete_sequence=['#283c54'])
    fig.update_layout(
        font=dict(
            color='#283c54',
            family='Roboto'
        ),
        xaxis=dict(
        tickfont=dict(
        color='#283c54',
        size=15
        ),
        title=dict(
            text='Preço de Venda (R$)',
            font=dict(
                color='#283c54',
                size=25
                )
            )
        ),
        yaxis=dict(
        tickfont=dict(
            color='#283c54',
            size=15
        ),
        title=dict(
            text='Quantidade de Imóveis',
            font=dict(
                color='#283c54',
                size=25
                )
            )
        ),
        bargap=0.1,
        showlegend=False,
        # width=200,
        height=500,
        plot_bgcolor='#d8e4e4',
        paper_bgcolor='#d8e4e4',
        
        hoverlabel=dict(
            bgcolor='#283c54',
            font=dict(
                color='#ffffff'
            ),
            bordercolor='#289c84'
        ),
        hovermode='x'
    )

    fig.update_traces(hovertemplate='Quantidade de imóveis: %{y}<extra></extra>')

    print(max(fig.data[0]))
    
    # Adicionando a legenda com a quantidade de imóveis
    total_imoveis = len(imoveis_a_venda)
    fig.add_annotation(
        x=max(imoveis_a_venda['precoVenda'])*0.9,
        yref='paper',
        y=1.1,
        showarrow=False,
        text=f'Total de Imóveis: {total_imoveis}',
        font=dict(size=30, color='#283c54')
    )

    fig.add_shape(
        # Rectangle with reference to the plot
            type="rect",
            xref="paper",
            yref="paper",
            x0=-0.08,
            y0=-0.2,
            x1=1,
            y1=1.15,
            line=dict(
                color="black",
                 width=1,
             )
         )

    # Exibir o gráfico interativo
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

fig_col1, fig_col2, fig_col3, fig_col4 = st.columns(4)
#===========================================================================
#Segunda linha do dashboard
with fig_col1:

    st.markdown('### :grey[Leads do Quarter Atual]')

    # Obter o quarter atual
    current_quarter = pd.Timestamp.now().quarter
    current_year = pd.Timestamp.now().year

    # Obter o quarter anterior
    if current_quarter == 1:
        previous_quarter = 4
        previous_year = current_year - 1
    else:
        previous_quarter = current_quarter - 1
        previous_year = current_year

    # Filtrar leads do quarter atual
    leads_current_quarter = clientes[(clientes['lead_date'].dt.year == current_year) & (clientes['lead_date'].dt.quarter == current_quarter)]

    # Filtrar leads do quarter anterior
    leads_previous_quarter = clientes[(clientes['lead_date'].dt.year == previous_year) & (clientes['lead_date'].dt.quarter == previous_quarter)]

    # Calcular a quantidade de leads para cada quarter
    qtd_leads_current_quarter = len(leads_current_quarter)
    qtd_leads_previous_quarter = len(leads_previous_quarter)

    # Calcular a diferença entre os quarters
    diferenca = qtd_leads_current_quarter - qtd_leads_previous_quarter

    # Criando o gráfico para exibir a quantidade de lead    s do quarter atual e a diferença
    fig, ax = plt.subplots(figsize=(4,2))
    fig.patch.set_facecolor('#d8e4e4')
    fig.set_linewidth(4)
    fig.set_edgecolor('#283c54')
    plt.text(0.5, 0.6, f'{qtd_leads_current_quarter}', ha='center', va='center', fontsize=40, color='#283c54', fontweight='bold')
    if(diferenca< 0):
        plt.text(0.5, 0.2, f' ↓ {diferenca}', ha='center', va='center', fontsize=14, color='red' )
    else:
        plt.text(0.5, 0.2, f' ↑ {diferenca}', ha='center', va='center', fontsize=14, color='green')
    plt.axis('off')  # Desligar o eixo
    fig.savefig('current_quarter_leads.png')  # Save the figure as an image
    st.image('current_quarter_leads.png') 

with fig_col2:
    
    st.markdown('### :grey[CAC Atual]')
        
    # Definindo custos de marketing e vendas (exemplo)
    custo_marketing_vendas = 100000  # Exemplo: R$100.000 por mês

    # Data atual e data do mês anterior
    data_atual = pd.Timestamp.now()
    data_anterior = data_atual - pd.DateOffset(months=1)

    # Calculando o número de novos clientes no mês atual e no mês anterior
    novos_clientes_mes_atual = len(clientes[clientes['lead_date'].dt.to_period('M') == data_atual.to_period('M')])
    novos_clientes_mes_anterior = len(clientes[clientes['lead_date'].dt.to_period('M') == data_anterior.to_period('M')])

    # Calculando o CAC atual e do mês anterior
    CAC_atual = custo_marketing_vendas / novos_clientes_mes_atual if novos_clientes_mes_atual > 0 else float('inf')
    CAC_anterior = custo_marketing_vendas / novos_clientes_mes_anterior if novos_clientes_mes_anterior > 0 else float('inf')

    # Plotando o valor do CAC atual e a comparação com o mês anterior
    fig, ax = plt.subplots(figsize=(4,2))
    fig.patch.set_facecolor('#d8e4e4')
    fig.set_linewidth(4)
    fig.set_edgecolor('#283c54')
    plt.text(0.5, 0.6, f'R${CAC_atual:.2f}', ha='center', va='center', fontsize=40, color='#283c54', fontweight='bold')
    diferenca_CAC = CAC_atual - CAC_anterior
    if diferenca_CAC > 0:
        plt.text(0.5, 0.2, f' ↑ R${diferenca_CAC:.2f}', ha='center', va='center', fontsize=14, color='red')
    else:
        plt.text(0.5, 0.2, f' ↓ R${diferenca_CAC:.2f}', ha='center', va='center', fontsize=14, color='green')
    plt.axis('off')  # Desligar o eixo
    fig.savefig('last_month_CAC.png')  # Save the figure as an image
    st.image('last_month_CAC.png') 

with fig_col3:

    st.markdown('### :grey[ROAS]')

    # Suponha que 'clientes' e 'imoveis' sejam seus DataFrames
    custo_marketing_vendas = 60000000  # Exemplo: R$100.000 por mês

    # Receita dos imóveis vendidos no mês atual e anterior
    data_atual = pd.Timestamp.now()
    data_anterior = data_atual - pd.DateOffset(months=1)

    receita_mes_atual = imoveis[(imoveis['vendido'] == 1) & (imoveis['sold_date'].dt.to_period('M') == data_atual.to_period('M'))]['precoVenda'].sum()
    receita_mes_anterior = imoveis[(imoveis['vendido'] == 1) & (imoveis['sold_date'].dt.to_period('M') == data_anterior.to_period('M'))]['precoVenda'].sum()

    # Calculando o ROI atual e do mês anterior
    ROI_atual = (receita_mes_atual - custo_marketing_vendas) / custo_marketing_vendas * 100
    ROI_anterior = (receita_mes_anterior - custo_marketing_vendas) / custo_marketing_vendas * 100
    ROI_diff = ROI_atual-ROI_anterior
    # Plotando o valor do ROI atual e a comparação com o mês anterior
    fig, ax = plt.subplots(figsize=(4,2))
    fig.patch.set_facecolor('#d8e4e4')
    fig.set_linewidth(4)
    fig.set_edgecolor('#283c54')
    plt.text(0.5, 0.6, f'{ROI_atual:.2f}%', ha='center', va='center', fontsize=40, color='#283c54', fontweight='bold')
    if ROI_diff > 0:
        plt.text(0.5, 0.2, f' ↑ {abs(ROI_diff):.2f}% ', ha='center', va='center', fontsize=14, color='green')
    else:
        plt.text(0.5, 0.2, f' ↓ {abs(ROI_diff):.2f}% ', ha='center', va='center', fontsize=14, color='red')
    
    plt.axis('off')
    fig.savefig('ROAS.png')  # Save the figure as an image
    st.image('ROAS.png') 

with fig_col4:

    st.markdown('### :grey[Tempo Médio Anunciado]')

    # Revisando as datas para garantir que 'data_anuncio' é anterior a 'sold_date'
    # e calculando a diferença de dias apenas para imóveis vendidos
    imoveis['Days_On_Market'] = imoveis.apply(lambda row: (row['sold_date'] - row['data_anuncio']).days if row['vendido'] == 1 and row['data_anuncio'] <= row['sold_date'] else np.nan, axis=1)

    # Calculando a média de Days On Market para os imóveis vendidos
    average_days_on_market = imoveis['Days_On_Market'].dropna().mean()
    fake_last_average = average_days_on_market + random.randint(-12 , -8)

    # Plotting the average Days On Market
    fig, ax = plt.subplots(figsize=(4,2))
    fig.patch.set_facecolor('#d8e4e4')
    fig.set_linewidth(4)
    fig.set_edgecolor('#283c54')
    plt.text(0.5, 0.6, f'{average_days_on_market} dias', ha='center', va='center', fontsize=40, color='#283c54', fontweight='bold')
    plt.text(0.5, 0.2, f' ↓ {fake_last_average} dias',  ha='center', va='center', fontsize=14, color='green', fontweight='bold')
    plt.axis('off')
    fig.savefig('anoucement_time.png')  # Save the figure as an image
    st.image('anoucement_time.png') 





    # Definir o quarter atual


fig_col1, fig_col2, fig_col3 = st.columns(3)
#===========================================================================
#Terceira linha do dashboard

with fig_col1:

    st.markdown('### :grey[Leads por Origem - Quarter Atual]')

    quarter_atual = pd.Timestamp.now().quarter

    # Filtrar os dados para o quarter atual
    leads_quarter_atual = clientes[clientes['lead_date'].dt.quarter == quarter_atual]

    # Contar leads por origem
    leads_por_origem = leads_quarter_atual['Origem'].value_counts()

    # Definir uma lista de cores
    cores = ['#283c54', '#285460', '#286c6c', '#288478', '#289c84']

    # Criar o gráfico de barras horizontal com cores diferentes
    fig = go.Figure(data=[go.Bar(
        x=leads_por_origem.values,
        y=leads_por_origem.index,
        orientation='h',
        marker=dict(color=cores[:len(leads_por_origem)])
    )])

    
    
    fig.update_layout(
        font=dict(
            color='#283c54',
            family='Roboto'
        ),
        xaxis=dict(
            tickfont=dict(
                color='#283c54',
                size=15
            ),
            title=dict(
                text='Número de Leads',
                font=dict(
                    color='#283c54',
                    size=25
                )
            )
        ),
        yaxis=dict(
            tickfont=dict(
                color='#283c54',
                size=15
            ),
        title=dict(
            text='Origem das Leads',
            font=dict(
                size=25,
                color='#283c54',
                )
            )
        ),
        bargap=0.1,
        showlegend=False,
        plot_bgcolor='#d8e4e4',
        paper_bgcolor='#d8e4e4',
        width=800,
        height=500,
        hoverlabel=dict(
            bgcolor='#283c54',
            font=dict(
                color='#ffffff'
            ),
            bordercolor='#289c84'
        ),
    )

    fig.update_traces(hovertemplate='Quantidade de Leads: %{x}<extra></extra>')

    fig.add_shape(
    # Rectangle with reference to the plot
        type="rect",
        xref="paper",
        yref="paper",
        x0=-0.23,
        y0=-0.2,
        x1=1,
        y1=1.15,
        line=dict(
            color="black",
                width=1,
            )
        )

    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

with fig_col2:

    st.markdown('### :grey[Tipos de Leads - Quarter Atual]')

    # Filtrar dados para o quarter atual
    quarter_atual = pd.Timestamp.now().quarter
    leads_quarter_atual = clientes[clientes['lead_date'].dt.quarter == quarter_atual]

    # Contar a quantidade de leads quentes, frios e mornos
    contagem_leads = leads_quarter_atual['Classificacao_Lead'].value_counts()

    # Definir cores para cada tipo de lead
    cores = ['#289c84', '#283c54', '#286c6c']

    # Criar gráfico de doughnut com as cores específicas
    fig = go.Figure(data=[go.Pie(
        labels=contagem_leads.index,
        values=contagem_leads.values,
        hole=0.6,
        marker=dict(colors=cores),
        textinfo='percent',
        hovertemplate='Quantidade de Leads: %{value}<extra></extra>'
    )])

    fig.update_layout(
        font=dict(
            color='#283c54',
            family='Roboto'
        ),
        legend=dict(
            font=dict(
                size=20,
                color='#283c54',
            ),
        ),
        plot_bgcolor='#d8e4e4',
        paper_bgcolor='#d8e4e4',
        width=300,
        height=500,
        hoverlabel=dict(
            bgcolor='#283c54',
            font=dict(
                color='#ffffff'
            ),
            bordercolor='#289c84'
        ),
    )

    fig.update_traces(textposition='inside', textfont_size=20, hovertemplate='Quantidade de Leads: %{value}<extra></extra>')

    fig.add_shape(
        # Rectangle with reference to the plot
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=-0.2,
            x1=1.242,
            y1=1.15,
            line=dict(
                color="black",
                    width=1,
                )
            )

    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    
with fig_col3:
    st.markdown('### :grey[Funil de Marketing]')

    # Definindo o quarter atual
    current_quarter = pd.Timestamp.now().quarter
    current_year = pd.Timestamp.now().year

    # Filtrando clientes deste quarter
    clientes_quarter_atual = clientes[(clientes['lead_date'].dt.year == current_year) &
                                    (clientes['lead_date'].dt.quarter == current_quarter)]

    # Contando clientes de cada etapa do funil
    total_clientes = len(clientes_quarter_atual)
    clientes_LP = len(clientes_quarter_atual[clientes_quarter_atual['LP'] == 1])
    clientes_LP_Form = len(clientes_quarter_atual[(clientes_quarter_atual['LP'] == 1) &
                                                (clientes_quarter_atual['Formulario'] == 1)])

    funil_cores = ['#283c54', '#286c6c', '#289c84']

    # Criando o gráfico de funil com cores específicas para cada etapa
    data = dict(
        number = ['Total Clientes', 'Foram ao Site', 'Preencheram o Formulário'],
        stage = [total_clientes, clientes_LP, clientes_LP_Form])

    fig = go.Figure(go.Funnel(
    y = data['number'],
    x = data['stage'],
    textinfo = "label+percent initial",
    marker = {"color": funil_cores},
    ))

    fig.update_layout(
        font=dict(
            color='#283c54',
            family='Roboto'
        ),
        yaxis=dict(
            showticklabels=False  
        ),
        plot_bgcolor='#d8e4e4',
        paper_bgcolor='#d8e4e4',
        width=800,
        height=500,
        hoverlabel=dict(
            bgcolor='#283c54',
            font=dict(
                color='#ffffff'
            ),
            bordercolor='#289c84'
        ),
    )

    fig.update_traces(textposition='inside', textfont_size=25, hovertemplate='Quantidade: %{x}<extra></extra>')

    fig.add_shape(
            # Rectangle with reference to the plot
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.01,
                y0=-0.2,
                x1=1,
                y1=1.15,
                line=dict(
                    color="black",
                        width=1,
                    )
                )

    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    


