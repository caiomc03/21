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

# Display the first few rows of the DataFrame for verification
imoveis.head()

