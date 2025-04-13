import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


#Cargar el dataset


df = pd.read_csv("car data.csv")  


print("Primeras filas del dataset:")
print(df.head())


#Información general

print("\nInformación del dataset:")
print(df.info())


#Estadísticas básicas

print("\nEstadísticas descriptivas:")
print(df.describe(include='all'))  


#  Verificar valores nulos

print("\nValores nulos por columna:")
print(df.isnull().sum())


#Visualizar distribución de precios

plt.figure()
sns.histplot(df['Selling_Price'], kde=True, bins=30, color='skyblue')
plt.title('Distribución del precio de venta')
plt.xlabel('Precio de venta')
plt.ylabel('Frecuencia')
plt.show()


#Boxplots para detectar outliers en variables numéricas importantes

numerical_cols = ['Selling_Price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']


for col in ['mileage', 'engine', 'max_power']:
    if col in df.columns:
        df[col] = df[col].str.extract('(\d+\.?\d*)') 
        df[col] = pd.to_numeric(df[col], errors='coerce') 

# Crear boxplots
for col in numerical_cols:
    if col in df.columns:
        plt.figure()
        sns.boxplot(data=df, x=col, color='lightcoral')
        plt.title(f'Boxplot de {col}')
        plt.show()


#Correlación entre variables numéricas


correlation_matrix = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación entre variables numéricas")
plt.show()


#Conteo de valores categóricos

categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']

for col in categorical_cols:
    if col in df.columns:
        plt.figure()
        sns.countplot(data=df, x=col, palette='pastel')
        plt.title(f'Distribución de {col}')
        plt.xticks(rotation=45)
        plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df_clean = df.drop(columns=['name'], errors='ignore')

for col in df_clean.columns:
    if df_clean[col].dtype in ['float64', 'int64']:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

df_encoded = pd.get_dummies(df_clean, drop_first=True)

X = df_encoded.drop(columns=['Selling_Price'])
y = df_encoded['Selling_Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

print("Dimensiones de entrenamiento:", X_train.shape)
print("Dimensiones de prueba:", X_test.shape)
