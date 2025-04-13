import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('heart_cleveland_upload.csv')  


print(df.head())
print(df.info())
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Estadísticas
print(df.describe())


sns.countplot(x='fbs', data=df, palette='Set2')
plt.title('Distribución de enfermedad cardíaca (fbs)')
plt.xlabel('0 = No enfermedad | 1 = Enfermedad')
plt.ylabel('Cantidad')
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matriz de correlación")
plt.show()

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop('fbs', axis=1)
y = df['fbs']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División entrenamiento 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de prueba:", X_test.shape)
