import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('winequality-red.csv') 


print(df.head())
print(df.info())

sns.countplot(x='quality', data=df, palette='viridis')
plt.title("Distribución de Calidad del Vino")
plt.xlabel("Calidad")
plt.ylabel("Cantidad")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación")
plt.show()

print("\nValores faltantes:")
print(df.isnull().sum())


X = df.drop('quality', axis=1)
y = df['quality']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de prueba:", X_test.shape)
