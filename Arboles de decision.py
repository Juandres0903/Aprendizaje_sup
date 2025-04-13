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

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Entrenar el árbol de decisión
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)  
tree_clf.fit(X_train, y_train)


y_pred = tree_clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f" Accuracy del modelo: {accuracy:.2f}\n")

print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt='d')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(tree_clf, feature_names=df.drop('quality', axis=1).columns,
          class_names=[str(c) for c in sorted(y.unique())],
          filled=True, rounded=True)
plt.title("Árbol de Decisión – Predicción de calidad del vino")
plt.show()
