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


#regrecion logistica
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print(" Evaluación del modelo de Regresión Logística:")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}\n")


print("Reporte de clasificación:\n")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='d')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.xticks([0.5, 1.5], ['No Enfermo', 'Enfermo'])
plt.yticks([0.5, 1.5], ['No Enfermo', 'Enfermo'], rotation=0)
plt.show()
