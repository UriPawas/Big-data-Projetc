# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Cargar los datos
df = pd.read_excel('Big data analysis Tourism.xlsx', sheet_name='Ответы на форму (1)')

# Eliminar columnas no relevantes para el clustering
df = df.drop(['Отметка времени', 'Any suggestions for improving tourism experiences in your country or region?'], axis=1)

# Preprocesar columnas categóricas
categorical_cols = ['Gender', 'Age', 'How often do you travel in a year?', 
                   'Which of the following best describes your main purpose for travel?',
                   'When choosing a destination, which factor is most important to you?',
                   'What type of accommodation do you usually prefer when traveling?',
                   'What is your preferred mode of transportations when traveling long distances?',
                   'How do you usually plan your trips?',
                   'How likely are you to use AI tools (like ChatGPT or travel chatbots) to plan a trip?',
                   'What are your biggest challenges while planning a trip?',
                   'What would encourage you to travel more often?',
                   'How do you usually share your travel experiences?']

# Crear variables dummy para las columnas categóricas
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Estandarizar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

# Método del codo para determinar el número óptimo de clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo para Determinar K Óptimo')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.xticks(range(1, 11))
plt.grid()
plt.show()

# Aplicar K-means con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Añadir los clusters al dataframe original
df['Cluster'] = clusters

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Crear dataframe para visualización
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters

# Visualizar clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100)
plt.title('Segmentación de Viajeros usando K-means Clustering')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# Analizar características de cada cluster
cluster_analysis = df.groupby('Cluster').agg({
    'Gender': lambda x: x.mode()[0],
    'Age': lambda x: x.mode()[0],
    'How often do you travel in a year?': lambda x: x.mode()[0],
    'Which of the following best describes your main purpose for travel?': lambda x: x.mode()[0],
    'When choosing a destination, which factor is most important to you?': lambda x: x.mode()[0],
    'What type of accommodation do you usually prefer when traveling?': lambda x: x.mode()[0],
    'How likely are you to use AI tools (like ChatGPT or travel chatbots) to plan a trip?': lambda x: x.mode()[0]
})

print(cluster_analysis)