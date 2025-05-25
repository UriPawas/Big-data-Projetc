import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder

# Cargar datos
file_path = "Big data analysis Tourism.xlsx"
df = pd.read_excel(file_path, sheet_name='Ответы на форму (1)')

# Mapear variable dependiente (uso de IA)
ai_map = {"Very unlikely": 0, "Unlikely": 1, "Neutral": 2, "Likely": 3, "Very likely": 4}
df['AI_Tools_Use'] = df['How likely are you to use AI tools (like ChatGPT or travel chatbots) to plan a trip?'].map(ai_map)

# Eliminar filas con AI_Tools_Use vacías
df = df.dropna(subset=['AI_Tools_Use'])

# Variables seleccionadas (puedes ajustar)
variables = ['Age', 'How often do you travel in a year?', 'How do you usually plan your trips?']

# Codificación simple a numérico
for col in variables:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Preparar para modelos
df = df[['AI_Tools_Use'] + variables].copy()

# Lin-Lin
model_linlin = smf.ols('AI_Tools_Use ~ Q("Age") + Q("How often do you travel in a year?") + Q("How do you usually plan your trips?")', data=df).fit()

# Log-Lin: log(Y) ~ X
df['log_Y'] = np.log(df['AI_Tools_Use'] + 1)  # +1 para evitar log(0)
model_loglin = smf.ols('log_Y ~ Q("Age") + Q("How often do you travel in a year?") + Q("How do you usually plan your trips?")', data=df).fit()

# Lin-Log: Y ~ log(X)
df['log_Age'] = np.log(df['Age'] + 1)
df['log_Travel'] = np.log(df['How often do you travel in a year?'] + 1)
df['log_Plan'] = np.log(df['How do you usually plan your trips?'] + 1)
model_linlog = smf.ols('AI_Tools_Use ~ log_Age + log_Travel + log_Plan', data=df).fit()

# Log-Log
model_loglog = smf.ols('log_Y ~ log_Age + log_Travel + log_Plan', data=df).fit()

# Mostrar resúmenes
print("Modelo Lineal-Lineal:")
print(model_linlin.summary())
print("\nModelo Log-Lineal:")
print(model_loglin.summary())
print("\nModelo Lineal-Log:")
print(model_linlog.summary())
print("\nModelo Log-Log:")
print(model_loglog.summary())
