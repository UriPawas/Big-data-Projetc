import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo Excel
file_path = "Big data analysis Tourism.xlsx"  # Asegúrate de tener el archivo en el mismo directorio
df = pd.read_excel(file_path, sheet_name='Ответы на форму (1)')

# Mapear la columna objetivo (uso de herramientas de IA)
ai_use_mapping = {
    "Very unlikely": 0,
    "Unlikely": 1,
    "Neutral": 2,
    "Likely": 3,
    "Very likely": 4
}
df['AI_Tools_Use'] = df['How likely are you to use AI tools (like ChatGPT or travel chatbots) to plan a trip?'].map(ai_use_mapping)

# Eliminar filas sin dato objetivo
df = df.dropna(subset=['AI_Tools_Use'])

# Seleccionar variables predictoras
features = [
    'Gender', 'Age', 'How often do you travel in a year?',
    'Which of the following best describes your main purpose for travel?',
    'When choosing a destination, which factor is most important to you?',
    'What type of accommodation do you usually prefer when traveling?',
    'What is your preferred mode of transportations when traveling long distances?',
    'How do you usually plan your trips?',
    'What are your biggest challenges while planning a trip?',
    'What would encourage you to travel more often?',
    'How do you usually share your travel experiences?'
]

X = df[features]
y = df['AI_Tools_Use']

# Preprocesador para variables categóricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, features)
    ]
)

# Crear pipeline con Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Extraer importancia de variables
feature_names = model.named_steps['preprocessor'].transformers_[0][1].named_steps['encoder'].get_feature_names_out(features)
importances = model.named_steps['classifier'].feature_importances_

# Crear DataFrame con las importancias
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Mostrar top 15 características más importantes
print("\nTop características que predicen el uso de IA:")
print(importance_df.head(15))

# Gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', palette='viridis')
plt.title("Características más influyentes en el uso de herramientas de IA para viajes")
plt.tight_layout()
plt.show()
