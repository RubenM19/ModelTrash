import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Cargar el dataset
file_path = './dataset/qt_dataset.csv'
data = pd.read_csv(file_path)

# Preprocesamiento de datos
# Convertir 'Result' a valores binarios
data['Result'] = data['Result'].map({'Negative': 0, 'Positive': 1})

# Seleccionar las características (features) y   el objetivo (target)
X = data[['Oxygen', 'PulseRate']]
y = data['Result']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print(f"Classification Report:\n{report}")
# Guardar el modelo
# Guardar el modelo


# Función para predecir el resultado
def predict_result(oxygen, pulse_rate):
    prediction = model.predict([[oxygen, pulse_rate]])
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Ejemplo de uso de la función de predicción
print(predict_result(95, 90))

with open('./models/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Guardar el scaler
with open('./models/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)