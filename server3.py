from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import uvicorn
import pickle
from pydantic import BaseModel

app_desc = """<h2>Subida de Imagenes `predict/image`</h2>
<h2>BASURA DETECTION</h2>"""

app = FastAPI(title='PROYECTO BASURITA', description=app_desc)

# Variables globales para el modelo de imagen, el modelo Random Forest y el scaler
image_model = None
rf_model = None
scaler = None

# Carga del modelo de imagen
try:
    image_model = load_model("./models/model3.keras")
    input_shape = image_model.input_shape[1:3]  # Obtener el tamaño de entrada del modelo (altura y ancho)
    print(f"Modelo de imagen cargado correctamente. Tamaño de entrada esperado: {input_shape}")
except Exception as e:
    print(f"Error al cargar el modelo de imagen: {e}")

# Carga del modelo de Random Forest y el scaler usando pickle
try:
    with open('./models/random_forest_model.pkl', 'rb') as model_file:
        rf_model = pickle.load(model_file)
    print("Modelo de Random Forest cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo de Random Forest: {e}")
    rf_model = None

try:
    with open('./models/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el scaler: {e}")
    scaler = None

def prepare_image(img: Image.Image, target_size: tuple) -> np.ndarray:
    """Prepara la imagen para la predicción."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalización
    return img_array

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    # Verifica que el archivo sea una imagen
    print(file.content_type)
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")
    try:
        # Lee la imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Prepara la imagen
        img_array = prepare_image(img, target_size=input_shape)  # Usar el tamaño de entrada del modelo

        # Realiza la predicción
        prediction = image_model.predict(img_array)
        predicted_class = "trash" if prediction[0][0] > 0.5 else "clean"
        print(prediction)
        print(prediction[0][0])
        return {"prediction": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {e}")

class PredictValues(BaseModel):
    oxygen: float
    pulse_rate: float

@app.post("/predict/values")
async def predict_values(values: PredictValues):
    try:
        if scaler is None:
            raise ValueError("Scaler no está cargado correctamente.")
        if rf_model is None:
            raise ValueError("Modelo de Random Forest no está cargado correctamente.")
        
        # Normalizar los datos de entrada
        input_data = np.array([[values.oxygen, values.pulse_rate]])
        scaled_input = scaler.transform(input_data)

        # Realizar la predicción
        prediction = rf_model.predict(scaled_input)
        result = 'Positive' if prediction[0] == 1 else 'Negative'
        
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando los valores: {e}")

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
