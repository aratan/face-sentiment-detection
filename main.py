# Importamos las librerías necesarias

import cv2
from deepface import DeepFace 
import pandas as pd

# Ruta donde guardaremos las imágenes capturadas
ruta_imagenes ="./caras/"

# Diccionario con la relación de imágenes 
# y nombres de usuarios autorizados
usuario = {
    "0001.png": "victor"
    }

# Capturamos varias imágenes
camera = cv2.VideoCapture(0)
for i in range(3,10):
    return_value, image = camera.read()
    cv2.imwrite(ruta_imagenes + '000'+str(i)+'.png', image)
# cerramos camara
del(camera)

metrics = ["cosine","euclidean"]

# Métricas de comparación
for metric in metrics:        
    verification = DeepFace.verify(
    img1_path = ruta_imagenes + "0001.png", 
    img2_path = ruta_imagenes + "0003.png",
    distance_metric = metric)

        

# Análisis de atributos faciales
analysis = DeepFace.analyze(
    img_path = ruta_imagenes + "0003.png", 
    actions = ["age", "gender", "emotion", "race"]
    ) 


# Imprimimos los resultados     
print("Is verified: " + usuario["0001.png"], verification["verified"])
print("Edad: ", analysis["age"])
print("Sexo: ", analysis["gender"])
print("Emoción: ", analysis["dominant_emotion"])
print("Etnia: ", analysis["dominant_race"])
