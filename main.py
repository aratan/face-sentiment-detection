from deepface import DeepFace 

#recinocimiento
ruta ="./caras/"
verification = DeepFace.verify(img1_path = ruta + "0001.png", img2_path = ruta + "0002.png")
print("Is verified: ", verification["verified"])

# atributos
analysis = DeepFace.analyze(img_path = ruta + "0001.png", actions = ["age", "gender", "emotion", "race"]) 
print(analysis)
