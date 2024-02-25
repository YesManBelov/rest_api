import os
import face_recognition
from PIL import Image
import cv2
import numpy as np

name = 'bred'
folder_path = f'face_jpg/{name}/'
file_list = [folder_path + f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
print(file_list)
for i, ph in enumerate(file_list):
    image = face_recognition.load_image_file(ph)
    face_locations = face_recognition.face_locations(image)
    top, right, bottom, left = face_locations[0]
    # Извлечение области лица из изображения
    face_image = image[top:bottom, left:right]
    print(type(face_image))
    face_image = cv2.resize(face_image, (224, 224))
    print(type(face_image))
    face_image_path = os.path.join('face_jpg/normalize_faces', f"faces_{i + 1}.jpg")
    Image.fromarray(face_image).save(face_image_path)