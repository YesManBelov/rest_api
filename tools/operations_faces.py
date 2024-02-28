import os
import matplotlib.pyplot as plt
import cv2
import face_recognition
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageDraw


def get_faces_coord(img):
    image = face_recognition.load_image_file(img)
    # Поиск лиц на изображении
    face_locations = face_recognition.face_locations(image)
    # Извлечение и сохранение каждого лица
    faces_locations = {}
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        faces_locations[i] = face_location
    return faces_locations


def test_age_and_gender():
    genderProto = "models/gender_deploy.prototxt"
    genderModel = "models/gender_net.caffemodel"
    ageProto = "models/age_deploy.prototxt"
    ageModel = "models/age_net.caffemodel"
    r_mean, g_mean, b_mean, = 78.4263377603, 87.7689143744, 114.895847746
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderList = ['Male ', 'Female']
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    name = 'face_jpg/normalize_faces/faces_8.jpg'
    image = cv2.imread(name)
    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (r_mean, g_mean, b_mean), swapRB=False)
    # отправляем его в нейросеть для определения пола
    genderNet.setInput(blob)
    # получаем результат работы нейросети
    genderPreds = genderNet.forward()
    # выбираем пол на основе этого результата
    gender = genderList[genderPreds[0].argmax()]
    # отправляем результат в переменную с полом
    print(f'Gender: {gender}')
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    print(f'Age: {age[1:-1]} years')


if __name__ == '__main__':
    ...
