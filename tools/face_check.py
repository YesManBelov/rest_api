import os
import cv2
import ArcFace
import face_recognition
import numpy as np
from ArcFace import ArcFaceClient
from PIL import Image
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import normalize
from deepface import DeepFace

model = ArcFaceClient()


def detect_faces(image_path):
    # Загрузка изображения
    image = face_recognition.load_image_file(image_path)

    # Поиск лиц на изображении
    face_locations = face_recognition.face_locations(image)

    # Извлечение и сохранение каждого лица
    face_images = []
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location

        # Извлечение области лица из изображения
        face_image = image[top:bottom, left:right]
        face_images.append(face_image)

    return face_images


def create_combined_descriptor(image):
    # Инициализация пустого дескриптора
    preprocessed_image = preprocess_image(image)
    descriptor = model.find_embeddings(preprocessed_image)
    return descriptor


def load_image(filename):
    image = cv2.imread(filename)
    return image


def preprocess_image(image):
    # Изменение размера изображения до (112, 112)
    resized_image = cv2.resize(image, (112, 112))
    batched_image = np.expand_dims(resized_image, axis=0)
    return batched_image


def get_path_photos(name):
    folder_path = f'face_jpg/{name}/'
    file_list = [folder_path + f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return file_list


def get_descriptors_man(name):
    photos = get_path_photos(name)
    # получение объектов images
    images = []
    for photo in photos:
        faces = detect_faces(photo)  ##### добавить проверку, что лицо одно
        print(f'Лиц на фото {photo}: {len(faces)}')
        images.append(faces[0])

    # получение дескрипторов
    descriptors = []
    for image in images:
        descriptors.append(create_combined_descriptor(image))
    print('перед усреднением')
    print(type(descriptors[0]))
    # усреднение
    average_descriptor = np.mean(descriptors, axis=0)
    return average_descriptor


def get_descriptors_group(path):
    # получение объектов images
    images = detect_faces(path)  ##### добавить проверку, что лицо одно
    print(f'Лиц на фото {path}: {len(images)}')

    # получение дескрипторов
    descriptors = []
    for id, image in enumerate(images):
        # сохранение лица для проверки
        face_image_path = os.path.join('face_jpg/gp_faces', f"face_{id + 1}.jpg")
        Image.fromarray(image).save(face_image_path)

        # сохраняем дескриптор лица
        descriptors.append(create_combined_descriptor(image))
    return descriptors


def main():
    # создание дескриптора для конкретного человека
    bred_descriptor = get_descriptors_man('bred')
    print(type(bred_descriptor))

    # создание списка дескрипторов для каждого человека на групповом фото
    group_descriptors = get_descriptors_group('face_jpg/group/group_b_p_2.jpg')





    # bred_descriptor = np.array(bred_descriptor).reshape(1,-1)
    # bred_descriptor = normalize(bred_descriptor)
    # group_descriptors = list(map(lambda x: normalize(np.array(x).reshape(1,-1)), group_descriptors))
    # сравниваем дескрипторы
    # for group_man_descriptor in group_descriptors:
    #     distance = np.dot(group_man_descriptor, bred_descriptor) / (np.linalg.norm(group_man_descriptor) * np.linalg.norm(bred_descriptor))
    #     print(distance)


main()
