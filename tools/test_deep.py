from deepface import DeepFace
import os


folder_path_test = f'face_jpg/normalize_faces/'
file_list_test = [folder_path_test + f for f in os.listdir(folder_path_test) if os.path.isfile(os.path.join(folder_path_test, f))]
print(file_list_test)




# folder_path = f'face_jpg/gp_faces/'
# file_list_check = [folder_path + f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# print(file_list_check)




# for goup in file_list_group:
#     for bred in file_list_name:
#         obj = DeepFace.verify(goup, bred, model_name = 'ArcFace', enforce_detection=False)
#         print(f"{goup} || {bred}\n{obj['verified']}\n")