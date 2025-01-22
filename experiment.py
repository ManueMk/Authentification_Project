import face_recognition as fr
import cv2
import numpy as np
import os

path = "/home/manuemk/Documents/M2/VISION_PO/authentification_projet_init/img/"

known_names = []
known_name_encodings = []

# Loop through all the images in the directory
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = fr.load_image_file(image_path)
            encoding = fr.face_encodings(image)[0]
            known_name_encodings.append(encoding)
            known_names.append(folder.capitalize())

# Print the known names
print(known_names)

test_image = "/home/manuemk/Documents/M2/VISION_PO/authentification_projet_init/test/takou/20220925_163923.jpg"
image = cv2.imread(test_image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)
print(face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)
    name = ""

    face_distances = fr.face_distance(known_name_encodings, face_encoding)
    # best_match = np.argmin(face_distances)
    # print(best_match)

    # if matches[best_match]:
    #     name = known_names[best_match]

    # cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    # cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
    # font = cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    best_match_index = np.argmin(face_distances)
    best_match_distance = face_distances[best_match_index]
    best_match_percentage = (1 - best_match_distance) * 100


    if matches[best_match_index]:
        name = known_names[best_match_index]
    else:
        name = "Unknown"

    print(best_match_percentage, name)    

    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, f"{name}: {best_match_percentage:.2f}%", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


cv2.imshow("Result", image)
cv2.imwrite("./output8.jpg", image)
key = cv2.waitKey(2000) & 0xFF  # Wait for 2 seconds or a key press
if key == ord("q"):
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()  # Close window after timeout or any key press


