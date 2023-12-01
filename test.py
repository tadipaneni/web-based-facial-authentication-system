import cv2
import face_recognition

def find_face_encodings(image_path):
    # reading image
    image = cv2.imread(image_path)
    # get face encodings from the image
    face_enc = face_recognition.face_encodings(image)
    # return face encodings
    return face_enc[0]

# getting face encodings for the first image
image_1 = find_face_encodings("./static/face/unknown/4.jpg")

# getting face encodings for the second image
image_2 = find_face_encodings("./static/face/4.jpg")

# checking if both images are same
is_same = face_recognition.compare_faces([image_1], image_2)[0]

print(f"Are the images same? {is_same}")

if is_same:
    # finding the distance level between images
    distance = face_recognition.face_distance([image_1], image_2)
    distance = round(distance[0] * 100)
    
    # calculating accuracy level between images
    accuracy = 100 - round(distance)
    
    if accuracy > 60:
        print("The images are same")
    else:
        print("The images are not same")
        print(f"Accuracy Level: {accuracy}%")
else:
    print("The images are not same")