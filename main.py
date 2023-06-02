import face_recognition as fr
import cv2

rohit1 = fr.load_image_file("first.jpg")
captaincool = fr.load_image_file("second.jpg")
rohit2 = fr.load_image_file("third.jpg")

rohit1 = cv2.cvtColor(rohit1, cv2.COLOR_BGR2RGB)
rohit2 = cv2.cvtColor(rohit2, cv2.COLOR_BGR2RGB)
captaincool = cv2.cvtColor(captaincool, cv2.COLOR_BGR2RGB)

enc1 = fr.face_encodings(rohit1)[0]
enc2 = fr.face_encodings(captaincool)[0]
enc3 = fr.face_encodings(rohit2)[0]

res = fr.compare_faces([enc2,enc3], enc1)

print(res)
