#! python36
# python36 -m pip install keras==2.4.3 --user
# python36 -m pip install tensorflow==2.3.0 --user
# python36 -m pip install mtcnn==0.1.0 --user
# python36 -m pip list

# link for dataset
# https://techlearn-cdn.s3.amazonaws.com/bs_face_recognition/dataset.zip
# link for Facenet model
# https://techlearn-cdn.s3.amazonaws.com/bs_face_recognition/facenet.h5
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
face_dataset_folder = "dataset/"

#1 Finding all people present inside our dataset

for folder in os.listdir(face_dataset_folder):
    print(folder)

#2 Visualizing some images from each folder

import matplotlib.pyplot as plt
import cv2
import numpy as np

plt.figure(figsize = (15 , 10))
counter = 0

for folder in os.listdir(face_dataset_folder):
    folder_counter = 0
    for image_file in os.listdir(face_dataset_folder + folder):
        if folder_counter == 4:
            break
        image = cv2.imread(face_dataset_folder + folder + '/' + image_file, 0)
        
        plt.subplot(4, 4, counter + 1)
        plt.axis('off')
        plt.imshow(image, cmap = "gray")
        plt.title(folder)
        
        counter = counter + 1
        folder_counter = folder_counter + 1

plt.show()
#gray is because matplotlib works in 2d so third layer which is blue is eliminated from the images.

#3 Detecting Faces in an Image

import mtcnn

# Creating an instance of MTCNN Face Detector
detector = mtcnn.MTCNN()

def face_detection(filename):
    all_faces = detector.detect_faces(plt.imread(filename))
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    #gca is the coordinates on which we have to draw the rectangle - get current axis
    for result in all_faces:
        x, y, width, height = result['box']
        print(result)
        rect = plt.Rectangle((x, y), width, height, fill = False, color = 'blue')
        ax.add_patch(rect)
        #Add rectangle on the image
        plt.show()

face_detection('dataset/Amitabh Bachchan/pic1.jpg')

#4 Finding Embeddings for an Image

#4a Loading Facenet model to find embeddings
from tensorflow.keras.models import load_model

facnet_path = 'facenet.h5'
required_size = (160, 160)
facenet = load_model(facnet_path)

#4b Finding face in image and passing it to facenet to get its embedding
def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

encodes = []
file_name = 'dataset/Amitabh Bachchan/pic1.jpg'

img = cv2.imread(file_name)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = detector.detect_faces(img_rgb)
if results:
    res = max(results, key = lambda b: b['box'][2] * b['box'][3]) # If they are many faces, he will take 
    face, _, _ = get_face(img_rgb, res['box'])                    # with max width and height
    face = normalize(face)
    face = cv2.resize(face, required_size) #160*160 dimension
    encode = facenet.predict(np.expand_dims(face, axis=0))[0] #128 vectors

print(encode)

#5 Creating Embeddings for all Images and storing it under a Dictionary

people_name_to_digit = {"Akshay Kumar" : 0,
                        "Amitabh Bachchan" : 1,
                        "Katrina Kaif" : 2,
                        "Narendra Modi" : 3}

people_digit_to_name = {0 : "Akshay Kumar",
                        1: "Amitabh Bachchan",
                        2: "Katrina Kaif",
                        3: "Narendra Modi"}

embeddings_dic = {}

from sklearn.preprocessing import Normalizer
# the face print of Amitabh Bachapan in 15 pictures will be different but mean will be same, so to process it faster we will process it.
l2_normalizer = Normalizer('l2')

for folder in os.listdir(face_dataset_folder):
    encodes = list()
    for image_file in os.listdir(face_dataset_folder + folder):
        
        # loading images from each folder
        image = cv2.imread(face_dataset_folder + folder + '/' + image_file)
        image = cv2.resize(image, (160, 160))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # finding face from each loaded image
        results = detector.detect_faces(image)
        if results:
            res = max(results, key = lambda b: b['box'][2] * b['box'][3])
            face, _, _ = get_face(img_rgb, res['box'])
            face = normalize(face)
            face = cv2.resize(face, required_size)
            encode = facenet.predict(np.expand_dims(face, axis=0))[0]
            encodes.append(encode)
    
    # finding mean of all encodings and saving it in embedding_dic
    if encodes:
        encode = np.sum(encodes, axis=0) # column wise sum
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        embeddings_dic[people_name_to_digit.get(folder)] = encode

print('Done finding embeddings for all persons')

print(embeddings_dic)

#6 Recognizing people in Image using Facenet and pre stored embeddings

from scipy.spatial.distance import cosine

def recognize_person(test_image_path, recognition_t = 1.0, confidence_t = 0.95, required_size = (160, 160)):
    test_image = cv2.imread(test_image_path)
    img_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        face = normalize(face)
        face = cv2.resize(face, required_size)
        encode = facenet.predict(np.expand_dims(face, axis=0))[0]
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'Unknown'
        distance = float("inf")
        
        for db_name, db_encode in embeddings_dic.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = people_digit_to_name.get(db_name)
                distance = dist
        
        print(name)
        cv2.rectangle(test_image, pt_1, pt_2, (255, 0, 0), 2)
        cv2.putText(test_image, name, (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 1)
    return test_image

window_name = 'test_image'
cv2.imshow(window_name, recognize_person('test1.png'))

#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

cv2.imshow(window_name, recognize_person('test2.png'))
cv2.waitKey(0)