import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

def process_images(folder_path, excel_path):
    # Load Excel file into a pandas DataFrame
    df = pd.read_excel(excel_path)

    # Lists to store images and labels
    images_list = []
    labels_list = []
    count = 0

    # Iterate through image files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            count += 1
            if count % 1000 == 0 : 
                print(count)
            # Extract the first 10 characters from the image filename
            image_prefix = filename[:9]
            
            # Find the corresponding row in the DataFrame where 'name' column matches the image prefix
            matching_row = df[df['name'] == int(image_prefix)]

            if not matching_row.empty:
                # Extract the label from the 'formulation' column
                label = matching_row['formulation'].values[0]

                # Copy the image to the output folder with the label as part of the filename
                source_path = os.path.join(folder_path, filename)
                img = cv2.imread(source_path)
                img = cv2.resize(img, (224, 224))

                # Append image and label to the lists
                images_list.append(img)
                labels_list.append(label)

    return np.array(images_list), np.array(labels_list)

# 데이터셋 생성 및 전처리
folder_path = './train/boxes'
excel_path = 'data_info_final.xlsx'
num_classes = 10

images, labels = process_images(folder_path, excel_path)

# 데이터를 train과 test로 나누기
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# CNN 모델 정의
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

_, accuracy = model.evaluate(X_test, y_test)

print(accuracy)