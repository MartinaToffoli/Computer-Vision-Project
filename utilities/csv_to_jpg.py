import cv2
import pandas as pd

def csv_to_jpg(csv, dir):
    dataset=pd.read_csv(csv)
    labels=dataset.iloc[:, 0]
    data=dataset.iloc[:, 1:]
    
    for image, y in zip(data, labels):
        print(type(image))
        image = image.to_numpy()
        image = image.reshape(28, 28)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        count = count + 1
        cv2.imwrite(f'{dir}/{y}.jpg', image)

csv_to_jpg('data/sign_mnist_train.csv', 'data/original_mnist_train')
