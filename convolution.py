import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import cv2
from keras.utils import np_utils
import numpy as np
from scipy.signal import convolve2d
import random
import os
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

#os.removedirs("image_folder")
#os.mkdir("image_folder")

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()


def plot_images_labels_prediction(images, labels,
                                  prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx += 1
    plt.show()


def relu_forward(x):

    out = None
    relu = lambda x: x * (x > 0).astype(float)
    out = relu(x)
    catch = x
    return out, catch


def generate_random_mtx(img):

    tmp_list = []
    row = random.randint(1, 26)
    cloun = random.randint(1, 26)
    tmp_list = [img[row - 1][cloun - 1], img[row][cloun - 1], img[row + 1][cloun - 1],
                img[row - 1][cloun],     img[row][cloun],     img[row + 1][cloun],
                img[row - 1][cloun + 1], img[row][cloun + 1], img[row + 1][cloun + 1],
                ]
    return np.array(tmp_list)


keras.datasets.mnist.load_data()
(x_Train, y_Train), (x_Test, y_Test) = keras.datasets.mnist.load_data()

print('x_train_image:',x_Train.shape)
print('y_train_label:',y_Train.shape)

print('x_test_image:',x_Test.shape)
print('y_test_label:',y_Test.shape)

cv2.imwrite("image_folder/origin.png",x_Train[0])
x_Train4D = x_Train.reshape(x_Train.shape[0], 28, 28, 1).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0], 28, 28, 1).astype('float32')

# normalize 0~1
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

test_norm = x_Train[0] / 255
#print(test_norm)
# w h = 25
#a = np.array((-1, 0, 1, 1, 0, -1, 1, 0, 1))

for i in range(16):
    tmp = generate_random_mtx(test_norm).reshape(3, 3)
    test_img = convolve2d(test_norm, tmp, mode='same')
    t, z = relu_forward(test_img)
    recover = t * 255.
    cv2.imwrite(f"image_folder/test_conv_{i}.png", recover)

# model = Sequential()
# #filter為16, Kernel size為(5,5),Padding為(same)
# model.add(Conv2D(filters=16,
#                  kernel_size=(5, 5),
#                  padding='same',
#                  input_shape=(28, 28 ,1),
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=36,
#                  kernel_size=(5,5),
#                  padding='same',
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# # Drop掉部分神經元避免over fitting
# model.add(Dropout(0.25))
#
# # 平坦化
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10,activation='softmax'))
# print(model.summary())
