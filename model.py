from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
import os
import cv2

def build_model(n_classes=10):
	inp = Input(shape=(28,28,1)) # 28*28 pixels = 784 flattened
	x = Conv2D(32, (3,3), activation="relu")(inp) # 32 nodes, 3-3 filter window, relu activation
	x = Conv2D(64, (3,3), activation="relu")(x) # 64 nodes
	x = MaxPooling2D((2, 2))(x) # filter the image by 2x2, so size = 28/2*28/2 or 14*14
	x = Dropout(0.25)(x)
	x = Flatten()(x) # convert them into a flat array
	x = Dense(128, activation="relu")(x) # 128 node feed forward layer
	x = Dropout(0.5)(x)
	x = Dense(n_classes, activation="softmax")(x) # get softmax prob

	model = Model(inputs=inp, outputs=x)
	model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adadelta(), metrics=["accuracy"])

	return model

def train_model(model, x, y, xt, yt, bs=128, ep=12):
	model.fit(x, y, batch_size=bs, epochs=ep, verbose=1, validation_data=(xt, yt))
	return model

def save_model(model):
	model.save("./save/model.h5")
	print("Saved model at ./save/model.h5")

def load_the_model(model):
	model = load_model("./save/model.h5")
	return model

def is_model_saved():
	if os.path.exists("./save/model.h5"):
		return True
	else:
		return False

def get_train_data():
	img_rows, img_cols = 28, 28
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255 # normalize image
	x_test /= 255 # normalize image
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)
	return (x_train, y_train), (x_test, y_test)
