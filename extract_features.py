import os
import cv2
import keras.backend as K
from utils import load_model
import numpy as np

def load_convnet():
	model = load_model()
	model.summary()
	return model

def find_paths(path):
	paths = []
	level_a = os.listdir(path)
	for level_name in level_a:
		for image_name in os.listdir(os.path.join(path, level_name)):
			paths += [os.path.join(path, level_name, image_name)]

	return paths

def load_image_data(path):
	im_data = cv2.imread(path)	
	if im_data.shape[0] < im_data.shape[1]:
		im_data = cv2.transpose(im_data)
		im_data = cv2.flip(im_data, flipCode=1)
	im_data = cv2.resize(im_data, (224,224))
	im_data = im_data / 255.
	return im_data

def main():
	net = load_convnet()
	get_features = K.function([net.layers[0].input, K.learning_phase()], [net.get_layer("flatten_2").output])

	base_path = "/home/ml/datasets/DeepLearningFiles"
	paths = find_paths(base_path)
	
	for image_path in paths:
		print image_path
		im_data = load_image_data(image_path)
		features = get_features([im_data[np.newaxis,...],0])[0]
		print features.shape, features.max(), features.min(), features.mean()
		cv2.imshow("frameB",im_data)
		cv2.waitKey(0)


if __name__ == "__main__":
	main()