"""
	文件说明：用于对输入数据进行增强变换
"""
import tensorflow as tf
import numpy as np
import os
import cv2

# 训练集路径
DATA_DIR = ""
# 离线数据增强后图片保存路径
SAVE_PATH = ""
BASE_NAME = "arg0_"
# 三种不同的增强方式
METHOD_ID = 0


def image_process(ind, image):

		if ind == 0:
			image = tf.image.flip_left_right(image)
			image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
			image = tf.image.random_brightness(image, max_delta=0.4)
			image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
			image = tf.image.random_hue(image, max_delta=0.1)

		elif ind == 1:
			image = tf.image.flip_up_down(image)
			image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
			image = tf.image.random_brightness(image, max_delta=0.4)
			image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
			image = tf.image.random_hue(image, max_delta=0.1)

		elif ind == 2:
			image = tf.image.transpose_image(image)
			image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
			image = tf.image.random_brightness(image, max_delta=0.4)
			image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
			image = tf.image.random_hue(image, max_delta=0.1)

		# image = tf.image.resize_images(image, [128, 128], method=0)
		return image


def open_pic(dataset_dir, base_name, ind):
	# arg_methods = 3
	index = 1
	files_list = []
	files = os.walk(dataset_dir).__next__()[2]
	sess = tf.Session()

	for file in files:
		file_path = os.path.join(dataset_dir, file)
		files_list.append(file_path)

	print("Begin to process pictures.....")
	for image_file in files_list:
		# ind = np.random.randint(0, arg_methods)
		image = cv2.imread(image_file)
		result = image_process(ind, image)
		image = np.array(sess.run(result))
		# image = np.reshape(image, [128, 128, 3])
		image_path = SAVE_PATH + "/" + base_name + str(index) + ".png"
		cv2.imwrite(image_path, image)
		index += 1
		# plt.imshow(image)
		# plt.show()
	print("Processing Finish! There are ", index-1, " photos.")


if __name__ == "__main__":
	open_pic(DATA_DIR, BASE_NAME, METHOD_ID)