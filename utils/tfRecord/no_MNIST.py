import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

FILE_DIR = "notMNIST_large"
BATCH_SIZE = 25
SAVE_DIR = "save"
NAME_TFRECORD = "test"


def get_file(file_dir):
	images = []
	temp = []
	for root, sub_folders, files in os.walk(file_dir):
		for name in files:
			images.append(os.path.join(root, name))
		for sub_dir in sub_folders:
			temp.append(os.path.join(root, sub_dir))

	# 添加label
	labels = []
	for one_folder in temp:
		n_img = len(os.listdir(one_folder))
		letter = one_folder.split('/')[-1]

		if letter == "A":
			labels = np.append(labels, n_img*[1])
		elif letter == "B":
			labels = np.append(labels, n_img*[2])
		elif letter == "C":
			labels = np.append(labels, n_img*[3])
		elif letter == "D":
			labels = np.append(labels, n_img*[4])
		elif letter == "E":
			labels = np.append(labels, n_img*[5])
		elif letter == "F":
			labels = np.append(labels, n_img*[6])
		elif letter == "G":
			labels = np.append(labels, n_img*[7])
		elif letter == "H":
			labels = np.append(labels, n_img*[8])
		elif letter == "I":
			labels = np.append(labels, n_img*[9])
		else:
			labels = np.append(labels, n_img*[10])

	temp = np.array([images, labels])
	temp = temp.transpose()
	np.random.shuffle(temp)
	image_list = list(temp[:, 0])
	label_list = list(temp[:, 1])
	label_list = [int(float(i)) for i in label_list]

	return image_list, label_list


# 生成整数型的数据
def int64_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 生成字符类型数据
def bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 将图片和标签写入文件
def convert_to_tfrecord(images, labels, save_dir, name):
	file_name = os.path.join(save_dir, name + '.tfrecords')
	n_samples = len(labels)

	if np.shape(images)[0] != n_samples:
		raise ValueError("Image size %d does not match label size %d." % (images.shape[0], n_samples))

	writer = tf.python_io.TFRecordWriter(file_name)
	print("\nTransform start......")
	for i in range(n_samples):
		try:
			image = io.imread(images[i])
			image_raw = image.tostring()
			label = int(labels[i])
			example = tf.train.Example(features=tf.train.Features(feature={
				'label': int64_feature(label),
				'image_raw': bytes_feature(image_raw)
			}))
			writer.write(example.SerializeToString())
		except IOError as e:
			print("Cound not read:", images[i])
			print('error: %s'% e)
			print("Skip it!\n")
	writer.close()
	print("Transform done!")


# 生成tfReocord文件（只需要执行一次）
# image_list, label_list = get_file(FILE_DIR)
# convert_to_tfrecord(image_list, label_list, SAVE_DIR, NAME_TFRECORD)


# 从tfrecords中读取数据
def read_and_decode(tfrecords_file, batch_size):
	# make an input queue from tfrecord file
	filename_queue = tf.train.string_input_producer([tfrecords_file])

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	img_features = tf.parse_single_example(serialized_example,
										   features={
											   'label': tf.FixedLenFeature([], tf.int64),
											   'image_raw': tf.FixedLenFeature([], tf.string),
										   })
	image = tf.decode_raw(img_features['image_raw'], tf.uint8)

	image = tf.reshape(image, [28, 28])
	label = tf.cast(img_features['label'], tf.int32)
	image_batch, label_batch = tf.train.batch([image, label],
											  batch_size=batch_size,
											  num_threads=16,
											  capacity=2000)
	return image_batch, tf.reshape(label_batch, [batch_size])


def plot_images(images, labels):
	# plot one batch size
	for i in range(BATCH_SIZE):
		plt.subplot(5, 5, i+1)
		plt.axis('off')
		plt.title(chr(ord('A') + labels[i] - 1), fontsize=14)# ord('A')返回A的ASCII码值， chr()函数为依据ASCII码返回对应字符
		plt.subplots_adjust(top=1.5)
		plt.imshow(images[i])
	plt.show()


tfrecords_file = "/home/guo/workspace/tfRecord/save/test.tfrecords"
image_batch, label_batch = read_and_decode(tfrecords_file, BATCH_SIZE)

with tf.Session() as sess:
	i = 0
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	try:
		while not coord.should_stop() and i<1:
			image_list, label_list = sess.run([image_batch, label_batch])
			plot_images(image_list, label_list)
			i += 1
	except tf.errors.OutOfRangeError:
		print('done')
	finally:
		coord.request_stop()
	coord.join(threads)



