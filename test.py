import tensorflow as tf
import cv2
import numpy as np
import pathlib
import os

def filename_list(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

train_path = './dataset/train/'
filenames = filename_list(train_path)
print(filenames)
'''
filenames = []
for i in range(1, 200):
    train_path = "./dataset/train/frame" + str(i) + ".jpg"
    if path.isfile(train_path):
        filenames.append(train_path)
'''
# step 2: create a dataset returning slices of `filenames`
dataset = tf.data.Dataset.from_tensor_slices(filenames)

# step 3: parse every image in the dataset using `map`
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image

dataset = dataset.map(_parse_function)
dataset = dataset.batch(5)

# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
images = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(images).shape)


'''
train_img = cv2.imread('./dataset/train/frame1.jpg')
train_tensor = tf.convert_to_tensor(train_img)
label_img = cv2.imread('./dataset/trainannot/frame17.png', 0)
label_tensor = tf.convert_to_tensor(label_img)
label_tensor = tf.reshape(label_tensor, [label_tensor.shape[0], label_tensor.shape[1], 1])
train_img = train_img / 255.0
print(train_tensor.shape)
print(train_tensor.dtype)
print(label_tensor.shape)
print(label_tensor.dtype)
#print(train_tensor.numpy().min(), train_tensor.numpy().max())
#train_dataset = tf.data.Dataset.from_tensor_slices(train_tensor, label_tensor)
#for pixel in label_img:
#    print(pixel)
#cv2.imshow("img", label_img)
#cv2.waitKey(0)
'''
