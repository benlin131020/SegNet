import tensorflow as tf
import os
from parameter import *

def filename_list(path):
    '''return a list of file names'''
    return [os.path.join(path, f) for f in os.listdir(path)]

def _parse_function(image_name, label_name):
    image_string = tf.read_file(image_name)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    
    label_string = tf.read_file(label_name)
    label_decoded = tf.image.decode_jpeg(label_string, channels=3)
    label_gray = tf.image.rgb_to_grayscale(label_decoded)
    label_flat = tf.reshape(label_gray, [-1])
    label = tf.one_hot(label_flat, NUM_CLASSES)
    return image, label

def get_train_iterator():
    img_filenames = filename_list(TRAIN_PATH)
    annot_filenames = filename_list(TRAINANNOT_PATH)
    dataset = tf.data.Dataset.from_tensor_slices((img_filenames, annot_filenames))
    dataset = dataset.map(_parse_function).shuffle(buffer_size=100).batch(BATCH_SIZE).repeat()
    iterator = dataset.make_initializable_iterator()
    return iterator

def get_test_iterator():
    img_filenames = filename_list(TEST_PATH)
    annot_filenames = filename_list(TESTANNOT_PATH)
    dataset = tf.data.Dataset.from_tensor_slices((img_filenames, annot_filenames))
    dataset = dataset.map(_parse_function).batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    return iterator

if __name__ == "__main__":
    iterator = get_train_iterator()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        X, Y = iterator.get_next()
        sess.run(tf.global_variables_initializer())
        x, y = sess.run([X, Y])
        print(x.shape, y.shape)
