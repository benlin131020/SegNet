import tensorflow as tf
import os

BATCH_SIZE = 5

def filename_list(path):
    '''return a list of file names'''
    return [os.path.join(path, f) for f in os.listdir(path)]

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image

if __name__ == "__main__":
    train_path = './dataset/train/'
    trainannot_path = './dataset/trainannot'
    train_filenames = filename_list(train_path)
    trainannot_filenames = filename_list(trainannot_path)

    dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
    dataset = dataset.map(_parse_function).shuffle(buffer_size=1000).batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    images = iterator.get_next()

    with tf.Session() as sess:
        print(sess.run(images).shape)
