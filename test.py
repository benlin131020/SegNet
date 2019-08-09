import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import segnet, cal_loss, cal_accuracy
from data_loader import get_train_iterator, get_test_iterator
from parameter import *

iterator = get_train_iterator()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    X, Y = iterator.get_next()
    X.set_shape([None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL])
    Y.set_shape([None, INPUT_HEIGHT * INPUT_WIDTH, NUM_CLASSES])
    logits, prediction = segnet(X, True)
    cross_entropy_loss = cal_loss(logits, Y)
    accuracy = cal_accuracy(prediction, Y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, MODEL_PATH)
    image, pred, loss, acc = sess.run([X, prediction, cross_entropy_loss, accuracy])
    image = image.astype(int)
    for y in range(INPUT_HEIGHT):
        for x in range(INPUT_WIDTH):
            if pred[0, y, x, 0] < pred[0, y, x, 1]:
                image[0, y, x, 0] = 255
                image[0, y, x, 1] = 0
                image[0, y, x, 2] = 0
    print("loss:{:.9f}".format(loss), "accuracy:{:.9f}".format(acc))
    plt.imshow(image[0])
    plt.show()
    #plt.savefig("fig.png")
