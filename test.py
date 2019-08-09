import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import segnet, cal_loss
from data_loader import get_iterator
from parameter import *

iterator = get_iterator(training=False)

with tf.Session() as sess:
    sess.run(iterator.initializer)
    X, Y = iterator.get_next()
    X.set_shape([BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL])
    Y.set_shape([BATCH_SIZE, INPUT_HEIGHT * INPUT_WIDTH, NUM_CLASSES])
    logits, prediction = segnet(X, True)
    #cross_entropy_loss = cal_loss(logits, Y)
    #optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    #train_op = optimizer.minimize(cross_entropy_loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, MODEL_PATH)
    image, softmax = sess.run([logits, prediction])
    print(softmax[0][0][0])
    #image = np.reshape(image, [])
