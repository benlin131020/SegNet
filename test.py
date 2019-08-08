import tensorflow as tf
from model import segnet, cal_loss
from data_loader import get_iterator
from parameter import *

def foo(X):
    
    pass

iterator = get_iterator()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    X, Y = iterator.get_next()
    X.set_shape([BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL])
    Y.set_shape([BATCH_SIZE, INPUT_HEIGHT * INPUT_WIDTH, NUM_CLASSES])

    logits, prediction = segnet(X, True)
    cross_entropy_loss = cal_loss(logits, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(cross_entropy_loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(EPOCHES):
        for batch in range(BATCHES):
            sess.run(train_op)
            image, label = sess.run([X, Y])
            print(image[0][0][0], label[0][0])
            