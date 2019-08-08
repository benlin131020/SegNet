import tensorflow as tf
from model import segnet, cal_loss
from data_loader import get_iterator
from parameter import *

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
    saver = tf.train.Saver()
    sess.run(init)
    for epoch in range(EPOCHES):
        loss = 0
        for batch in range(BATCHES):
            _, loss_batch = sess.run([train_op, cross_entropy_loss])
            loss += loss_batch
        loss /= BATCHES
        print("Epoch%02d->" % (epoch+1), "loss:{:.9f}".format(loss))

    save_path = saver.save(sess, MODEL_PATH)
    print("Model saved in file: %s" % save_path)
