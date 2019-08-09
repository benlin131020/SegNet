import tensorflow as tf
from model import segnet, cal_loss, cal_accuracy
from data_loader import get_train_iterator
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
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(cross_entropy_loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    for epoch in range(EPOCHES):
        loss = 0
        acc = 0
        for batch in range(BATCHES):
            _, loss_batch, acc_batch = sess.run([train_op, cross_entropy_loss, accuracy])
            loss += loss_batch
            acc += acc_batch
        loss /= BATCHES
        acc /= BATCHES
        print("Epoch{:0>2d}->".format(epoch+1), "loss:{:.9f}".format(loss), "accuracy:{:.9f}".format(acc))

    save_path = saver.save(sess, MODEL_PATH)
    print("Model saved in file: %s" % save_path)
