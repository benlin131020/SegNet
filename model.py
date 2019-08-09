import tensorflow as tf
from data_loader import get_iterator
from parameter import *

def conv_batchnorm_relu(x, W, b, strides=1, training=False):
    conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, b)
    conv = tf.keras.layers.BatchNormalization()(conv)
    #conv = tf.layers.batch_normalization(conv, training=training)
    return tf.nn.relu(conv)

def deconv(x, W, strides=1, training=False):
    conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return tf.layers.batch_normalization(conv, training=training)

def upsample(pooled, ind, ksize=[1, 2, 2, 1]):
    """
      To unpool the tensor after  max_pool_with_argmax.
      Argumnets:
          pooled:    the max pooled output tensor
          ind:       argmax indices , the second output of max_pool_with_argmax
          ksize:     ksize should be the same as what you have used to pool
      Returns:
          unpooled:      the tensor after unpooling
      Some points to keep in mind ::
          1. In tensorflow the indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes flattened index ((b * height + y) * width + x) * channels + c
          2. Due to point 1, use broadcasting to appropriately place the values at their right locations ! 
    """
    # Get the the shape of the tensor in th form of a list
    input_shape = pooled.get_shape().as_list()
    # Determine the output shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # Ceshape into one giant tensor for better workability
    pooled_ = tf.reshape(pooled, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
    # The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes flattened index ((b * height + y) * width + x) * channels + c
    # Create a single unit extended cuboid of length bath_size populating it with continous natural number from zero to batch_size
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.concat([b_, ind_],1)
    ref = tf.Variable(tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]))
    # Update the sparse matrix with the pooled values , it is a batch wise operation
    unpooled_ = tf.scatter_nd_update(ref, ind_, pooled_)
    # Reshape the vector to get the final result 
    unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
    return unpooled

def segnet(x, training):
    conv1 = conv_batchnorm_relu(x, weights['wc1'], biases['bc1'], training=training)
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = conv_batchnorm_relu(pool1, weights['wc2'], biases['bc2'], training=training)
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = conv_batchnorm_relu(pool2, weights['wc3'], biases['bc3'], training=training)
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv4 = conv_batchnorm_relu(pool3, weights['wc4'], biases['bc4'], training=training)
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    upsample4 = upsample(pool4, pool4_indices)
    deconv4 = deconv(upsample4, weights['wd4'], training=training)
    upsample3 = upsample(deconv4, pool3_indices)
    deconv3 = deconv(upsample3, weights['wd3'], training=training)
    upsample2 = upsample(deconv3, pool2_indices)
    deconv2 = deconv(upsample2, weights['wd2'], training=training)
    upsample1 = upsample(deconv2, pool1_indices)
    deconv1 = deconv(upsample1, weights['wd1'], training=training)

    logits = tf.nn.conv2d(deconv1, weights['wo1'], [1, 1, 1, 1], padding='SAME')
    logits = tf.nn.bias_add(logits, biases['bo1'])
    prediction = tf.nn.softmax(logits)

    return logits, prediction

def cal_loss(logits, label):
    logits_flat = tf.reshape(logits, [-1, NUM_CLASSES])
    label_flat = tf.reshape(label, [-1, NUM_CLASSES])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_flat, labels=label_flat))
    return loss

def cal_accuracy(pred, label):
    pred_flat = tf.reshape(pred, [-1, NUM_CLASSES])
    label_flat = tf.reshape(label, [-1, NUM_CLASSES])
    correct_prediction = tf.equal(tf.argmax(pred_flat, 1), tf.argmax(label_flat, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

weights = {
    'wc1': tf.Variable(tf.random_normal([7, 7, 3, 64]), name='wc1'),
    'wc2': tf.Variable(tf.random_normal([7, 7, 64, 64]), name='wc2'),
    'wc3': tf.Variable(tf.random_normal([7, 7, 64, 64]), name='wc3'),
    'wc4': tf.Variable(tf.random_normal([7, 7, 64, 64]), name='wc4'),
    'wd4': tf.Variable(tf.random_normal([7, 7, 64, 64]), name='wd4'),
    'wd3': tf.Variable(tf.random_normal([7, 7, 64, 64]), name='wd3'),
    'wd2': tf.Variable(tf.random_normal([7, 7, 64, 64]), name='wd2'),
    'wd1': tf.Variable(tf.random_normal([7, 7, 64, 64]), name='wd1'),
    'wo1': tf.Variable(tf.random_normal([1, 1, 64, NUM_CLASSES]), name='wo1')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([64]), name='bc2'),
    'bc3': tf.Variable(tf.random_normal([64]), name='bc3'),
    'bc4': tf.Variable(tf.random_normal([64]), name='bc4'),
    'bo1': tf.Variable(tf.random_normal([NUM_CLASSES]), name='bo1')
}

if __name__ == "__main__":
    iterator = get_iterator()

    with tf.Session() as sess:
        X, Y = iterator.get_next()
        X, Y = sess.run([X, Y])
        logits, prediction = segnet(X, True)
        cross_entropy_loss = cal_loss(logits, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(cross_entropy_loss)
        sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCHES):
            for batch in range(BATCHES):
                sess.run(train_op)
            loss = sess.run(cross_entropy_loss)
            print("Epoch%02d->" % (epoch+1), "loss:{:.9f}".format(loss))
