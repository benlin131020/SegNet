import tensorflow as tf
from data_loader import get_iterator
from parameter import *

def conv_batchnorm_relu(x, W, b, strides=1, training=False):
    conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, b)
    conv = tf.layers.batch_normalization(conv, training=training)
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
    upsample1 = upsample(pool1, pool1_indices)
    deconv1 = deconv(upsample1, weights['wd1'], training=training)
    logits = tf.nn.conv2d(deconv1, weights['wo1'], [1, 1, 1, 1], padding='SAME')
    logits = tf.nn.bias_add(logits, biases['bo1'])
    prediction = tf.nn.softmax(logits)

    return logits, prediction

weights = {
    'wc1': tf.Variable(tf.random_normal([7, 7, 3, 64])),
    'wd1': tf.Variable(tf.random_normal([7, 7, 64, 64])),
    'wo1': tf.Variable(tf.random_normal([1, 1, 64, NUM_CLASSES]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bo1': tf.Variable(tf.random_normal([NUM_CLASSES]))
}

#x = tf.constant([[[[]]]])
#x = tf.ones([1, 4, 4, 1])
#x = tf.random_normal([5, 4, 4, 3])
#x = x * 10

if __name__ == "__main__":
    #X = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL])
    #Y = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGHT * INPUT_WIDTH, NUM_CLASSES])
    iterator = get_iterator()    
    with tf.Session() as sess:
        X, Y = iterator.get_next()
        X, Y = sess.run([X, Y])
        logits, prediction = segnet(X, True)
        sess.run(tf.global_variables_initializer())
        logits, prediction = sess.run([logits, prediction])
        print(logits.shape, prediction.shape)
        #print(logits, "end")
        #print(prediction)
