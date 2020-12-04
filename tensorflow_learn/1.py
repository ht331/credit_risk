import tensorflow as tf

message = tf.constant('Welcome to the exciting world of Deep Neural Networks')
with tf.compat.v1.Session() as sess:
    print(sess.run(message).decode())
    