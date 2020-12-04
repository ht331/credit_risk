import tensorflow as tf

v_1 = tf.constant([1, 2, 3, 4])
v_2 = tf.constant([2, 1, 5, 3])
v_add = tf.add(v_1, v_2)  # You can also write v_1 + v_2 instead

with tf.compat.v1.Session() as sess:
    print(sess.run(v_add))
