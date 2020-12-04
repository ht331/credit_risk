import tensorflow as tf

# Start an Interactive Session
sess = tf.InteractiveSession()

# Define a 5x5 Identity matrix
I_matrix = tf.eye(5)
print(I_matrix.eval())
# This will print a 5x5 Identity matrix

# Define a Variable initialized to 10x10 identity matrix
X = tf.Variable(tf.eye(10))
X.initializer.run()  # Initialize the Variable
print(X.eval())
# Evaluate the Variable and print the result

# Create a random 5x10 matrix
A = tf.Variable(tf.random_normal([5, 10]))
A.initializer.run()

# Multiply two martices
product = tf.matmul(A, X)
print(product.eval())
# create a random matrix of 1s and 0s, size 5x10
b = tf.Variable(tf.random_uniform([5, 10], 0, 2, dtype=tf.int32))
b.initializer.run()
print(b.eval())
b_new = tf.cast(b, dtype=tf.float32)
# Cast to float32 data type

# Add the two matrices
t_sum = tf.add(product, b_new)
t_sub = product - b_new
print('A * X + b \n', t_sum.eval())
print('A * X - b \n', t_sub.eval())
