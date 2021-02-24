import tensorflow as tf

tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x
tf.compat.v1.reset_default_graph()

# Build a graph
var1 = tf.constant(1, name="a")
var2 = tf.constant(2, name="b")
var3 = tf.constant(3, name="c")

mulv = tf.multiply(var1, var2)
add = tf.add(mulv,var3)

with tf.compat.v1.Session() as sess: # Launch the graph in a session.
    print(sess.run(add) # Evaluate
