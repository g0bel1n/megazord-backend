from tensorflow.keras import layers
import tensorflow  as tf
class InterLayer(layers.Layer):

    def __init__(self, unit):
        super(InterLayer, self).__init__()

    def call(self, inputs):

        ind = tf.math.argmax(inputs)

        l = [0 for _ in range(inputs_dim[0])]
        l[ind] = 1
        return tf.math.