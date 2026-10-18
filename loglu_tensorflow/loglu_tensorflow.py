import tensorflow as tf

class LogLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LogLU, self).__init__(**kwargs)

    def call(self, inputs):
        return self.loglu_activation(inputs)

    @tf.custom_gradient
    def loglu_activation(self, inputs):
        epsilon = 1e-8
        inputs_clipped = tf.clip_by_value(
            inputs,
            tf.cast(float('-inf'), inputs.dtype),
            tf.cast(1.0 - epsilon, inputs.dtype)
        )

        outputs = tf.where(
            inputs > 0,
            inputs,
            -tf.math.log(1.0 - inputs_clipped)
        )

        def grad(dy):
            grad_inputs = tf.where(
                inputs > 0,
                tf.ones_like(inputs),
                1.0 / (1.0 - inputs_clipped)
            )
            return dy * grad_inputs

        return outputs, grad

    def get_config(self):
        config = super(LogLU, self).get_config()
        return config