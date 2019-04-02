from keras.layers import Conv2D
import tensorflow as tf

class QuantizedConv2D(Conv2D):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, inputs):
        max_range = 100000000
        # compute quantized inputs and kernel
        input_max = tf.reduce_max(inputs)
        kernel_max = tf.reduce_max(self.kernel)
        bias_max = tf.reduce_max(self.bias)
        max_concat = tf.stack([input_max, kernel_max, bias_max])
        both_max = tf.reduce_max(max_concat)
        input_muled = tf.multiply(inputs, max_range / both_max)
        kernel_muled = tf.multiply(self.kernel, max_range / both_max)
        bias_muled = tf.multiply(self.bias, max_range / both_max)
        input_quan = tf.cast(input_muled, tf.int32)
        kernel_quan = tf.cast(kernel_muled, tf.int32)
        bias_quan = tf.cast(bias_muled, tf.int32)
        float_input_quan = tf.cast(input_quan, tf.float32)
        float_kernel_quan = tf.cast(kernel_quan, tf.float32)
        float_bias_quan = tf.cast(bias_quan, tf.float32)
        #set inputs and kernel to quantized values
        kernel_backup = self.kernel
        bias_backup = self.bias
        self.kernel = float_kernel_quan
        self.bias = float_bias_quan
        #run super
        super_output = super().call(float_input_quan)
        #dequantize
        self.kernel = kernel_backup
        self.bias = bias_backup
        conv_dequan_one = tf.multiply(super_output, both_max / max_range)
        conv_dequan = tf.multiply(conv_dequan_one, both_max / max_range)

        return conv_dequan