import tensorflow as tf


def conv(filters, kernel_size, name, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4),
           bias_regularizer=tf.keras.regularizers.l2(0.0)):

    return tf.keras.layers.Conv2D(filters, kernel_size, activation=activation,
                                  padding='same',
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.01),
                                  bias_initializer='zeros',
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  name=name)


class HCIModel(tf.keras.Model):

    def __init__(self, masked_outputs=True):
        super(HCIModel, self).__init__()

        self.masked_outputs = masked_outputs
        self.images_normalized = tf.keras.layers.Lambda(lambda x: x / 256 - 0.5)

        ####################################################################################################################

        # First 10 layers of VGG19

        # VGG19 - Layer 1
        x = conv(images_normalized, 64, 3, "vgg_conv_1", (weight_decay, 0))
        x = relu(x)

        # VGG19 - Layer 2
        x = conv(x, 64, 3, "vgg_conv_2", (weight_decay, 0))
        x = relu(x)
        x = pooling(x, 2, 2, "vgg_pool_1")

        # VGG19 - Layer 3
        x = conv(x, 128, 3, "vgg_conv_3", (weight_decay, 0))
        x = relu(x)

        # VGG19 - Layer 4
        x = conv(x, 128, 3, "vgg_conv_4", (weight_decay, 0))
        x = relu(x)
        x = pooling(x, 2, 2, "vgg_pool_2")

        # VGG19 - Layer 5
        x = conv(x, 256, 3, "vgg_conv_5", (weight_decay, 0))
        x = relu(x)

        # VGG19 - Layer 6
        x = conv(x, 256, 3, "vgg_conv_6", (weight_decay, 0))
        x = relu(x)

        # VGG19 - Layer 7
        x = conv(x, 256, 3, "vgg_conv_7", (weight_decay, 0))
        x = relu(x)

        # VGG19 - Layer 8
        x = conv(x, 256, 3, "vgg_conv_8", (weight_decay, 0))
        x = relu(x)
        x = pooling(x, 2, 2, "vgg_pool_3")

        # VGG19 - Layer 9
        x = conv(x, 512, 3, "vgg_conv_9", (weight_decay, 0))
        x = relu(x)

        # VGG19 - Layer 10
        x = conv(x, 512, 3, "vgg_conv_10", (weight_decay, 0))
        x = relu(x)
        vgg_out = x

        # Convolutional Pose Machine (CPM) Layers

        # CPM Layer 1
        x = conv(vgg_out, 256, 3, "cpm_1", (weight_decay, 0))
        x = relu(x)

        # CPM Layer 2
        x = conv(x, 128, 3, "cpm_2", (weight_decay, 0))
        x = relu(x)
        cpm_out = x
