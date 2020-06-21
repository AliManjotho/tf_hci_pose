import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Multiply, Lambda


def conv(filters, kernel_size, name, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4),
           bias_regularizer=tf.keras.regularizers.l2(0.0)):

    return Conv2D(filters, kernel_size, activation=activation,
                                  padding='same',
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.01),
                                  bias_initializer='zeros',
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  name=name)


class HCIModel(Model):

    def __init__(self, masked_outputs=True):
        super(HCIModel, self).__init__()

        self.masked_outputs = masked_outputs
        self.normalize_images = Lambda(lambda x: x / 256 - 0.5)

        ####################################################################################################################

        # First 10 layers of VGG19

        self.vgg_conv_1  = conv(64, 3, 'vgg_conv_1')
        self.vgg_conv_2  = conv(64, 3, 'vgg_conv_2')
        self.vgg_pool_1  = MaxPooling2D((2, 2), strides=(2, 2), name='vgg_pool_1')
        self.vgg_conv_3  = conv(128, 3, 'vgg_conv_3')
        self.vgg_conv_4  = conv(128, 3, 'vgg_conv_4')
        self.vgg_pool_2  = MaxPooling2D((2, 2), strides=(2, 2), name='vgg_pool_2')
        self.vgg_conv_5  = conv(256, 3, 'vgg_conv_5')
        self.vgg_conv_6  = conv(256, 3, 'vgg_conv_6')
        self.vgg_conv_7  = conv(256, 3, 'vgg_conv_7')
        self.vgg_conv_8  = conv(256, 3, 'vgg_conv_8')
        self.vgg_pool_3  = MaxPooling2D((2, 2), strides=(2, 2), name='vgg_pool_3')
        self.vgg_conv_9  = conv(512, 3, 'vgg_conv_9')
        self.vgg_conv_10 = conv(512, 3, 'vgg_conv_10')


        # Convolutional Pose Machine (CPM) Layers

        self.cpm_conv_1 = conv(256, 3, 'cpm_conv_1')
        self.cpm_conv_2 = conv(128, 3, 'cpm_conv_2')


        # HCIPose Stage 1

        self.stage_1_branch_1_conv_1 = conv(128, 3, "stage_1_branch_1_conv_1")
        self.stage_1_branch_1_conv_2 = conv(128, 3, "stage_1_branch_1_conv_2")
        self.stage_1_branch_1_conv_3 = conv(128, 3, "stage_1_branch_1_conv_3")
        self.stage_1_branch_1_conv_4 = conv(512, 1, "stage_1_branch_1_conv_4")
        self.stage_1_branch_1_conv_5 = conv(19,  1, "stage_1_branch_1_conv_5", activation=None)

        self.stage_1_branch_2_conv_1 = conv(128, 3, "stage_1_branch_2_conv_1")
        self.stage_1_branch_2_conv_2 = conv(128, 3, "stage_1_branch_2_conv_2")
        self.stage_1_branch_2_conv_3 = conv(128, 3, "stage_1_branch_2_conv_3")
        self.stage_1_branch_2_conv_4 = conv(512, 1, "stage_1_branch_2_conv_4")
        self.stage_1_branch_2_conv_5 = conv(38,  1, "stage_1_branch_2_conv_5", activation=None)

        self.stage_1_concat = Concatenate(axis=3)


        # HCIPose Stage 2

        self.stage_2_branch_1_conv_1 = conv(128, 7, "stage_2_branch_1_conv_1")
        self.stage_2_branch_1_conv_2 = conv(128, 7, "stage_2_branch_1_conv_2")
        self.stage_2_branch_1_conv_3 = conv(128, 7, "stage_2_branch_1_conv_3")
        self.stage_2_branch_1_conv_4 = conv(128, 7, "stage_2_branch_1_conv_4")
        self.stage_2_branch_1_conv_5 = conv(128, 7, "stage_2_branch_1_conv_5")
        self.stage_2_branch_1_conv_6 = conv(128, 1, "stage_2_branch_1_conv_6")
        self.stage_2_branch_1_conv_7 = conv(19,  1, "stage_2_branch_1_conv_7", activation=None)

        self.stage_2_branch_2_conv_1 = conv(128, 7, "stage_2_branch_2_conv_1")
        self.stage_2_branch_2_conv_2 = conv(128, 7, "stage_2_branch_2_conv_2")
        self.stage_2_branch_2_conv_3 = conv(128, 7, "stage_2_branch_2_conv_3")
        self.stage_2_branch_2_conv_4 = conv(128, 7, "stage_2_branch_2_conv_4")
        self.stage_2_branch_2_conv_5 = conv(128, 7, "stage_2_branch_2_conv_5")
        self.stage_2_branch_2_conv_6 = conv(128, 1, "stage_2_branch_2_conv_6")
        self.stage_2_branch_2_conv_7 = conv(38,  1, "stage_2_branch_2_conv_7", activation=None)

        self.stage_2_concat = Concatenate(axis=3)


        # HCIPose Stage 3

        self.stage_3_branch_1_conv_1 = conv(128, 7, "stage_3_branch_1_conv_1")
        self.stage_3_branch_1_conv_2 = conv(128, 7, "stage_3_branch_1_conv_2")
        self.stage_3_branch_1_conv_3 = conv(128, 7, "stage_3_branch_1_conv_3")
        self.stage_3_branch_1_conv_4 = conv(128, 7, "stage_3_branch_1_conv_4")
        self.stage_3_branch_1_conv_5 = conv(128, 7, "stage_3_branch_1_conv_5")
        self.stage_3_branch_1_conv_6 = conv(128, 1, "stage_3_branch_1_conv_6")
        self.stage_3_branch_1_conv_7 = conv(19 , 1, "stage_3_branch_1_conv_7", activation=None)

        self.stage_3_branch_2_conv_1 = conv(128, 7, "stage_3_branch_2_conv_1")
        self.stage_3_branch_2_conv_2 = conv(128, 7, "stage_3_branch_2_conv_2")
        self.stage_3_branch_2_conv_3 = conv(128, 7, "stage_3_branch_2_conv_3")
        self.stage_3_branch_2_conv_4 = conv(128, 7, "stage_3_branch_2_conv_4")
        self.stage_3_branch_2_conv_5 = conv(128, 7, "stage_3_branch_2_conv_5")
        self.stage_3_branch_2_conv_6 = conv(128, 1, "stage_3_branch_2_conv_6")
        self.stage_3_branch_2_conv_7 = conv(38 , 1, "stage_3_branch_2_conv_7", activation=None)

        self.stage_3_concat = Concatenate(axis=3)


        # HCIPose Stage 4

        self.stage_4_branch_1_conv_1 = conv(128, 7, "stage_4_branch_1_conv_1")
        self.stage_4_branch_1_conv_2 = conv(128, 7, "stage_4_branch_1_conv_2")
        self.stage_4_branch_1_conv_3 = conv(128, 7, "stage_4_branch_1_conv_3")
        self.stage_4_branch_1_conv_4 = conv(128, 7, "stage_4_branch_1_conv_4")
        self.stage_4_branch_1_conv_5 = conv(128, 7, "stage_4_branch_1_conv_5")
        self.stage_4_branch_1_conv_6 = conv(128, 1, "stage_4_branch_1_conv_6")
        self.stage_4_branch_1_conv_7 = conv(19 , 1, "stage_4_branch_1_conv_7", activation=None)

        self.stage_4_branch_2_conv_1 = conv(128, 7, "stage_4_branch_2_conv_1")
        self.stage_4_branch_2_conv_2 = conv(128, 7, "stage_4_branch_2_conv_2")
        self.stage_4_branch_2_conv_3 = conv(128, 7, "stage_4_branch_2_conv_3")
        self.stage_4_branch_2_conv_4 = conv(128, 7, "stage_4_branch_2_conv_4")
        self.stage_4_branch_2_conv_5 = conv(128, 7, "stage_4_branch_2_conv_5")
        self.stage_4_branch_2_conv_6 = conv(128, 1, "stage_4_branch_2_conv_6")
        self.stage_4_branch_2_conv_7 = conv(38 , 1, "stage_4_branch_2_conv_7", activation=None)

        self.stage_4_concat = Concatenate(axis=3)


        # HCIPose Stage 5

        self.stage_5_branch_1_conv_1 = conv(128, 7, "stage_5_branch_1_conv_1")
        self.stage_5_branch_1_conv_2 = conv(128, 7, "stage_5_branch_1_conv_2")
        self.stage_5_branch_1_conv_3 = conv(128, 7, "stage_5_branch_1_conv_3")
        self.stage_5_branch_1_conv_4 = conv(128, 7, "stage_5_branch_1_conv_4")
        self.stage_5_branch_1_conv_5 = conv(128, 7, "stage_5_branch_1_conv_5")
        self.stage_5_branch_1_conv_6 = conv(128, 1, "stage_5_branch_1_conv_6")
        self.stage_5_branch_1_conv_7 = conv(19 , 1, "stage_5_branch_1_conv_7", activation=None)

        self.stage_5_branch_2_conv_1 = conv(128, 7, "stage_5_branch_2_conv_1")
        self.stage_5_branch_2_conv_2 = conv(128, 7, "stage_5_branch_2_conv_2")
        self.stage_5_branch_2_conv_3 = conv(128, 7, "stage_5_branch_2_conv_3")
        self.stage_5_branch_2_conv_4 = conv(128, 7, "stage_5_branch_2_conv_4")
        self.stage_5_branch_2_conv_5 = conv(128, 7, "stage_5_branch_2_conv_5")
        self.stage_5_branch_2_conv_6 = conv(128, 1, "stage_5_branch_2_conv_6")
        self.stage_5_branch_2_conv_7 = conv(38 , 1, "stage_5_branch_2_conv_7", activation=None)

        self.stage_5_concat = Concatenate(axis=3)


        # HCIPose Stage 6

        self.stage_6_branch_1_conv_1 = conv(128, 7, "stage_6_branch_1_conv_1")
        self.stage_6_branch_1_conv_2 = conv(128, 7, "stage_6_branch_1_conv_2")
        self.stage_6_branch_1_conv_3 = conv(128, 7, "stage_6_branch_1_conv_3")
        self.stage_6_branch_1_conv_4 = conv(128, 7, "stage_6_branch_1_conv_4")
        self.stage_6_branch_1_conv_5 = conv(128, 7, "stage_6_branch_1_conv_5")
        self.stage_6_branch_1_conv_6 = conv(128, 1, "stage_6_branch_1_conv_6")
        self.stage_6_branch_1_conv_7 = conv(19 , 1, "stage_6_branch_1_conv_7", activation=None)

        self.stage_6_branch_2_conv_1 = conv(128, 7, "stage_6_branch_2_conv_1")
        self.stage_6_branch_2_conv_2 = conv(128, 7, "stage_6_branch_2_conv_2")
        self.stage_6_branch_2_conv_3 = conv(128, 7, "stage_6_branch_2_conv_3")
        self.stage_6_branch_2_conv_4 = conv(128, 7, "stage_6_branch_2_conv_4")
        self.stage_6_branch_2_conv_5 = conv(128, 7, "stage_6_branch_2_conv_5")
        self.stage_6_branch_2_conv_6 = conv(128, 1, "stage_6_branch_2_conv_6")
        self.stage_6_branch_2_conv_7 = conv(38 , 1, "stage_6_branch_2_conv_7", activation=None)


        if self.masked_outputs:
            self.masked_1_branch_1 = Multiply()
            self.masked_2_branch_1 = Multiply()
            self.masked_3_branch_1 = Multiply()
            self.masked_4_branch_1 = Multiply()
            self.masked_5_branch_1 = Multiply()
            self.masked_6_branch_1 = Multiply()
            self.masked_1_branch_2 = Multiply()
            self.masked_2_branch_2 = Multiply()
            self.masked_3_branch_2 = Multiply()
            self.masked_4_branch_2 = Multiply()
            self.masked_5_branch_2 = Multiply()
            self.masked_6_branch_2 = Multiply()


    def call(self, inputs):

        inputs_with_masks = isinstance(inputs, (list, tuple)) and len(inputs) == 3

        images = inputs[0] if inputs_with_masks else inputs

        x = self.normalize_images(images)


        # First 10 layers of VGG19

        x = self.vgg_conv_1(x)
        x = self.vgg_conv_2(x)
        x = self.vgg_pool_1(x)
        x = self.vgg_conv_3(x)
        x = self.vgg_conv_4(x)
        x = self.vgg_pool_2(x)
        x = self.vgg_conv_5(x)
        x = self.vgg_conv_6(x)
        x = self.vgg_conv_7(x)
        x = self.vgg_conv_8(x)
        x = self.vgg_pool_3(x)
        x = self.vgg_conv_9(x)
        vgg_out = self.vgg_conv_10(x)


        # Convolutional Pose Machine (CPM) Layers

        x = self.cpm_conv_1(vgg_out)
        cpm_out = self.cpm_conv_1(x)


        # HCIPose Stage 1

        x = self.stage_1_branch_1_conv_1(cpm_out)
        x = self.stage_1_branch_1_conv_2(x)
        x = self.stage_1_branch_1_conv_3(x)
        x = self.stage_1_branch_1_conv_4(x)
        stage_1_branch_1_out = self.stage_1_branch_1_conv_5(x)
        x = self.stage_1_branch_2_conv_1(cpm_out)
        x = self.stage_1_branch_2_conv_2(x)
        x = self.stage_1_branch_2_conv_3(x)
        x = self.stage_1_branch_2_conv_4(x)
        stage_1_branch_2_out = self.stage_1_branch_2_conv_5(x)
        stage_1_out = self.stage_1_concat([stage_1_branch_1_out, stage_1_branch_2_out, cpm_out])

        if self.masked_outputs and inputs_with_masks:
            output_1_branch_1 = self.masked_1_branch_1([stage_1_branch_1_out, inputs[1]])
            output_1_branch_2 = self.masked_1_branch_2([stage_1_branch_2_out, inputs[2]])


        # HCIPose Stage 2

        x = self.stage_2_branch_1_conv_1(stage_1_out)
        x = self.stage_2_branch_1_conv_2(x)
        x = self.stage_2_branch_1_conv_3(x)
        x = self.stage_2_branch_1_conv_4(x)
        x = self.stage_2_branch_1_conv_5(x)
        x = self.stage_2_branch_1_conv_6(x)
        stage_2_branch_1_out = self.stage_2_branch_1_conv_7(x)
        x = self.stage_2_branch_2_conv_1(stage_1_out)
        x = self.stage_2_branch_2_conv_2(x)
        x = self.stage_2_branch_2_conv_3(x)
        x = self.stage_2_branch_2_conv_4(x)
        x = self.stage_2_branch_2_conv_5(x)
        x = self.stage_2_branch_2_conv_6(x)
        stage_2_branch_2_out = self.stage_2_branch_2_conv_7(x)
        stage_2_out = self.stage_2_concat([stage_2_branch_1_out, stage_2_branch_2_out, cpm_out])

        if self.masked_outputs and inputs_with_masks:
            output_2_branch_1 = self.masked_2_branch_1([stage_2_branch_1_out, inputs[1]])
            output_2_branch_2 = self.masked_2_branch_2([stage_2_branch_2_out, inputs[2]])


        # HCIPose Stage 3

        x = self.stage_3_branch_1_conv_1(stage_2_out)
        x = self.stage_3_branch_1_conv_2(x)
        x = self.stage_3_branch_1_conv_3(x)
        x = self.stage_3_branch_1_conv_4(x)
        x = self.stage_3_branch_1_conv_5(x)
        x = self.stage_3_branch_1_conv_6(x)
        stage_3_branch_1_out = self.stage_3_branch_1_conv_7(x)
        x = self.stage_3_branch_2_conv_1(stage_2_out)
        x = self.stage_3_branch_2_conv_2(x)
        x = self.stage_3_branch_2_conv_3(x)
        x = self.stage_3_branch_2_conv_4(x)
        x = self.stage_3_branch_2_conv_5(x)
        x = self.stage_3_branch_2_conv_6(x)
        stage_3_branch_2_out = self.stage_3_branch_2_conv_7(x)
        stage_3_out = self.stage_3_concat([stage_3_branch_1_out, stage_3_branch_2_out, cpm_out])

        if self.masked_outputs and inputs_with_masks:
            output_3_branch_1 = self.masked_3_branch_1([stage_3_branch_1_out, inputs[1]])
            output_3_branch_2 = self.masked_3_branch_2([stage_3_branch_2_out, inputs[2]])


        # HCIPose Stage 4

        x = self.stage_4_branch_1_conv_1(stage_3_out)
        x = self.stage_4_branch_1_conv_2(x)
        x = self.stage_4_branch_1_conv_3(x)
        x = self.stage_4_branch_1_conv_4(x)
        x = self.stage_4_branch_1_conv_5(x)
        x = self.stage_4_branch_1_conv_6(x)
        stage_4_branch_1_out = self.stage_4_branch_1_conv_7(x)
        x = self.stage_4_branch_2_conv_1(stage_3_out)
        x = self.stage_4_branch_2_conv_2(x)
        x = self.stage_4_branch_2_conv_3(x)
        x = self.stage_4_branch_2_conv_4(x)
        x = self.stage_4_branch_2_conv_5(x)
        x = self.stage_4_branch_2_conv_6(x)
        stage_4_branch_2_out = self.stage_4_branch_2_conv_7(x)
        stage_4_out = self.stage_4_concat([stage_4_branch_1_out, stage_4_branch_2_out, cpm_out])

        if self.masked_outputs and inputs_with_masks:
            output_4_branch_1 = self.masked_4_branch_1([stage_4_branch_1_out, inputs[1]])
            output_4_branch_2 = self.masked_4_branch_2([stage_4_branch_2_out, inputs[2]])


        # HCIPose Stage 5

        x = self.stage_5_branch_1_conv_1(stage_4_out)
        x = self.stage_5_branch_1_conv_2(x)
        x = self.stage_5_branch_1_conv_3(x)
        x = self.stage_5_branch_1_conv_4(x)
        x = self.stage_5_branch_1_conv_5(x)
        x = self.stage_5_branch_1_conv_6(x)
        stage_5_branch_1_out = self.stage_5_branch_1_conv_7(x)
        x = self.stage_5_branch_2_conv_1(stage_4_out)
        x = self.stage_5_branch_2_conv_2(x)
        x = self.stage_5_branch_2_conv_3(x)
        x = self.stage_5_branch_2_conv_4(x)
        x = self.stage_5_branch_2_conv_5(x)
        x = self.stage_5_branch_2_conv_6(x)
        stage_5_branch_2_out = self.stage_5_branch_2_conv_7(x)
        stage_5_out = self.stage_5_concat([stage_5_branch_1_out, stage_5_branch_2_out, cpm_out])

        if self.masked_outputs and inputs_with_masks:
            output_5_branch_1 = self.masked_5_branch_1([stage_5_branch_1_out, inputs[1]])
            output_5_branch_2 = self.masked_5_branch_2([stage_5_branch_2_out, inputs[2]])


        # HCIPose Stage 6

        x = self.stage_6_branch_1_conv_1(stage_5_out)
        x = self.stage_6_branch_1_conv_2(x)
        x = self.stage_6_branch_1_conv_3(x)
        x = self.stage_6_branch_1_conv_4(x)
        x = self.stage_6_branch_1_conv_5(x)
        x = self.stage_6_branch_1_conv_6(x)
        stage_6_branch_1_out = self.stage_6_branch_1_conv_7(x)
        x = self.stage_6_branch_2_conv_1(stage_5_out)
        x = self.stage_6_branch_2_conv_2(x)
        x = self.stage_6_branch_2_conv_3(x)
        x = self.stage_6_branch_2_conv_4(x)
        x = self.stage_6_branch_2_conv_5(x)
        x = self.stage_6_branch_2_conv_6(x)
        stage_6_branch_2_out = self.stage_6_branch_2_conv_7(x)

        if self.masked_outputs and inputs_with_masks:
            output_6_branch_1 = self.masked_6_branch_1([stage_6_branch_1_out, inputs[1]])
            output_6_branch_2 = self.masked_6_branch_2([stage_6_branch_2_out, inputs[2]])

            return output_1_branch_1, output_1_branch_2, \
                   output_2_branch_1, output_2_branch_2, \
                   output_3_branch_1, output_3_branch_2, \
                   output_4_branch_1, output_4_branch_2, \
                   output_5_branch_1, output_5_branch_2, \
                   output_6_branch_1, output_6_branch_2
        else:
            return stage_1_branch_1_out, stage_1_branch_2_out, \
                   stage_2_branch_1_out, stage_2_branch_2_out, \
                   stage_3_branch_1_out, stage_3_branch_2_out, \
                   stage_4_branch_1_out, stage_4_branch_2_out, \
                   stage_5_branch_1_out, stage_5_branch_2_out, \
                   stage_6_branch_1_out, stage_6_branch_2_out


