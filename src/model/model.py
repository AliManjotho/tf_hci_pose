from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Activation, Input, Lambda, MaxPooling2D, Concatenate, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal, Constant

NUM_CONFIDENCE_MAPS = 19
NUM_PAFS = 38

def relu(x):
    return Activation('relu')(x)

def conv(x, filters, kernel_size, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(filters=filters,
               kernel_size = (kernel_size, kernel_size),
               padding='same',
               name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=RandomNormal(stddev=0.01),
               bias_initializer=Constant(0.0))(x)

    return x


def pooling(x, kernel_size, stride_size, name):
    return MaxPooling2D((kernel_size, kernel_size), strides=(stride_size, stride_size), name=name)(x)


def apply_mask(x, mask1, mask2, stage, branch):
    name = "weight_stage_{0}_branch_{1}".format(stage, branch)
    if branch == 1:
        weighted = Multiply(name=name)([x, mask1]) # Hmap_Weight
    elif branch == 2:
        weighted = Multiply(name=name)([x, mask2])  # PAF_Weight
    return weighted



def getTrainModel(weight_decay):

    inputs = []
    outputs = []

    input_images = Input(shape=(None, None, 3))
    input_hmaps = Input(shape=(None, None, 19))
    input_pafs = Input(shape=(None, None, 38))

    inputs.append(input_images)
    inputs.append(input_hmaps)
    inputs.append(input_pafs)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(input_images)

    ####################################################################################################################

    # First 10 layers of VGG19

    # VGG19 - Layer 1
    x = conv(img_normalized, 64, 3, "vgg_conv_1", (weight_decay, 0))
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


    ####################################################################################################################


    # HCIPose Stages

    # Stage 1 - Branch 1
    stage = 1
    branch = 1
    x = conv(cpm_out, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 1), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 2), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 3), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 4), (weight_decay, 0))
    x = relu(x)
    x = conv(x, NUM_CONFIDENCE_MAPS, 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 5), (weight_decay, 0))
    x = relu(x)
    stage_1_branch_1_out = x


    # Stage 1 - Branch 2
    stage = 1
    branch = 2
    x = conv(cpm_out, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 1), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 2), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 3), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 4), (weight_decay, 0))
    x = relu(x)
    x = conv(x, NUM_PAFS, 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 5), (weight_decay, 0))
    x = relu(x)
    stage_1_branch_2_out = x


    w1 = apply_mask(stage_1_branch_1_out, input_hmaps, input_pafs, NUM_CONFIDENCE_MAPS, stage, branch)
    w2 = apply_mask(stage_1_branch_2_out, input_hmaps, input_pafs, NUM_PAFS, stage, branch)

    x = Concatenate()([stage_1_branch_1_out, stage_1_branch_2_out, cpm_out])

    outputs.append(w1)
    outputs.append(w2)


    ####################################################################################################################


    # Stage 2 to 6 - Branches 1 and 2
    stages = [2, 3, 4, 5, 6]
    branches = [1, 2]
    num_outputs = [NUM_CONFIDENCE_MAPS, NUM_PAFS]

    for s, stage in stages:
        stage_n_branch_1_out = 0
        stage_n_branch_2_out = 0
        w1 = 0
        w2 = 0
        for b, branch in branches:

            x = conv(x, 128, 7, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 1), (weight_decay, 0))
            x = relu(x)
            x = conv(x, 128, 7, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 2), (weight_decay, 0))
            x = relu(x)
            x = conv(x, 128, 7, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 3), (weight_decay, 0))
            x = relu(x)
            x = conv(x, 128, 7, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 4), (weight_decay, 0))
            x = relu(x)
            x = conv(x, 128, 7, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 5), (weight_decay, 0))
            x = relu(x)
            x = conv(x, 128, 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 6), (weight_decay, 0))
            x = relu(x)
            x = conv(x, num_outputs[b], 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 7), (weight_decay, 0))
            x = relu(x)

            if branch == 1:
                stage_n_branch_1_out = x
                w1 = apply_mask(stage_n_branch_1_out, input_hmaps, input_pafs, NUM_CONFIDENCE_MAPS, stage, branch)
            elif branch == 2:
                stage_n_branch_2_out = x
                w2 = apply_mask(stage_n_branch_2_out, input_hmaps, input_pafs, NUM_PAFS, stage, branch)

        outputs.append(w1)
        outputs.append(w2)

        if (stage < 6):
            x = Concatenate()([stage_n_branch_1_out, stage_n_branch_2_out, cpm_out])

    model = Model(inputs=inputs, outputs=outputs)

    return model



def getTestModel(weight_decay):

    input_images = Input(shape=(None, None, 3))
    img_normalized = Lambda(lambda x: x / 256 - 0.5)(input_images)  # [-0.5, 0.5]

    ####################################################################################################################

    # First 10 layers of VGG19

    # VGG19 - Layer 1
    x = conv(img_normalized, 64, 3, "vgg_conv_1", (weight_decay, 0))
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

    ####################################################################################################################

    # HCIPose Stages

    # Stage 1 - Branch 1
    stage = 1
    branch = 1
    x = conv(cpm_out, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 1), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 2), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 3), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 4), (weight_decay, 0))
    x = relu(x)
    x = conv(x, NUM_CONFIDENCE_MAPS, 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 5), (weight_decay, 0))
    x = relu(x)
    stage_1_branch_1_out = x

    # Stage 1 - Branch 2
    stage = 1
    branch = 2
    x = conv(cpm_out, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 1), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 2), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 3), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 4), (weight_decay, 0))
    x = relu(x)
    x = conv(x, NUM_PAFS, 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 5), (weight_decay, 0))
    x = relu(x)
    stage_1_branch_2_out = x

    x = Concatenate()([stage_1_branch_1_out, stage_1_branch_2_out, cpm_out])


    ####################################################################################################################

    # Stage 2 to 6 - Branches 1 and 2
    stages = [2, 3, 4, 5, 6]
    branches = [1, 2]
    num_outputs = [NUM_CONFIDENCE_MAPS, NUM_PAFS]

    for s, stage in stages:
        stage_n_branch_1_out = 0
        stage_n_branch_2_out = 0
        w1 = 0
        w2 = 0
        for b, branch in branches:

            x = conv(x, 128, 7, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 1), (weight_decay, 0))
            x = relu(x)
            x = conv(x, 128, 7, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 2), (weight_decay, 0))
            x = relu(x)
            x = conv(x, 128, 7, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 3), (weight_decay, 0))
            x = relu(x)
            x = conv(x, 128, 7, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 4), (weight_decay, 0))
            x = relu(x)
            x = conv(x, 128, 7, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 5), (weight_decay, 0))
            x = relu(x)
            x = conv(x, 128, 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 6), (weight_decay, 0))
            x = relu(x)
            x = conv(x, num_outputs[b], 1, "stage_{0}_branch_{1}_conv_{2}".format(stage, branch, 7), (weight_decay, 0))
            x = relu(x)

            if branch == 1:
                stage_n_branch_1_out = x
            elif branch == 2:
                stage_n_branch_2_out = x

        if (stage < 6):
            x = Concatenate()([stage_n_branch_1_out, stage_n_branch_2_out, cpm_out])

        model = Model(inputs=[input_images], outputs=[stage_n_branch_1_out, stage_n_branch_2_out])

        return model
