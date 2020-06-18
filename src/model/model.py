import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D as conv_, Activation, Input, Lambda, MaxPooling2D as pool, Concatenate, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal, Constant


def relu(x):
    return Activation('relu')(x)


def Conv2D(x, filters, kernel_size, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = conv_(filters, (kernel_size, kernel_size), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=RandomNormal(stddev=0.01),
               bias_initializer=Constant(0.0))(x)
    return x


def MaxPooling2D(x, kernel_size, stride_size, name):
    x = pool((kernel_size, kernel_size), strides=(stride_size, stride_size), name=name)(x)
    return x


def vgg_block(x, weight_decay):
    # Block 1
    x = Conv2D(x, 64, 3, "conv1_1", (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 64, 3, "conv1_2", (weight_decay, 0))
    x = relu(x)
    x = MaxPooling2D(x, 2, 2, "pool1_1")

    # Block 2
    x = Conv2D(x, 128, 3, "conv2_1", (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 128, 3, "conv2_2", (weight_decay, 0))
    x = relu(x)
    x = MaxPooling2D(x, 2, 2, "pool2_1")

    # Block 3
    x = Conv2D(x, 256, 3, "conv3_1", (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 256, 3, "conv3_2", (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 256, 3, "conv3_3", (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 256, 3, "conv3_4", (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 2, 2, "pool3_1")

    # Block 4
    x = Conv2D(x, 512, 3, "conv4_1", (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 512, 3, "conv4_2", (weight_decay, 0))
    x = relu(x)

    # Additional non vgg layers
    x = Conv2D(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)

    return x


def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = Conv2D(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = Conv2D(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = Conv2D(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

    return x


def apply_mask(x, mask1, mask2, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == 38:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def get_training_model(weight_decay):

    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)
    vec_input_shape = (None, None, 38)
    heat_input_shape = (None, None, 19)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_testing_model():
    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model