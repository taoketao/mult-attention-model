# 10/22/16   Morgan Bryant
#
# See comments at top of ./_mram.py
# One note though: a comprehensive port of matplotlib, pyplot, etc are being 
# done in order to permit image saving instead of showing.

import tensorflow as tf
import tf_mnist_loader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import sys
import os

print "Matplotlib version:", matplotlib.__version__

try:
    xrange
except NameError:
    xrange = range

dataset = tf_mnist_loader.read_data_sets("mnist_data")
save_dir = "chckPts/"
save_prefix = "save"
summaryFolderName = "summary/"
outfilecount = 0 # MORGAN: for differentiating output images

if len(sys.argv) == 2:
    simulationName = str(sys.argv[1])
    print "Simulation name = " + simulationName
    summaryFolderName = summaryFolderName + simulationName + "/"
    saveImgs = True
    imgsFolderName = "imgs/" + simulationName + "/"
    if os.path.isdir(summaryFolderName) == False:
        os.mkdir(summaryFolderName)
    # if os.path.isdir(imgsFolderName) == False:
    #     os.mkdir(imgsFolderName)
else:
    saveImgs = False
    print "Testing... image files will not be saved."


start_step = 0
#load_path = None
load_path = save_dir + save_prefix + str(start_step) + ".ckpt"
# to enable visualization, set draw to True
eval_only = False
no_display_env = True # added by MORGAN: for a $DISPLAY-less environment such as afs.
draw = 1
animate = 0  # 10-22-16 MORGAN: this ON and no_display_env is todo.

# condition
translateMnist = 1 # MORGAN: What does this do? update: looks like it randomizes the locations 
 # of initial glimpses and therefore decreases permitted # of glimpses & bandwidth from 6->4
nAttn = 3; # MORGAN: this was added. A postitive integer.
eyeCentered = 0

preTraining = 1
preTraining_epoch = 21# default: 20000
drawReconsturction = 0 # default 1.  10-22-16 MORGAN: this ON and no_display_env is todo.

# about translation
MNIST_SIZE = 28
translated_img_size = 60             # side length of the picture

if translateMnist: # MORGAN: I think this is the true situation
    print "TRANSLATED MNIST"
    if nAttn == 1:
        img_size = translated_img_size
        depth = 3  # number of zooms
        sensorBandwidth = 12
        minRadius = 6  # zooms -> minRadius * 2**<depth_level>

        initLr = 5e-3
        lrDecayRate = .995
        lrDecayFreq = 500
        momentumValue = .9
        batch_size = 20
    else: # other limited parameters. For now they are identical.
        img_size = translated_img_size
        depth = 3  # number of zooms
        sensorBandwidth = 12
        minRadius = 6  # zooms -> minRadius * 2**<depth_level>

        initLr = 5e-3
        lrDecayRate = .995
        lrDecayFreq = 500
        momentumValue = .9
        batch_size = 20
        
else:
    print "CENTERED MNIST"
    img_size = MNIST_SIZE
    depth = 1  # number of zooms
    sensorBandwidth = 8
    minRadius = 4  # zooms -> minRadius * 2**<depth_level>

    initLr = 5e-3
    lrDecayRate = .99
    lrDecayFreq = 200
    momentumValue = .9
    batch_size = 20


# model parameters
channels = 1                # mnist are grayscale images
totalSensorBandwidth = depth * channels * (sensorBandwidth **2)
nGlimpses = 6               # number of glimpses
loc_sd = 0.11               # std when setting the location

# network units
hg_size = 128               #
hl_size = 127               #
g_size = 256                #
cell_size = 255             #
cell_out_size = cell_size   #

# paramters about the training examples
n_classes = 10              # card(Y)

# training parameters
max_iters = 4 # default: 1000000
SMALL_NUM = 1e-10

# resource prellocation
mean_locs = []              # expectation of locations
sampled_locs = []           # sampled locations ~N(mean_locs[.], loc_sd)
mean_locs_stopGrad = []
sampled_locs_stopGrad = []


baselines = []              # baseline, the value prediction
glimpse_images = []         # to show in window


# set the weights to be small random values, with truncated normal distribution
def weight_variable(shape, myname, train):
    initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1)
    return tf.Variable(initial, name=myname, trainable=train)

# get local glimpses
def glimpseSensor(img, normLoc):
    loc = tf.round(((normLoc + 1) / 2.0) * img_size)  # normLoc coordinates are between -1 and 1
    loc = tf.cast(loc, tf.int32)

    img = tf.reshape(img, (batch_size, img_size, img_size, channels))

    # process each image individually
    zooms = []
    for k in xrange(batch_size):
        imgZooms = []
        one_img = img[k,:,:,:]
        max_radius = minRadius * (2 ** (depth - 1))
        offset = 2 * max_radius

        # pad image with zeros
        one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, \
                                               max_radius * 4 + img_size, max_radius * 4 + img_size)

        for i in xrange(depth):
            r = int(minRadius * (2 ** (i)))

            d_raw = 2 * r
            d = tf.constant(d_raw, shape=[1])
            d = tf.tile(d, [2])
            loc_k = loc[k,:]
            adjusted_loc = offset + loc_k - r
            one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value, one_img.get_shape()[1].value))

            # crop image to (d x d)
            zoom = tf.slice(one_img2, adjusted_loc, d)

            # resize cropped image to (sensorBandwidth x sensorBandwidth)
            zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)), (sensorBandwidth, sensorBandwidth))
            zoom = tf.reshape(zoom, (sensorBandwidth, sensorBandwidth))
            imgZooms.append(zoom)

        zooms.append(tf.pack(imgZooms))

    zooms = tf.pack(zooms)

    glimpse_images.append(zooms)

    return zooms

def repeatMe(M, rank, pos=-1):
    if pos<0: 
        pos = rank-1
    if rank==2:
        if pos==1:
            return tf.tile(M, tf.pack([1,tf.shape(M)[1]]))
        elif pos==0:
            return tf.tile(M, tf.pack([tf.shape(M)[0],1]))
    elif rank==3:
        if pos==2:
            return tf.tile(M, tf.pack([1,1,tf.shape(M)[2]]))
        elif pos==1:
            return tf.tile(M, tf.pack([1,tf.shape(M)[1],1]))
        elif pos==0:
            return tf.tile(M, tf.pack([tf.shape(M)[0],1,1]))

def get_glimpses(locs, nAttn): # TODO MORGAN: resume work here
    features = []
    for i in xrange(nAttn):
        glimpse_input = glimpseSensor(inputs_placeholder, locs[:,:,i]) # MORGAN: slicing cause dims down
        glimpse_input = tf.reshape(glimpse_input, (batch_size, totalSensorBandwidth))
        
        # the hidden units that process location & the input
        act_glimpse_hidden = tf.nn.relu(tf.matmul(glimpse_input, Wg_g_h) + Bg_g_h)
        act_loc_hidden = tf.nn.relu(tf.matmul(locs[:,:,i], Wg_l_h) + Bg_l_h)

        # the hidden units that integrates the location & the glimpses
        glimpseFeature1 = tf.nn.relu(tf.matmul(act_glimpse_hidden, Wg_hg_gf1) + tf.matmul(act_loc_hidden, Wg_hl_gf1) + Bg_hlhg_gf1)
        # return g
        # glimpseFeature2 = tf.matmul(glimpseFeature1, Wg_gf1_gf2) + Bg_gf1_gf2
        features.append(glimpseFeature1)
    # return features ...for now
    return tf.concat(1,features)  # concat to shape: (batches, imsize * nAttn)


# implements the input network
def get_glimpse(loc):
    # get input using the previous location
    glimpse_input = glimpseSensor(inputs_placeholder, loc)
    glimpse_input = tf.reshape(glimpse_input, (batch_size, totalSensorBandwidth))

    # the hidden units that process location & the input
    act_glimpse_hidden = tf.nn.relu(tf.matmul(glimpse_input, Wg_g_h) + Bg_g_h)
    act_loc_hidden = tf.nn.relu(tf.matmul(loc, Wg_l_h) + Bg_l_h)

    # the hidden units that integrates the location & the glimpses
    glimpseFeature1 = tf.nn.relu(tf.matmul(act_glimpse_hidden, Wg_hg_gf1) + tf.matmul(act_loc_hidden, Wg_hl_gf1) + Bg_hlhg_gf1)
    # return g
    # glimpseFeature2 = tf.matmul(glimpseFeature1, Wg_gf1_gf2) + Bg_gf1_gf2
    return glimpseFeature1



# ------------------------------------------ MORGAN temp, for utility right now 10/22/16

def get_next_input(output): # MORGAN: this is unaffected by nAttr, so no changes made.
    # the next location is computed by the location network
    # MORGAN lines:
    print "tag [03]", output.get_shape(), Wb_h_b.get_shape(), Bb_h_b.get_shape()
    baseline = tf.sigmoid(tf.matmul(output,Wb_h_b) + Bb_h_b)
    baselines.append(baseline)
    # compute the next location, then impose noise
    print "output, Wl_h_l shapes:", output.get_shape(), Wl_h_l.get_shape()
    if eyeCentered:
        # add the last sampled glimpse location
        # TODO max(-1, min(1, u + N(output, sigma) + prevLoc))
        mean_loc = tf.maximum(-1.0, tf.minimum(1.0, tf.matmul(output, Wl_h_l) + sampled_locs[-1] ))
        print "tag [05]"
    else:
        print "tag [04]. output, Wl_h_l:", output.get_shape(), Wl_h_l.get_shape()
        mean_loc = tf.matmul(output, Wl_h_l)
        mean_loc = tf.reshape(mean_loc, [batch_size, 2, nAttn])

    # mean_loc = tf.stop_gradient(mean_loc)
    mean_locs.append(mean_loc)
    print " tag [08]: mean_loc shape", mean_loc.get_shape()
    mean_locs_stopGrad.append(tf.stop_gradient(mean_loc))

    # add noise
    # sample_loc = tf.tanh(mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd))
    if nAttn==1:
        sample_loc = tf.maximum(-1.0, tf.minimum(1.0, \
            mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)))
    else:
        sample_loc = tf.maximum(-1.0, tf.minimum(1.0, \
            mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)))
        sample_loc = tf.reshape(sample_loc, [batch_size, 2, nAttn])

    # don't propagate throught the locations
    # sample_loc = tf.stop_gradient(sample_loc)
    sampled_locs.append(sample_loc)
    sampled_locs_stopGrad.append(tf.stop_gradient(sample_loc))

    if nAttn==1:
        return get_glimpse(sample_loc)
    else:
        return get_glimpses(sample_loc, nAttn)


def affineTransform(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim])
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b


def model():
    # initialize the location under unif[-1,1], for all example in the batch
    if nAttn==1:
        initial_loc = tf.random_uniform((batch_size, 2), minval=-1, maxval=1)
    else:
        initial_loc = tf.random_uniform((batch_size, 2, nAttn), minval=-1, maxval=1)
    mean_locs.append(initial_loc)
    mean_locs_stopGrad.append(tf.stop_gradient(initial_loc))

    initial_loc = tf.tanh(initial_loc + tf.random_normal(initial_loc.get_shape(), 0, loc_sd))
    print 'initial_loc:', initial_loc
    sampled_locs.append(initial_loc)
    sampled_locs_stopGrad.append(tf.stop_gradient(initial_loc))

    # get the input using the input network
    if nAttn==1:
        initial_glimpse = get_glimpse(initial_loc) # MORGAN this yields a single glimpe
    else:
        initial_glimpse = get_glimpses(initial_loc, nAttn) # MORGAN yields a list of glimpses
        print nAttn, "glimpses taken."

    # set up the recurrent structure
    inputs = [0] * nGlimpses
    outputs = [0] * nGlimpses
    glimpse = initial_glimpse
    REUSE = None
    for t in range(nGlimpses):
        if t == 0:  # initialize the hidden state to be the zero vector
            hiddenState_prev = tf.zeros((batch_size, cell_size))
        else:
            hiddenState_prev = outputs[t-1]

        # forward prop
        with tf.variable_scope("coreNetwork", reuse=REUSE):
            # the next hidden state is a function of the previous hidden state and the current glimpse
            print cell_size, hiddenState_prev.get_shape(), glimpse.get_shape(), Wc_g_h.get_shape(), Bc_g_h.get_shape()
#            hiddenState = tf.nn.relu(affineTransform(hiddenState_prev, cell_size * nAttn) + \
#                                     (tf.matmul(glimpse, Wc_g_h) + Bc_g_h))
            hiddenState_0 = affineTransform(hiddenState_prev, cell_size) 
#                                     (tf.matmul(glimpse, Wc_g_h) + Bc_g_h))
            print "hiddenState_0", hiddenState_0.get_shape()
            print "  shapes: glimpse, Wc_g_h:", glimpse.get_shape(), Wc_g_h.get_shape()
            intermediate_0 = tf.matmul(glimpse, Wc_g_h)
            print "intermediate_0", intermediate_0.get_shape()
            intermediate_1 = intermediate_0 +  Bc_g_h
            print "intermediate_1", intermediate_1.get_shape()
            hiddenState_1 = hiddenState_0 + intermediate_1
            print "hiddenState_1", hiddenState_1.get_shape()
            hiddenState = tf.nn.relu(hiddenState_1)
#            hiddenState = tf.nn.relu(affineTransform(hiddenState_prev, cell_size * nAttn) + \
#                                     (tf.matmul(glimpse, Wc_g_h) + Bc_g_h))

        # save the current glimpse and the hidden state
        inputs[t] = glimpse
        outputs[t] = hiddenState
        # get the next input glimpse
        if t != nGlimpses -1:
            glimpse = get_next_input(hiddenState)
        else:
            baseline = tf.sigmoid(tf.matmul(hiddenState, Wb_h_b) + Bb_h_b) 
            # MORGAN: perhaps combine the multiple attentions here?
            baselines.append(baseline)
        REUSE = True  # share variables for later recurrence

    print 'sampled_locs shapes:', [s.get_shape() for s in sampled_locs]
    return outputs


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# to use for maximum likelihood with input location
def gaussian_pdf(mean, sample):
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * np.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
    return Z * tf.exp(a)


def calc_reward(outputs):

    # consider the action at the last time step
    outputs = outputs[-1] # look at ONLY THE END of the sequence
    outputs = tf.reshape(outputs, (batch_size, cell_out_size))

    # get the baseline
    b = tf.pack(baselines)
    b = tf.concat(2, [b, b])
    print 'tag {12}: baselines shape:', b.get_shape(), [B.get_shape() for B in baselines]
    if nAttn==1:
        b = tf.reshape(b, (batch_size, (nGlimpses) * 2))
    else:
        b = tf.reshape(b, (batch_size, (nGlimpses) * 2))
        #b = tf.reshape(b, (batch_size, nGlimpses*2, nAttn))
        print "tag [09] b shape:", b.get_shape()
    no_grad_b = tf.stop_gradient(b)

    # get the action(classification)
    p_y = tf.nn.softmax(tf.matmul(outputs, Wa_h_a) + Ba_h_a)
    max_p_y = tf.arg_max(p_y, 1)
    print 'tag {13}: max_p_y shape, p_y shape',max_p_y.get_shape(), p_y.get_shape()
    correct_y = tf.cast(labels_placeholder, tf.int64)

    # reward for all examples in the batch
    R = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)
    reward = tf.reduce_mean(R) # mean reward
    print R.get_shape()
    if nAttn==1:
        R = tf.reshape(R, (batch_size, 1))
    else:
        R = tf.reshape(R, (batch_size, 1))
        #R = tf.reshape(R, (batch_size, 1, nAttn))
    R = tf.tile(R, [1, (nGlimpses)*2])
    print "tag [11] R shape:", R.get_shape()

    # get the location
    p_loc = gaussian_pdf(mean_locs_stopGrad, sampled_locs_stopGrad)
    p_loc = tf.tanh(p_loc)
    # p_loc_orig = p_loc
    if nAttn==1:
        p_loc = tf.reshape(p_loc, (batch_size, (nGlimpses) * 2))
    else:
        p_loc = tf.reshape(p_loc, (batch_size, (nGlimpses) * 2, nAttn))

    # define the cost function
    J = tf.concat(1, [tf.log(p_y + SMALL_NUM) * (onehot_labels_placeholder), tf.log(p_loc + SMALL_NUM) * (R - no_grad_b)])
    J = tf.reduce_sum(J, 1)
    J = J - tf.reduce_sum(tf.square(R - b), 1)
    J = tf.reduce_mean(J, 0)
    cost = -J

    # define the optimizer
    optimizer = tf.train.MomentumOptimizer(lr, momentumValue)
    train_op = optimizer.minimize(cost, global_step)

    return cost, reward, max_p_y, correct_y, train_op, b, tf.reduce_mean(b), tf.reduce_mean(R - b), lr


def preTrain(outputs):
    lr_r = 1e-3
    # consider the action at the last time step
    outputs = outputs[-1] # look at ONLY THE END of the sequence
    outputs = tf.reshape(outputs, (batch_size, cell_out_size))

    # get the location
    p_loc_r = gaussian_pdf(mean_locs_stopGrad, sampled_locs_stopGrad)
    p_loc_r = tf.tanh(p_loc_r)

    reconstruction = tf.sigmoid(tf.matmul(outputs, Wr_h_r) + Br_h_r)
    reconstructionCost = tf.reduce_mean(tf.square(inputs_placeholder - reconstruction)) * tf.log(p_loc_r + SMALL_NUM)

    train_op_r = tf.train.RMSPropOptimizer(lr_r).minimize(reconstructionCost)
    return tf.reduce_mean(reconstructionCost), reconstruction, train_op_r



def evaluate():
    data = dataset.test
    batches_in_epoch = len(data._images) // batch_size
    accuracy = 0

    for i in xrange(batches_in_epoch):
        nextX, nextY = dataset.test.next_batch(batch_size)
        if translateMnist:
            nextX, _ = convertTranslated(nextX, MNIST_SIZE, img_size)
        feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY,
                     onehot_labels_placeholder: dense_to_one_hot(nextY)}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r

    accuracy /= batches_in_epoch
    print("ACCURACY: " + str(accuracy))


def convertTranslated(images, initImgSize, finalImgSize):
    size_diff = finalImgSize - initImgSize
    newimages = np.zeros([batch_size, finalImgSize*finalImgSize])
    imgCoord = np.zeros([batch_size,2])
    for k in xrange(batch_size):
        image = images[k, :]
        image = np.reshape(image, (initImgSize, initImgSize))
        # generate and save random coordinates
        randX = random.randint(0, size_diff)
        randY = random.randint(0, size_diff)
        imgCoord[k,:] = np.array([randX, randY])
        # padding
        image = np.lib.pad(image, ((randX, size_diff - randX), (randY, size_diff - randY)), 'constant', constant_values = (0))
        newimages[k, :] = np.reshape(image, (finalImgSize*finalImgSize))

    return newimages, imgCoord



def toMnistCoordinates(coordinate_tanh):
    '''
    Transform coordinate in [-1,1] to mnist
    :param coordinate_tanh: vector in [-1,1] x [-1,1]
    :return: vector in the corresponding mnist coordinate
    '''
    return np.round(((coordinate_tanh + 1) / 2.0) * img_size)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('param_summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('param_mean/' + name, mean)
        with tf.name_scope('param_stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('param_sttdev/' + name, stddev)
        tf.scalar_summary('param_max/' + name, tf.reduce_max(var))
        tf.scalar_summary('param_min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def drawMGlimpses(X,Y, img, img_size, sampled_locs):
    print X.shape, Y.shape
    plt.ylim((img_size - 1, 0))
    plt.xlim((0, img_size - 1))

    # transform the coordinate to mnist map
    sampled_locs_mnist_fetched = toMnistCoordinates(sampled_locs_fetched)
    # visualize the trace of successive nGlimpses (note that x and y coordinates are "flipped")
    plt.plot(sampled_locs_mnist_fetched[0, :, 1], sampled_locs[0, :, 0], '-o', color='lawngreen')
    plt.plot(sampled_locs_mnist_fetched[0, -1, 1], sampled_locs[0, -1, 0], 'o', color='red')


def plotWholeImg(img, img_size, sampled_locs_fetched):
    plt.imshow(np.reshape(img, [img_size, img_size]),
               cmap=plt.get_cmap('gray'), interpolation="nearest")

    plt.ylim((img_size - 1, 0))
    plt.xlim((0, img_size - 1))

    # transform the coordinate to mnist map
    sampled_locs_mnist_fetched = toMnistCoordinates(sampled_locs_fetched)
    # visualize the trace of successive nGlimpses (note that x and y coordinates are "flipped")
    plt.plot(sampled_locs_mnist_fetched[0, :, 1], sampled_locs_mnist_fetched[0, :, 0], '-o',
             color='lawngreen')
    plt.plot(sampled_locs_mnist_fetched[0, -1, 1], sampled_locs_mnist_fetched[0, -1, 0], 'o',
             color='red')



with tf.Graph().as_default():

    # set the learning rate
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(initLr, global_step, lrDecayFreq, lrDecayRate, staircase=True)

    # preallocate x, y, baseline
    if nAttn==1:
      labels = tf.placeholder("float32", shape=[batch_size, n_classes])
      labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size), name="labels_raw")
      onehot_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 10), name="labels_onehot")
      inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_size * img_size), name="images")
    else:
      labels = tf.placeholder("float32", shape=[batch_size, n_classes])
      labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size), name="labels_raw")
      onehot_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 10), name="labels_onehot")
      inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_size * img_size), \
                           name="images") # MORGAN tag here.

    # declare the model parameters, here're naming rule:
    # the 1st captical letter: weights or bias (W = weights, B = bias)
    # the 2nd lowercase letter: the network (e.g.: g = glimpse network)
    # the 3rd and 4th letter(s): input-output mapping, which is clearly written in the variable name argument

    if nAttn==1:
        Wg_l_h = weight_variable((2, hl_size), "glimpseNet_wts_location_hidden", True  )
        Bg_l_h = weight_variable((1,hl_size), "glimpseNet_bias_location_hidden", True)

        Wg_g_h = weight_variable((totalSensorBandwidth, hg_size), "glimpseNet_wts_glimpse_hidden", True)
        Bg_g_h = weight_variable((1,hg_size), "glimpseNet_bias_glimpse_hidden", True)

        Wg_hg_gf1 = weight_variable((hg_size, g_size), "glimpseNet_wts_hiddenGlimpse_glimpseFeature1", True)
        Wg_hl_gf1 = weight_variable((hl_size, g_size), "glimpseNet_wts_hiddenLocation_glimpseFeature1", True)
        Bg_hlhg_gf1 = weight_variable((1,g_size), "glimpseNet_bias_hGlimpse_hLocs_glimpseFeature1", True)

        Wc_g_h = weight_variable((g_size, cell_size), "coreNet_wts_glimpse_hidden", True)
        Bc_g_h = weight_variable((1,cell_size), "coreNet_bias_glimpse_hidden", True)

        Wr_h_r = weight_variable((cell_out_size, img_size**2), "reconstructionNet_wts_hidden_action", True)
        Br_h_r = weight_variable((1, img_size**2), "reconstructionNet_bias_hidden_action", True)

        Wb_h_b = weight_variable((cell_size, 1), "baselineNet_wts_hiddenState_baseline", True)
        Bb_h_b = weight_variable((1,1), "baselineNet_bias_hiddenState_baseline", True)

        Wl_h_l = weight_variable((cell_out_size, 2), "locationNet_wts_hidden_location", True)

        Wa_h_a = weight_variable((cell_out_size, n_classes), "actionNet_wts_hidden_action", True)
        Ba_h_a = weight_variable((1,n_classes),  "actionNet_bias_hidden_action", True)
    else:
        # MORGAN list of matrices that are changed: Wg_l_h, Bg_l_h, Wl_h_l, Wg_g_h, Bg_g_h
        # update: above not true.
        # update: after meeting with Andrew, recommends combining the multiple glimpses in the equation
        #   that makes the corenet from glimpseFeatures and reccurent previous hidden.
        # new changed matrices: Wc_g_h, Bc_g_h
        # new updates, 10/31: Wl_h_l, Wc_g_h, Bb_h_b, Wb_h_b
        Wg_l_h = weight_variable((2, hl_size), "glimpseNet_wts_location_hidden", True  )
        Bg_l_h = weight_variable((1,hl_size), "glimpseNet_bias_location_hidden", True)

        Wg_g_h = weight_variable(( totalSensorBandwidth,hg_size), "glimpseNet_wts_glimpse_hidden", True)
        Bg_g_h = weight_variable((1,hg_size), "glimpseNet_bias_glimpse_hidden", True)

        Wg_hg_gf1 = weight_variable((hg_size, g_size), "glimpseNet_wts_hiddenGlimpse_glimpseFeature1", True)
        Wg_hl_gf1 = weight_variable((hl_size, g_size), "glimpseNet_wts_hiddenLocation_glimpseFeature1", True)
        Bg_hlhg_gf1 = weight_variable((1,g_size), "glimpseNet_bias_hGlimpse_hLocs_glimpseFeature1", True)

        Wc_g_h = weight_variable((g_size*nAttn, cell_size), "coreNet_wts_glimpse_hidden", True)
        Bc_g_h = weight_variable((1, cell_size), "coreNet_bias_glimpse_hidden", True)

        Wr_h_r = weight_variable((cell_out_size, img_size**2), "reconstructionNet_wts_hidden_action", True)
        Br_h_r = weight_variable((1, img_size**2), "reconstructionNet_bias_hidden_action", True)

        Wb_h_b = weight_variable((cell_size, 1), "baselineNet_wts_hiddenState_baseline", True)
        Bb_h_b = weight_variable((1,1), "baselineNet_bias_hiddenState_baseline", True)

        # MORGAN: change following line: increase dimensions by 1, size nAttn?
        Wl_h_l = weight_variable((cell_out_size, 2*nAttn), "locationNet_wts_hidden_location", True)
        # MORGAN: Suspected bug makes this the better option potentially:
        #Wl_h_l = weight_variable((cell_out_size, 2), "locationNet_wts_hidden_location", True)

        Wa_h_a = weight_variable((cell_out_size, n_classes), "actionNet_wts_hidden_action", True)
        Ba_h_a = weight_variable((1,n_classes),  "actionNet_bias_hidden_action", True)


    # query the model ouput
    outputs = model()

    # convert list of tensors to one big tensor
    print sampled_locs
    print tf.pack(baselines).get_shape() # == (nGlimpses x batchsize x 1)
    # MORGAN todo: update this to reflect nAttn differences.
    # The following has a miscommunication.  Three lines below this errs.
    if nAttn==1:
        sampled_locs = tf.concat(0, sampled_locs)
        sampled_locs = tf.reshape(sampled_locs, (nGlimpses, batch_size, 2))
        sampled_locs = tf.transpose(sampled_locs, [1, 0, 2])

        mean_locs = tf.concat(0, mean_locs)
        mean_locs = tf.reshape(mean_locs, (nGlimpses, batch_size, 2))
        mean_locs = tf.transpose(mean_locs, [1, 0, 2])

        sampled_locs_stopGrad = tf.concat(0, sampled_locs_stopGrad)
        sampled_locs_stopGrad = tf.reshape(sampled_locs_stopGrad, (nGlimpses, batch_size, 2))
        sampled_locs_stopGrad = tf.transpose(sampled_locs_stopGrad, [1, 0, 2])

        mean_locs_stopGrad = tf.concat(0, mean_locs_stopGrad)
        mean_locs_stopGrad = tf.reshape(mean_locs_stopGrad, (nGlimpses, batch_size, 2))
        mean_locs_stopGrad = tf.transpose(mean_locs_stopGrad, [1, 0, 2])

    else:
        sampled_locs = tf.concat(0, sampled_locs)
        print "tag [06]", (nGlimpses, batch_size, 2, nAttn), sampled_locs.get_shape()
        sampled_locs = tf.reshape(sampled_locs, (nGlimpses, batch_size, 2, nAttn))
        sampled_locs = tf.transpose(sampled_locs, [1, 0, 2, 3])

        print [m.get_shape() for m in mean_locs]
        mean_locs = tf.concat(0, mean_locs)
        print "tag [07]", (nGlimpses, batch_size, 2), mean_locs.get_shape()
        mean_locs = tf.reshape(mean_locs, (nGlimpses, batch_size, 2, nAttn))
        mean_locs = tf.transpose(mean_locs, [1, 0, 2, 3])

        sampled_locs_stopGrad = tf.concat(0, sampled_locs_stopGrad)
        sampled_locs_stopGrad = tf.reshape(sampled_locs_stopGrad, (nGlimpses, batch_size, 2, nAttn))
        sampled_locs_stopGrad = tf.transpose(sampled_locs_stopGrad, [1, 0, 2, 3])

        mean_locs_stopGrad = tf.concat(0, mean_locs_stopGrad)
        mean_locs_stopGrad = tf.reshape(mean_locs_stopGrad, (nGlimpses, batch_size, 2, nAttn))
        mean_locs_stopGrad = tf.transpose(mean_locs_stopGrad, [1, 0, 2, 3])

    glimpse_images = tf.concat(0, glimpse_images)



    # compute the reward
    reconstructionCost, reconstruction, train_op_r = preTrain(outputs)
    cost, reward, predicted_labels, correct_labels, train_op, b, avg_b, rminusb, lr = calc_reward(outputs)

    # tensorboard visualization for the parameters
    variable_summaries(Wg_l_h, "glimpseNet_wts_location_hidden")
    variable_summaries(Bg_l_h, "glimpseNet_bias_location_hidden")
    variable_summaries(Wg_g_h, "glimpseNet_wts_glimpse_hidden")
    variable_summaries(Bg_g_h, "glimpseNet_bias_glimpse_hidden")
    variable_summaries(Wg_hg_gf1, "glimpseNet_wts_hiddenGlimpse_glimpseFeature1")
    variable_summaries(Wg_hl_gf1, "glimpseNet_wts_hiddenLocation_glimpseFeature1")
    variable_summaries(Bg_hlhg_gf1, "glimpseNet_bias_hGlimpse_hLocs_glimpseFeature1")

    variable_summaries(Wc_g_h, "coreNet_wts_glimpse_hidden")
    variable_summaries(Bc_g_h, "coreNet_bias_glimpse_hidden")

    variable_summaries(Wb_h_b, "baselineNet_wts_hiddenState_baseline")
    variable_summaries(Bb_h_b, "baselineNet_bias_hiddenState_baseline")

    variable_summaries(Wl_h_l, "locationNet_wts_hidden_location")

    variable_summaries(Wa_h_a, 'actionNet_wts_hidden_action')
    variable_summaries(Ba_h_a, 'actionNet_bias_hidden_action')

    # tensorboard visualization for the performance metrics
    tf.scalar_summary("reconstructionCost", reconstructionCost)
    tf.scalar_summary("reward", reward)
    tf.scalar_summary("cost", cost)
    tf.scalar_summary("mean(b)", avg_b)
    tf.scalar_summary(" mean(R - b)", rminusb)
    summary_op = tf.merge_all_summaries()


    ####################################### START RUNNING THE MODEL #######################################
    sess = tf.Session()
    saver = tf.train.Saver()
    b_fetched = np.zeros((batch_size, (nGlimpses)*2))

    init = tf.initialize_all_variables()
    sess.run(init)

    if eval_only:
        evaluate()
    else:
        summary_writer = tf.train.SummaryWriter(summaryFolderName, graph=sess.graph)

        if draw:
            fig = plt.figure(1)
            txt = fig.suptitle("-", fontsize=36, fontweight='bold')
            if not no_display_env:
                plt.ion()
                plt.show()
                plt.subplots_adjust(top=0.7)
                plotImgs = []

        if drawReconsturction:
            fig = plt.figure(2)
            txt = fig.suptitle("-", fontsize=36, fontweight='bold')
            if not no_display_env:
                plt.ion()
                plt.show()

        if preTraining:
            for epoch_r in xrange(1,preTraining_epoch):
                print "PRETRAINING EPOCH",epoch_r
                nextX, _ = dataset.train.next_batch(batch_size)
                nextX_orig = nextX
                if translateMnist:
                    nextX, _ = convertTranslated(nextX, MNIST_SIZE, img_size)

                fetches_r = [reconstructionCost, reconstruction, train_op_r]

                reconstructionCost_fetched, reconstruction_fetched, train_op_r_fetched = sess.run(fetches_r, feed_dict={inputs_placeholder: nextX})

                if epoch_r % 20 == 0:
                    print('Step %d: reconstructionCost = %.5f' % (epoch_r, reconstructionCost_fetched))
                    if epoch_r % 100 == 0:
                        if drawReconsturction:
                            fig = plt.figure(2)

                            plt.subplot(1, 2, 1)
                            plt.imshow(np.reshape(nextX[0, :], [img_size, img_size]),
                                       cmap=plt.get_cmap('gray'), interpolation="nearest")
                            plt.ylim((img_size - 1, 0))
                            plt.xlim((0, img_size - 1))

                            plt.subplot(1, 2, 2)
                            plt.imshow(np.reshape(reconstruction_fetched[0, :], [img_size, img_size]),
                                       cmap=plt.get_cmap('gray'), interpolation="nearest")
                            plt.ylim((img_size - 1, 0))
                            plt.xlim((0, img_size - 1))
                            plt.draw()
                            plt.pause(0.0001)
                            # plt.show()
                    if no_display_env:
                        pass# drawMGlimpses()
                        # here, draw the last MxN glimpses.


        # training
        for epoch in xrange(start_step + 1, max_iters):
            print "TRAINING EPOCH",epoch
            start_time = time.time()

            # get the next batch of examples
            nextX, nextY = dataset.train.next_batch(batch_size)
            nextX_orig = nextX
            if translateMnist:
                nextX, nextX_coord = convertTranslated(nextX, MNIST_SIZE, img_size)

            feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY, \
                         onehot_labels_placeholder: dense_to_one_hot(nextY)}

            fetches = [train_op, cost, reward, predicted_labels, correct_labels, glimpse_images, avg_b, rminusb, \
                       mean_locs, sampled_locs, lr]
            # feed them to the model
            results = sess.run(fetches, feed_dict=feed_dict)

            _, cost_fetched, reward_fetched, prediction_labels_fetched, correct_labels_fetched, glimpse_images_fetched, \
            avg_b_fetched, rminusb_fetched, mean_locs_fetched, sampled_locs_fetched, lr_fetched = results


            duration = time.time() - start_time

            if epoch % 20 == 0:
                print('Step %d: cost = %.5f reward = %.5f (%.3f sec) b = %.5f R-b = %.5f, LR = %.5f'
                      % (epoch, cost_fetched, reward_fetched, duration, avg_b_fetched, rminusb_fetched, lr_fetched))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, epoch)
                # if saveImgs:
                #     plt.savefig(imgsFolderName + simulationName + '_ep%.6d.png' % (epoch))

                if epoch % 5000 == 0:
                    saver.save(sess, save_dir + save_prefix + str(epoch) + ".ckpt")
                    evaluate()

                ##### DRAW WINDOW ################
                f_glimpse_images = np.reshape(glimpse_images_fetched, \
                                              (nGlimpses, batch_size, depth, sensorBandwidth, sensorBandwidth))

                if draw:
                    if no_display_env:
                        drawMGlimpses(nextX, nextY, nextX[0,:], img_size, sampled_locs_fetched)
                        plt.savefig(simulationName+'_train_'+str(outfilecount)+'.png')
                        outfilecount+=1
                        #TODO
                        pass
                    if animate:
                        fillList = False
                        if len(plotImgs) == 0:
                            fillList = True

                        # display the first image in the in mini-batch
                        nCols = depth+1
                        plt.subplot2grid((depth, nCols), (0, 1), rowspan=depth, colspan=depth)
                        # display the entire image
                        plotWholeImg(nextX[0, :], img_size, sampled_locs_fetched)

                        # display the glimpses
                        for y in xrange(nGlimpses):
                            txt.set_text('Epoch: %.6d \nPrediction: %i -- Truth: %i\nStep: %i/%i'
                                         % (epoch, prediction_labels_fetched[0], correct_labels_fetched[0], (y + 1), nGlimpses))

                            for x in xrange(depth):
                                plt.subplot(depth, nCols, 1 + nCols * x)
                                if fillList:
                                    plotImg = plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'),
                                                         interpolation="nearest")
                                    plotImg.autoscale()
                                    plotImgs.append(plotImg)
                                else:
                                    plotImgs[x].set_data(f_glimpse_images[y, 0, x])
                                    plotImgs[x].autoscale()
                            fillList = False

                            # fig.canvas.draw()
                            time.sleep(0.1)
                            plt.pause(0.00005)

                    else:
                        txt.set_text('PREDICTION: %i\nTRUTH: %i' % (prediction_labels_fetched[0], correct_labels_fetched[0]))
                        for x in xrange(depth):
                           for y in xrange(nGlimpses):
                              for z in xrange(nAttn):
                                plt.subplot(depth, nGlimpses, x * nGlimpses + y + 1)
                                plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'), interpolation="nearest")

                        plt.draw()
                        time.sleep(0.05)
                        plt.pause(0.0001)

    sess.close()
