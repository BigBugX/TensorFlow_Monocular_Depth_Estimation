__author__ = 'Will@PCVG'
# Modified from shekkizh's FCN.tensorflow

# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io

def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


def save_image(image, save_dir, name, mean=None):
    """
    Save image by unprocessing if mean given else just save
    :param mean:
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)

# as describe at Sec.4.2
def get_variable(weights, name):
    if name == 'conv1_1_w':
        k1, k2, ic, oc = weights.shape
        concat_weights =  np.random.normal(0.0, 1.0, size=(k1, k2, 2 * ic, oc))
        concat_weights[:, :, 0:ic, :] = weights
        init = tf.constant_initializer(concat_weights, dtype=tf.float32)
        var = tf.get_variable(name=name, initializer=init,  shape=concat_weights.shape)
        return var
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var

def get_variable_dup(weights, name):
    if name == 'MobilenetV2/Conv/weights':
        k1, k2, ic, oc = weights.shape
        concat_weights =  np.random.normal(0.0, 1.0, size=(k1, k2, 2 * ic, oc))
        concat_weights[:, :, 0:ic, :] = weights
        init = tf.constant_initializer(concat_weights, dtype=tf.float32)
        var = tf.get_variable(name=name, initializer=init,  shape=concat_weights.shape)
        return var
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

"""
def conv2d_de(block_net, layer_id, filter, kernel_size, stride, name=None):
    for i in range(0, layer_id):
        input = input + block_net[i]
    conv = tf.nn.conv2d(input, filter, 1, 1, padding="SAME", name=name+'conv1')
    return conv
"""
def de_get_nk(weights, block, layer):
    
    features = 'features/'
    denseblock = 'denseblock'
    denselayer = 'denselayer'
    transition = 'transition'
    conv0 = 'conv0/'
    conv1 = 'conv1/'
    conv2 = 'conv2/'
    norm0 = 'norm0/'
    norm1 = 'norm1/'
    norm2 = 'norm2/'
    sub_l = ['weight', 'bias', 'running_mean', 'running_var']
    denseparams = [6, 12, 32, 32]
    block_outputs = []    
    
    nk_lst = []
    
    if(block == 0):
        for i, layer_name in enumerate(sub_l):
            nk_lst.append(weights[features+norm0+layer_name])
    else:
        for i, layer_name in enumerate(sub_l):
            nk_lst.append(weights[features+denseblock+(str)(block)+'/'
                +denselayer+(str)(layer)+'/'+norm1+layer_name])
        for i, layer_name in enumerate(sub_l):
            nk_lst.append(weights[features+denseblock+(str)(block)+'/'
                +denselayer+(str)(layer)+'/'+norm2+layer_name])
    return nk_lst


def tran_de(weights, input, scope):
    scope_ = 'features/' + 'transition' + scope[-1] + '/'
    weight = weights[scope_+'norm/weight']
    bias = weights[scope_+'norm/bias']
    mean = weights[scope_+'norm/running_mean']
    var = weights[scope_+'norm/running_var']
    ck = weights[scope_+'conv/weight']
    output = batch_norm_de(input, mean, var, bias, weight, scope='bn')
    output = relu_de(output, name='relu')
    output = conv2d_de(output, ck, 1, 1, name='conv')
    output = avgpool_de(output, 2, 2, "VALID", name='pool')
    return output


def relu_de(input, name=None):
    output = tf.nn.relu(input, name=name)
    return output

def batch_norm_de(x, mean, var, offset, scale, scope=None, eps=1e-5):
    with tf.variable_scope(scope):
        mean_tmp = get_variable_de(mean, name='running_mean')
        var_tmp = get_variable_de(var, name='running_var')
        beta = get_variable_de(offset, name='bias')
        gamma = get_variable_de(scale, name='weight')
        normed = tf.nn.batch_normalization(x, mean_tmp, var_tmp, beta, gamma, eps)
    return normed


def maxpool_de(input, pool_size, stride, padding="SAME", name=None):
    output = tf.nn.max_pool(input, [1, pool_size, pool_size, 1], [1, stride, stride, 1], padding=padding, name=name)
    return output


def avgpool_de(input, pool_size, stride, padding="VALID", name=None):
    output = tf.nn.avg_pool(input, [1, pool_size, pool_size, 1], [1, stride, stride, 1], padding, name=name)
    return output


def conv2d_de(input, filters, kernel_size, stride=1, name=None, padding="SAME"):
    filters = filters.transpose((2,3,1,0))
    #filters = get_variable_de(filters, name=name+'init_var')
    conv = tf.nn.conv2d(input, filters, [1, stride, stride, 1], padding, name=name)
    return conv


def conv2d_bias_de(feature_maps, filters, kernel_size, stride=1, name=None, padding="SAME", bias=None):
    conv = tf.nn.conv2d(feature_maps, filters, [1, stride, stride, 1], name=None, padding="SAME")
    return tf.nn.bias_add(conv, bias, name=name)


def up_block_refine_de(feature_maps, filters, width, height, up_refine_idx):
    with tf.variable_scope("up%d" % up_refine_idx):
        name = 'up%d/' % up_refine_idx
        c1_w = filters[name + 'conv1/weight'] # name = upblock layer's name, e.g. 'up1/'
        bn1_w = filters[name + 'bn1/weight']
        bn1_b = filters[name + 'bn1/bias']
        bn1_m = filters[name + 'bn1/running_mean']
        bn1_v = filters[name + 'bn1/running_var']   

        c1_2_w = filters[name + 'conv1_2/weight']
        bn1_2_w = filters[name + 'bn1_2/weight']
        bn1_2_b = filters[name + 'bn1_2/bias']
        bn1_2_m = filters[name + 'bn1_2/running_mean']
        bn1_2_v = filters[name + 'bn1_2/running_var']   

        c2_w = filters[name + 'conv2/weight']
        bn2_w = filters[name + 'bn2/weight']
        bn2_b = filters[name + 'bn2/bias']
        bn2_m = filters[name + 'bn2/running_mean']
        bn2_v = filters[name + 'bn2/running_var']   

        x = upsampling_init_de(feature_maps, width, height)  

        res1 = conv2d_de(x, c1_w, 5, 1, padding="SAME", name='conv1')
        res1 = batch_norm_de(res1, bn1_m, bn1_v, bn1_b, bn1_w, scope='bn1')
        x_conv1 = relu_de(res1, name='relu')    

        bran1 = conv2d_de(x_conv1, c1_2_w, 3, 1, padding="SAME", name='conv1_2')
        bran1 = batch_norm_de(bran1, bn1_2_m, bn1_2_v, bn1_2_b, bn1_2_w, scope='bn1_2') 

        bran2 = conv2d_de(x, c2_w, 5, 1, padding="SAME", name='conv2')
        bran2 = batch_norm_de(bran2, bn2_m, bn2_v, bn2_b, bn2_w, scope='bn2')   

        #output = relu_de(bran1 + bran2, name='relu')
        output = relu_de(tf.add(bran1, bran2, name="fuse"), name='relu')
    return output


def up_block_de(feature_maps, filters, up_idx):
    with tf.variable_scope("up%d" % up_idx):
        name = 'up%d/' % up_idx
        c1_w = filters[name + 'conv1/weight'] # name = upblock layer's name, e.g. 'up1/'
        #c1_w = c1_w.transpose((2,3,1,0))
        bn1_w = filters[name + 'bn1/weight']
        bn1_b = filters[name + 'bn1/bias']
        bn1_m = filters[name + 'bn1/running_mean']
        bn1_v = filters[name + 'bn1/running_var']   

        c1_2_w = filters[name + 'conv1_2/weight']
        #c1_2_w = c1_2_w.transpose((2,3,1,0))
        bn1_2_w = filters[name + 'bn1_2/weight']
        bn1_2_b = filters[name + 'bn1_2/bias']
        bn1_2_m = filters[name + 'bn1_2/running_mean']
        bn1_2_v = filters[name + 'bn1_2/running_var']   

        c2_w = filters[name + 'conv2/weight']
        #c2_w = c2_w.transpose((2,3,1,0))
        bn2_w = filters[name + 'bn2/weight']
        bn2_b = filters[name + 'bn2/bias']
        bn2_m = filters[name + 'bn2/running_mean']
        bn2_v = filters[name + 'bn2/running_var']   

        if(up_idx == 1):
            x = upsampling_init_de(feature_maps, 19, 14)
        elif(up_idx == 2):
            x = upsampling_init_de(feature_maps, 38, 28)
        elif(up_idx == 3):
            x = upsampling_init_de(feature_maps, 76, 57)
        else:
            x = upsampling_de(feature_maps, 2)  

        res1 = conv2d_de(x, c1_w, 5, 1, padding="SAME", name='conv1')
        res1 = batch_norm_de(res1, bn1_m, bn1_v, bn1_b, bn1_w, scope='bn1')
        x_conv1 = relu_de(res1, name='relu')    

        bran1 = conv2d_de(x_conv1, c1_2_w, 3, 1, padding="SAME", name='conv1_2')
        bran1 = batch_norm_de(bran1, bn1_2_m, bn1_2_v, bn1_2_b, bn1_2_w, scope='bn1_2') 

        bran2 = conv2d_de(x, c2_w, 5, 1, padding="SAME", name='conv2')
        bran2 = batch_norm_de(bran2, bn2_m, bn2_v, bn2_b, bn2_w, scope='bn2')   

        #output = relu_de(bran1 + bran2, name='relu')
        output = relu_de(tf.add(bran1, bran2, name="fuse"), name='relu')
    return output


def upsampling_de(feature_maps, scale_factor):
    shape = feature_maps.get_shape().as_list()
    feature_maps = tf.image.resize_images(feature_maps, (shape[1]*scale_factor, shape[2]*scale_factor), align_corners=True)
    return feature_maps

def upsampling_init_de(feature_maps, height, width):
    shape = feature_maps.get_shape().as_list()
    feature_maps = tf.image.resize_images(feature_maps, (width, height), align_corners=True)
    return feature_maps



def get_variable_de(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_basic_expt(x, W):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return conv


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def conv2d_transpose_strided_expt(x, W, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return conv


def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


def process_image(image, mean_pixel):
    return image - mean_pixel


def unprocess_image(image, mean_pixel):
    return image + mean_pixel


def bottleneck_unit(x, out_chan1, out_chan2, down_stride=False, up_stride=False, name=None):
    """
    Modified implementation from github ry?!
    """

    def conv_transpose(tensor, out_channel, shape, strides, name=None):
        out_shape = tensor.get_shape().as_list()
        in_channel = out_shape[-1]
        kernel = weight_variable([shape, shape, out_channel, in_channel], name=name)
        shape[-1] = out_channel
        return tf.nn.conv2d_transpose(x, kernel, output_shape=out_shape, strides=[1, strides, strides, 1],
                                      padding='SAME', name='conv_transpose')

    def conv(tensor, out_chans, shape, strides, name=None):
        in_channel = tensor.get_shape().as_list()[-1]
        kernel = weight_variable([shape, shape, in_channel, out_chans], name=name)
        return tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME', name='conv')

    def bn(tensor, name=None):
        """
        :param tensor: 4D tensor input
        :param name: name of the operation
        :return: local response normalized tensor - not using batch normalization :(
        """
        return tf.nn.lrn(tensor, depth_radius=5, bias=2, alpha=1e-4, beta=0.75, name=name)

    in_chans = x.get_shape().as_list()[3]

    if down_stride or up_stride:
        first_stride = 2
    else:
        first_stride = 1

    with tf.variable_scope('res%s' % name):
        if in_chans == out_chan2:
            b1 = x
        else:
            with tf.variable_scope('branch1'):
                if up_stride:
                    b1 = conv_transpose(x, out_chans=out_chan2, shape=1, strides=first_stride,
                                        name='res%s_branch1' % name)
                else:
                    b1 = conv(x, out_chans=out_chan2, shape=1, strides=first_stride, name='res%s_branch1' % name)
                b1 = bn(b1, 'bn%s_branch1' % name, 'scale%s_branch1' % name)

        with tf.variable_scope('branch2a'):
            if up_stride:
                b2 = conv_transpose(x, out_chans=out_chan1, shape=1, strides=first_stride, name='res%s_branch2a' % name)
            else:
                b2 = conv(x, out_chans=out_chan1, shape=1, strides=first_stride, name='res%s_branch2a' % name)
            b2 = bn(b2, 'bn%s_branch2a' % name, 'scale%s_branch2a' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2b'):
            b2 = conv(b2, out_chans=out_chan1, shape=3, strides=1, name='res%s_branch2b' % name)
            b2 = bn(b2, 'bn%s_branch2b' % name, 'scale%s_branch2b' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2c'):
            b2 = conv(b2, out_chans=out_chan2, shape=1, strides=1, name='res%s_branch2c' % name)
            b2 = bn(b2, 'bn%s_branch2c' % name, 'scale%s_branch2c' % name)

        x = b1 + b2
        return tf.nn.relu(x, name='relu')


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.histogram_summary(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.histogram_summary(var.op.name + "/activation", var)
        tf.scalar_summary(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.histogram_summary(var.op.name + "/gradient", grad)
