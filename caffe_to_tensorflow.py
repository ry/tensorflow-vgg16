import os
os.environ["GLOG_minloglevel"] = "2"

from utils import *
import matplotlib.pyplot as plt
import skimage
import caffe
import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('only_caffe', False,
                            """Only run caffe""")
tf.app.flags.DEFINE_boolean('only_tf', False,
                            """Only run tf""")

def imshow(image):
  skimage.io.imshow(image)
  skimage.io.show()

# input as gotten from skimage.io.imread() that is
# [width, height, 3]
# and scaled between 0 and 1
#
# output is scaled to 0 - 255 and depending on is_caffe
# the shape is
# output caffe: [in_channels, in_width, in_height]
# output tf: [in_height, in_width, in_channels]
def preprocess(img):
  # convert to numpy.ndarray
  out = np.copy(img) * 255
  # swap channel from RGB to BGR
  out = out[:, :, [2,1,0]]
  # sub mean
  out[:,:,0] -= VGG_MEAN[0]
  out[:,:,1] -= VGG_MEAN[1]
  out[:,:,2] -= VGG_MEAN[2]
  out = out.transpose((2,0,1))
  return out

def deprocess(img):
  out = np.copy(img)
  out = out.transpose((1,2,0))

  out[:,:,0] += VGG_MEAN[0]
  out[:,:,1] += VGG_MEAN[1]
  out[:,:,2] += VGG_MEAN[2]
  out = out[:, :, [2,1,0]]
  out /= 255
  return out

#caffe.set_mode_cpu()
net_caffe = caffe.Net("VGG_2014_16.prototxt", "VGG_ILSVRC_16_layers.caffemodel", caffe.TEST)


caffe_layers = {}
for i, layer in enumerate(net_caffe.layers):
    layer_name = net_caffe._layer_names[i]
    caffe_layers[layer_name] = layer

def caffe_weights(layer_name):
    layer = caffe_layers[layer_name]
    return layer.blobs[0].data

def caffe_bias(layer_name):
    layer = caffe_layers[layer_name]
    return layer.blobs[1].data

# converts caffe filter to tf
def caffe2tf_filter(name):
  weights = caffe_weights(name)
  return weights.transpose((3,2,1,0))

def caffe2tf_conv_blob(name):
  blob = net_caffe.blobs[name].data[0]
  return blob.transpose((2,1,0))

def caffe2tf_1d_blob(name):
  blob = net_caffe.blobs[name].data[0]
  return blob

# tensorflow uses [filter_height, filter_width, in_channels, out_channels]
#                  0              1             2            3
# need to transpose channel axis in the weights
# caffe:  a convolution layer with 96 filters of 11 x 11 spatial dimension
# and 3 inputs the blob is 96 x 3 x 11 x 11
# caffe uses [out_channels, in_channels, filter_width, filter_height] 
#             3             2            1             0
def conv_layer(m, bottom, name):
  with tf.variable_scope(name) as scope:
    w = caffe2tf_filter(name)
    #print name + " w shape", w.shape
    conv_weight = tf.constant(w, dtype=tf.float32, name="filter")

    conv = tf.nn.conv2d(bottom, conv_weight, [1, 1, 1, 1], padding='SAME')

    conv_biases = tf.constant(caffe_bias(name), dtype=tf.float32, name="biases")
    bias = tf.nn.bias_add(conv, conv_biases)

    if name == "conv1_1":
      m["conv1_1_weights"] = conv_weight
      m["conv1_1a"] = bias

    relu = tf.nn.relu(bias)
    return relu

def fc_layer(bottom, name):
  shape = bottom.get_shape().as_list()
  shape[0] = 1### FIXME
  dim = 1
  for d in shape[1:]:
     dim *= d
  x = tf.reshape(bottom, [shape[0], dim])

  weights = tf.constant(caffe_weights(name), dtype=tf.float32)
  weights = tf.transpose(weights) # caffe weights are [out, in] and tf is opposite
  biases = tf.constant(caffe_bias(name), dtype=tf.float32)

  # Fully connected layer. Note that the '+' operation automatically
  # broadcasts the biases.
  fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

  return fc

def inference(rgb):
  m = {}

  rgb_scaled = rgb * 255.0

  # Convert RGB to BGR
  red, green, blue = tf.split(3, 3, rgb_scaled)
  assert red.get_shape().as_list()[1:] == [224, 224, 1]
  assert green.get_shape().as_list()[1:] == [224, 224, 1]
  assert blue.get_shape().as_list()[1:] == [224, 224, 1]
  bgr = tf.concat(3, [
    blue - VGG_MEAN[0],
    green - VGG_MEAN[1],
    red - VGG_MEAN[2],
  ])
  assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

  # height needs to come first
  bgr_t = tf.transpose(bgr, (0, 2, 1, 3))
  

  relu1_1 = conv_layer(m, bgr_t, "conv1_1")
  relu1_2 = conv_layer(m, relu1_1, "conv1_2")
  pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  m["relu1_1"] = relu1_1

  relu2_1 = conv_layer(m, pool1, "conv2_1")
  relu2_2 = conv_layer(m, relu2_1, "conv2_2")
  pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  relu3_1 = conv_layer(m, pool2, "conv3_1")
  relu3_2 = conv_layer(m, relu3_1, "conv3_2")
  relu3_3 = conv_layer(m, relu3_2, "conv3_3")
  pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')

  relu4_1 = conv_layer(m, pool3, "conv4_1")
  relu4_2 = conv_layer(m, relu4_1, "conv4_2")
  relu4_3 = conv_layer(m, relu4_2, "conv4_3")
  pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4')

  relu5_1 = conv_layer(m, pool4, "conv5_1")
  relu5_2 = conv_layer(m, relu5_1, "conv5_2")
  relu5_3 = conv_layer(m, relu5_2, "conv5_3")
  pool5 = tf.nn.max_pool(relu5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool5')

  m['pool5'] = pool5

  # TODO this shouldn't be necessary...
  pool5_rotated = tf.transpose(pool5, perm=(0,3,2,1))

  fc6 = fc_layer(pool5_rotated, "fc6")
  assert fc6.get_shape().as_list() == [1, 4096]
  m['fc6a'] = fc6

  relu6 = tf.nn.relu(fc6)
  drop6 = tf.nn.dropout(relu6, 0.5)

  fc7 = fc_layer(drop6, "fc7")
  relu7 = tf.nn.relu(fc7)
  drop7 = tf.nn.dropout(relu7, 0.5)

  fc8 = fc_layer(drop7, "fc8")
  prob = tf.nn.softmax(fc8, name="prob")

  m["prob"] = prob

  return m

def show_caffe_net_input():
  x = net_caffe.blobs['data'].data[0]
  assert x.shape == (3, 224, 224)
  i = deprocess(x)
  imshow(i)

def same_tensor(a, b):
  return np.linalg.norm(a - b) < 1 

def main():
  global tf_activations

  cat = load_image("cat.jpg")
  print "cat shape", cat.shape

  if not FLAGS.only_tf:
    print "caffe session"
    assert same_tensor(deprocess(preprocess(cat)), cat)
    net_caffe.blobs['data'].data[0] = preprocess(cat)
    assert net_caffe.blobs['data'].data[0].shape == (3, 224, 224)
    #show_caffe_net_input()
    net_caffe.forward()
    prob = net_caffe.blobs['prob'].data[0]
    print_prob(prob)
    #show_caffe_net_input()

    caffe_activations = {
      'conv1_1_weights': caffe_weights("conv1_1"),
      'prob': net_caffe.blobs['prob'].data[0],
    }

  if not FLAGS.only_caffe:
    images = tf.placeholder("float", [None, 224, 224, 3], name="images")
    m = inference(images)

    tf_activations = {}

    print "tensorflow session"
    with tf.Session() as sess:
      init = tf.initialize_all_variables()
      sess.run(init)
      print "variables initialized"

      assert cat.shape == (224, 224, 3)
      batch = cat.reshape((1, 224, 224, 3))
      assert batch.shape == (1, 224, 224, 3)

      out = sess.run([m['prob'], m['relu1_1'], m['conv1_1_weights'], \
          m['conv1_1a'], m['pool5'], m['fc6a']], feed_dict={ images: batch })
      tf_activations = {
        'prob': out[0][0],
        'relu1_1': out[1][0],
        'conv1_1_weights': out[2],
        'conv1_1a': out[3][0],
        'pool5': out[4][0],
        'fc6a': out[5][0],
      }

      print_prob(tf_activations['prob'])

      graph = tf.get_default_graph()
      # tf.import_graph_def() to reopen this
      graph_def = graph.as_graph_def()
      print "graph_def byte size", graph_def.ByteSize()
      graph_def_s = graph_def.SerializeToString()

      save_path = "vgg16.tfmodel"
      with open(save_path, "wb") as f:
        f.write(graph_def_s)

      print "saved model to %s" % save_path

  # Now we compare tf_activations to net_caffe's if we ran a forward pass
  # in both networks.
  if not FLAGS.only_caffe and not FLAGS.only_tf:
    assert same_tensor(caffe2tf_conv_blob("conv1_1a"), tf_activations["conv1_1a"])

    assert same_tensor(caffe2tf_filter("conv1_1"), tf_activations['conv1_1_weights'])

    assert same_tensor(caffe2tf_conv_blob("conv1_1"), tf_activations['relu1_1'])

    assert same_tensor(caffe2tf_conv_blob("pool5"), tf_activations['pool5'])

    print "diff fc6a", np.linalg.norm(caffe2tf_1d_blob("fc6a") - tf_activations['fc6a'])
    assert caffe_weights("fc6").shape == (4096, 25088)
    assert caffe_bias("fc6").shape == (4096,)

    assert same_tensor(caffe2tf_1d_blob("fc6a"), tf_activations['fc6a'])

    assert same_tensor(caffe2tf_1d_blob("prob"), tf_activations['prob'])


if __name__ == "__main__":
  main()
