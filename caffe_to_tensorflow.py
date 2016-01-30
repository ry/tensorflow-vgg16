import os
os.environ["GLOG_minloglevel"] = "2"

from utils import *
import matplotlib.pyplot as plt
import skimage
import caffe
import numpy as np
import tensorflow as tf
import vgg16



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('only_caffe', False,
                            """Only run caffe""")
tf.app.flags.DEFINE_boolean('only_tf', False,
                            """Only run tf""")

def tf_show_layer(image):
  skimage.io.imshow(image)
  skimage.io.show()

# input as gotten from skimage.io.imread() that is
# [height, width, 3] and scaled between 0 and 1
# output is scaled to 0 - 255 with mean subtracted
# output [in_channels, in_height, in_width]
def preprocess(img):
  out = np.copy(img) * 255
  out = out[:, :, [2,1,0]] # swap channel from RGB to BGR
  # sub mean
  out[:,:,0] -= vgg16.VGG_MEAN[0]
  out[:,:,1] -= vgg16.VGG_MEAN[1]
  out[:,:,2] -= vgg16.VGG_MEAN[2]
  out = out.transpose((2,0,1)) # h, w, c -> c, h, w
  return out

def deprocess(img):
  out = np.copy(img)
  out = out.transpose((1,2,0)) # c, h, w -> h, w, c

  out[:,:,0] += vgg16.VGG_MEAN[0]
  out[:,:,1] += vgg16.VGG_MEAN[1]
  out[:,:,2] += vgg16.VGG_MEAN[2]
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
# tensorflow uses [filter_height, filter_width, in_channels, out_channels]
#                  2               3            1            0
# need to transpose channel axis in the weights
# caffe:  a convolution layer with 96 filters of 11 x 11 spatial dimension
# and 3 inputs the blob is 96 x 3 x 11 x 11
# caffe uses [out_channels, in_channels, filter_height, filter_width] 
#             0             1            2              3
def caffe2tf_filter(name):
  f = caffe_weights(name)
  return f.transpose((2, 3, 1, 0))

# caffe blobs are [ channel, height, width ]
# this returns  [ height, width, channel ]
def caffe2tf_conv_blob(name):
  blob = net_caffe.blobs[name].data[0]
  return blob.transpose((1, 2, 0))

def caffe2tf_1d_blob(name):
  blob = net_caffe.blobs[name].data[0]
  return blob

class ModelFromCaffe(vgg16.Model):
    def get_conv_filter(self, name):
        w = caffe2tf_filter(name)
        return tf.constant(w, dtype=tf.float32, name="filter")

    def get_bias(self, name):
        b = caffe_bias(name)
        return tf.constant(b, dtype=tf.float32, name="bias")

    def get_fc_weight(self, name):
        cw = caffe_weights(name)
        if name == "fc6":
            assert cw.shape == (4096, 25088)
            cw = cw.reshape((4096, 512, 7, 7)) 
            cw = cw.transpose((2, 3, 1, 0))
            cw = cw.reshape(25088, 4096)
        else:
            cw = cw.transpose((1, 0))

        return tf.constant(cw, dtype=tf.float32, name="weight")

def show_caffe_net_input():
  x = net_caffe.blobs['data'].data[0]
  assert x.shape == (3, 224, 224)
  i = deprocess(x)
  skimage.io.imshow(i)
  skimage.io.show()

def same_tensor(a, b):
  return np.linalg.norm(a - b) < 0.1

def main():
  global tf_activations

  cat = load_image("cat.jpg")

  run_caffe = not FLAGS.only_tf
  run_tf = not FLAGS.only_caffe
  ran_both = run_caffe and run_tf

  if run_caffe:
    print "caffe session"
    assert same_tensor(deprocess(preprocess(cat)), cat)
    assert (0 <= cat).all() and (cat <= 1.0).all()
    net_caffe.blobs['data'].data[0] = preprocess(cat)
    assert net_caffe.blobs['data'].data[0].shape == (3, 224, 224)
    #show_caffe_net_input()
    net_caffe.forward()
    prob = net_caffe.blobs['prob'].data[0]
    top1 = print_prob(prob)
    assert top1 == "n02123045 tabby, tabby cat"

  if run_tf:
    print "tensorflow session"

    images = tf.placeholder("float", [None, 224, 224, 3], name="images")
    m = ModelFromCaffe()
    m.build(images)

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())

      assert cat.shape == (224, 224, 3)
      batch = cat.reshape((1, 224, 224, 3))
      assert batch.shape == (1, 224, 224, 3)
      assert (0 <= batch).all() and (batch <= 1.0).all()

      out = sess.run([m.prob, m.relu1_1, m.pool5, m.fc6], feed_dict={ images: batch })
      tf_activations = {
        'prob': out[0][0],
        'relu1_1': out[1][0],
        'pool5': out[2][0],
        'fc6': out[3][0],
      }

      top1 = print_prob(tf_activations['prob'])
      assert top1 == "n02123045 tabby, tabby cat"

  # Now we compare tf_activations to net_caffe's if we ran a forward pass
  # in both networks.
  if ran_both:
    assert same_tensor(caffe2tf_conv_blob("conv1_1"), tf_activations['relu1_1'])

    assert same_tensor(caffe2tf_conv_blob("pool5"), tf_activations['pool5'])

    print "diff fc6", np.linalg.norm(caffe2tf_1d_blob("fc6a") - tf_activations['fc6'])
    assert caffe_weights("fc6").shape == (4096, 25088)
    assert caffe_bias("fc6").shape == (4096,)

    assert same_tensor(caffe2tf_1d_blob("fc6a"), tf_activations['fc6'])

    assert same_tensor(caffe2tf_1d_blob("prob"), tf_activations['prob'])

    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    print "graph_def byte size", graph_def.ByteSize()
    graph_def_s = graph_def.SerializeToString()

    save_path = "vgg16.tfmodel"
    with open(save_path, "wb") as f:
      f.write(graph_def_s)

    print "saved model to %s" % save_path


if __name__ == "__main__":
  main()
