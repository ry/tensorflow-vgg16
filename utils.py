import skimage
import skimage.io
import skimage.transform
import numpy as np

synset = [l.strip() for l in open('synset.txt').readlines()]
VGG_MEAN = np.array([103.939, 116.779, 123.68])

def load_image(path):
  # load image
  img = skimage.io.imread(path)
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  return resized_img

# input as gotten from skimage.io.imread() that is
# [width, height, 3]
# and scaled between 0 and 1
#
# output is scaled to 0 - 255 and depending on is_caffe
# the shape is
# output caffe: [in_channels, in_width, in_height]
# output tf: [in_height, in_width, in_channels]
def preprocess(img, is_caffe):
  # convert to numpy.ndarray
  out = np.copy(img) * 255
  # swap channel from RGB to BGR
  out = out[:, :, [2,1,0]]
  # sub mean
  out[:,:,0] -= VGG_MEAN[0]
  out[:,:,1] -= VGG_MEAN[1]
  out[:,:,2] -= VGG_MEAN[2]
  if is_caffe:
    out = out.transpose((2,0,1))
  else:
    out = out.transpose((1,0,2))
  return out

def deprocess(img, is_caffe):
  out = np.copy(img)
  if is_caffe:
    out = out.transpose((1,2,0))
  else:
    out = out.transpose((1,0,2))

  out[:,:,0] += VGG_MEAN[0]
  out[:,:,1] += VGG_MEAN[1]
  out[:,:,2] += VGG_MEAN[2]
  out = out[:, :, [2,1,0]]
  out /= 255
  return out

def print_prob(prob):
  #print prob
  print "prob shape", prob.shape
  pred = np.argsort(prob)[::-1]

  # Get top1 label
  top1 = synset[pred[0]]
  print "Top1: ", top1
  # Get top5 label
  top5 = [synset[pred[i]] for i in range(5)]
  print "Top5: ", top5
