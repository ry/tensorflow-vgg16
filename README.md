# TensorFlow VGG-16 pre-trained model

VGG-16 is my favorite image classification model to run 
because of its simplicity and accuracy. The creators of this model 
published a pre-trained binary that can be used in Caffe.

https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

MD5 (VGG_ILSVRC_16_layers.caffemodel) = 441315b0ff6932dbfde97731be7ca852

This is to convert that specific file to a TensorFlow model and check
its correctness.

Run `make` to download the original caffe model and convert it.
`tf_forward.py` has an example of how to use the generated `vgg16.tfmodel`
file.

If you don't feel like installing caffe, you can download the output here 
https://github.com/ry/tensorflow-vgg16/raw/master/vgg16-20160129.tfmodel.torrent

The input ("images") to the TF model is expected to be [batch, height, width, channel]
where height = width = 224 and channel = 3. Values should be between 0 and 1.

The output ("prob") is a 1000 dimensional class probabiity vector whose indexes
correspond to line numbers in syset.txt.
