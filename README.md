# TensorFlow VGG-16 pre-trained model

VGG-16 is my favorite image classification model to run 
because of its simplicity. The creators of this model 
published a pre-trained binary that can be used in Caffe.

https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

This project is to convert it to TensorFlow and check its
correctness.

Just run `make` to download the original caffe model and convert it.
`tf_forward.py` has an example of how to use the generated `vgg16.tfmodel`
file.

I modified the prototxt file slightly for debugging purposes.
(Renaming a couple of the blobs.)

Obviously it would be great to have a general purpose caffemodel
-> TensorFlow graph program. This is only a first step in that
direction. I'm not sure I need any other pre-trained models, so I
might not generalize it-feel free to use any of this code if you
do that work.
