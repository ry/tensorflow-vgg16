vgg16.tfmodel: VGG_ILSVRC_16_layers.caffemodel
	python caffe_to_tensorflow.py

VGG_ILSVRC_16_layers.caffemodel:
	curl -O http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

clean:
	rm -f vgg16.tfmodel


