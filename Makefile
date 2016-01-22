vgg16.tfmodel: VGG_ILSVRC_16_layers.caffemodel
	python caffe_to_tensorflow.py

VGG_ILSVRC_16_layers.caffemodel:
	curl -O http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

upload:
	aws s3 cp vgg16-20160121.tfmodel s3://tinyclouds-storage/  --acl public-read

clean:
	rm -f vgg16.tfmodel


