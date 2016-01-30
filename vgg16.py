import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

class Model():
    def get_conv_filter(self, name):
        raise NotImplementedError

    def get_bias(self, name):
        raise NotImplementedError

    def get_fc_weight(self, name):
        raise NotImplementedError

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                 dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def build(self, rgb, train=False):
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

        self.relu1_1 = self._conv_layer(bgr, "conv1_1")
        self.relu1_2 = self._conv_layer(self.relu1_1, "conv1_2")
        self.pool1 = self._max_pool(self.relu1_2, 'pool1')

        self.relu2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.relu2_2 = self._conv_layer(self.relu2_1, "conv2_2")
        self.pool2 = self._max_pool(self.relu2_2, 'pool2')

        self.relu3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.relu3_2 = self._conv_layer(self.relu3_1, "conv3_2")
        self.relu3_3 = self._conv_layer(self.relu3_2, "conv3_3")
        self.pool3 = self._max_pool(self.relu3_3, 'pool3')

        self.relu4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.relu4_2 = self._conv_layer(self.relu4_1, "conv4_2")
        self.relu4_3 = self._conv_layer(self.relu4_2, "conv4_3")
        self.pool4 = self._max_pool(self.relu4_3, 'pool4')

        self.relu5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.relu5_2 = self._conv_layer(self.relu5_1, "conv5_2")
        self.relu5_3 = self._conv_layer(self.relu5_2, "conv5_3")
        self.pool5 = self._max_pool(self.relu5_3, 'pool5')

        self.fc6 = self._fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]

        self.relu6 = tf.nn.relu(self.fc6)
        if train:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self._fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        self.fc8 = self._fc_layer(self.relu7, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

