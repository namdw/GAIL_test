# Generative Adversarial Imitation Learning
# Specifically for Naver CLAIR
# Daniel Nam dnjsxodp@gmail.com

import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 


learning_rate = 1e-3

class DGnet:
	def __init__(self, sessions, input_size, output_size, name):
		self.session = session
		self.input_size = input_size
		self.output_size = output_size
		self.name = name

		self._build_net()

	def _build_net(self, h_size=100, lr=learning_rate):
		with tf.variable_scope(self.name):
			self._input = tf.placeholder(tf.float32, [None, self.input_size])
			W1 = tf.get_variable("W1", shape=[input_size, self.h_size], initializer=tf.contrib.layers.xavier_initializer())
			layer1 = tf.nn.tanh(tf.matmul(self._Z, W1))
			W2 = tf.get_variable("W2", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
			self._output = tf.matmul(layer1, W2)