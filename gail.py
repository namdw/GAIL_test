# Generative Adversarial Imitation Learning
# Specifically for Naver CLAIR
# Daniel Nam dnjsxodp@gmail.com

import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 


learning_rate = 1e-3

class Dnet:
	def __init__(self, sessions, input_size, output_size, name):
		self.session = session
		self.input_size = input_size
		self.output_size = output_size
		self.name = "D"

		self._build_net()

	def _build_net(self, h_size=100, lr=learning_rate):
		with tf.variable_scope(self.name):
			self._traj = tf.placeholder(tf.float32, [None, self.input_size])
			self._trajE = tf.placeholder(tf.float32, [None, self.input_size])
			W1 = tf.get_variable("W1", shape=[input_size, self.h_size], initializer=tf.contrib.layers.xavier_initializer())
			layer1 = tf.nn.tanh(tf.matmul(self._traj, W1))
			layer1E = tf.nn.tanh(tf.matmul(self._trajE, W1))
			W2 = tf.get_variable("W2", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
			self._D = tf.matmul(layer1, W2)
			self._DE = tf.matmul(layer1E, W2)
			self._loss = -tf.reduce_mean(tf.log(self._D)+tf.log(1-self._DE))
			self._train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss)

class PInet:
	def __init__(self, sessions, input_size, output_size, name):
		self.session = session
		self.input_size = input_size
		self.output_size = output_size
		self.name = "pi"

		self._build_net()

	def _build_net(self, h_size=100, lr=learning_rate):
		with tf.variable_scope(self.name):
			self._input = tf.placeholder(tf.float32, [None, self.input_size])
			W1 = tf.get_variable("W1", shape=[input_size, self.h_size], initializer=tf.contrib.layers.xavier_initializer())
			layer1 = tf.nn.tanh(tf.matmul(self._Z, W1))
			W2 = tf.get_variable("W2", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
			self._output = tf.matmul(layer1, W2)