# Generative Adversarial Imitation Learning
# Specifically for Naver CLAIR
# Daniel Nam dnjsxodp@gmail.com

import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 

import gym

# env = gym.make('Cartpole-v0')
learning_rate = 1e-2

class Dnet:
	def __init__(self, session, input_size, output_size, name):
		self.session = session
		self.input_size = input_size
		self.output_size = output_size
		self.name = name

		self._build_net()

	def _build_net(self, h1_size=100, h2_size=100, lr=learning_rate):
		with tf.variable_scope(self.name):
			self._traj = tf.placeholder(tf.float32, [None, self.input_size])
			self._trajE = tf.placeholder(tf.float32, [None, self.input_size])
			W1 = tf.get_variable("W1", shape=[self.input_size, h1_size], initializer=tf.contrib.layers.xavier_initializer())
			b1 = tf.get_variable("b1", [h1_size], initializer=tf.constant_initializer(0.0))
			layer1 = tf.nn.relu(tf.matmul(self._traj, W1) + b1)
			layer1E = tf.nn.relu(tf.matmul(self._trajE, W1) + b1)
			W2 = tf.get_variable("W2", shape=[h1_size, h2_size], initializer=tf.contrib.layers.xavier_initializer())
			b2 = tf.get_variable("b2", shape=[h2_size], initializer=tf.constant_initializer(0.0))
			layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
			layer2E = tf.nn.relu(tf.matmul(layer1E, W2) + b2)
			W3 = tf.get_variable("W3", shape=[h2_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
			b3 = tf.get_variable("b3", shape=[self.output_size], initializer=tf.constant_initializer(0.0))
			self._D = tf.nn.sigmoid(tf.matmul(layer2, W3) + b3)
			self._DE = tf.nn.sigmoid(tf.matmul(layer2E, W3) + b3)
		
		self._lossD = -tf.reduce_mean(tf.log(self._D)+tf.log(1-self._DE))
		self._trainD = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._lossD)

	def predict(self, state, action, expt):
		traj = np.reshape(np.append(state, action), [1, self.input_size])
		if expt:
			return self.session.run(self._DE, feed_dict={self._trajE: traj})
		else:
			return self.session.run(self._D, feed_dict={self._traj: traj})

	def update(self, state, action, stateE, actionE):
		traj = np.reshape(np.append(state, action), [1, self.input_size])
		trajE = np.reshape(np.append(stateE, actionE), [1, self.input_size])
		return self.session.run([self._lossD, self._trainD], feed_dict={self._traj:traj, self._trajE:trajE})

class PInet:
	def __init__(self, session, input_size, output_size, name):
		self.session = session
		self.input_size = input_size
		self.output_size = output_size
		self.name = name

		self._build_net()

	def _build_net(self, h1_size=100, h2_size=100, lr=learning_rate, lamb=0.1):
		with tf.variable_scope(self.name):
			self._state = tf.placeholder(tf.float32, [None, self.input_size])
			self._action = tf.placeholder(tf.float32, [None, self.output_size])
			W1 = tf.get_variable("W1", shape=[self.input_size, h1_size], initializer=tf.contrib.layers.xavier_initializer())
			b1 = tf.get_variable("b1", shape=[h1_size], initializer=tf.constant_initializer(0.0))
			layer1 = tf.nn.relu(tf.matmul(self._state, W1) + b1)
			W2 = tf.get_variable("W2", shape=[h1_size, h2_size], initializer=tf.contrib.layers.xavier_initializer())
			b2 = tf.get_variable("b2", shape=[h2_size], initializer=tf.constant_initializer(0.0))
			layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
			W3 = tf.get_variable("W3", shape=[h2_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
			b3 = tf.get_variable("b3", shape=[self.output_size], initializer=tf.constant_initializer(0.0))
			self._PI = tf.nn.softmax(tf.matmul(layer2, W3) + b3)
			self._D = tf.placeholder(tf.float32, [None, 1])

		self._pa = tf.reduce_max(tf.multiply(self._PI, self._action))
		self._lossPI = -tf.reduce_mean(tf.log(self._pa)*tf.log(self._D)-lamb*tf.log(self._pa))
		self._trainPI = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self._lossPI)

	def predict(self, state, action_mat):
		x = np.reshape(state, [1, self.input_size])
		return self.session.run(self._pa, feed_dict={self._state: x, self._action:action_mat})
		

	def update(self, state, action_mat, D):
		x = np.reshape(state, [1, self.input_size])
		return self.session.run([self._lossPI, self._trainPI], feed_dict={self._state:x, self._action:action_mat, self._D:D})


def main():
	env = gym.make('CartPole-v0')
	input_size_D = env.observation_space.shape[0] + env.action_space.n
	output_size_D = 1
	input_size_P = env.observation_space.shape[0]
	output_size_P = env.action_space.n
	with tf.Session() as sess:
		discriminator = Dnet(sess, input_size_D, output_size_D, "D")
		policy = PInet(sess, input_size_P, output_size_P, "policy")
		tf.global_variables_initializer().run()

if __name__ == "__main__":
	main()