import functools
import tensorflow as tf

def scope(function):
	attr = '_cache_' + function.__name__

	@property
	@functools.wraps(function)
	def decorator(self):
		if not hasattr(self,attr):
			with tf.variable_scope(function.__name)		