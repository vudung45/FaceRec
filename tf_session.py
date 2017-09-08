'''
Load pretrain models and create a tensorflow session to run them

@Author: David Vu
'''
import tensorflow as tf


class FaceRecSession(object):
    def __init__(self):
        '''
        :param recog_model: directory to the saved variables of the recognition pretrain model
        :param enable_mtcnn: True to load mtcnn facial detection model
        '''
        self.session_graph = tf.Graph();
