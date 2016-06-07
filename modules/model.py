import tensorflow as tf
import math

# Define Variables

NUMBER_OF_CLASSES=10
INPUT_DIMENSION=64
CHANNELS=3
number_of_features = INPUT_DIMENSION*INPUT_DIMENSION*CHANNELS

class Model:

   # def __init__(self):
   #    self.tf_x = None
   #    self.tf_y1 = None
   #    self.tf_y2 = None
   #    self.tf_y3 = None
   #    
   #    self.tf_length = None
   #   
   #    self.keep_prob= None

   #    self.digit1_accuracy =None
   #    self.digit2_accuracy =None
   #    self.digit3_accuracy =None
   #    self.optimiser = None

    def getGraph(self):
        g = tf.Graph()
        with g.as_default():
            
            # tf Graph input
            self.tf_x = tf.placeholder(tf.float32, [None, number_of_features])
            
            
            self.tf_y1 = tf.placeholder(tf.float32, [None, NUMBER_OF_CLASSES])
            
            #adding a class for "blank"
            self.tf_y2 = tf.placeholder(tf.float32, [None, NUMBER_OF_CLASSES+1])
            self.tf_y3 = tf.placeholder(tf.float32, [None, NUMBER_OF_CLASSES+1])
            
            self.tf_length = tf.placeholder('uint8')
            
            self.keep_prob= tf.placeholder('float')
            
            op_algorithm = tf.train.AdamOptimizer
            starting_learning_rate = 0.0001
            
            
            
            def conv2d(name, l_input, w, b):
                return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], 
                                                              padding='SAME'),b), name=name)
            
            def max_pool(name, l_input, k):
                return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], 
                                      padding='SAME', name=name)
            
            def convolution_layers(in_X, _weights, _biases, _dropout, dim):
            
                # Reshape input picture
                _X = tf.reshape(in_X, shape=[-1, dim, dim, CHANNELS])
            
                # Convolution Layer
                conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
                # Max Pooling (down-sampling)
                pool1 = max_pool('pool1', conv1, k=2)
                # Apply Dropout
                dropout1 = tf.nn.dropout(pool1, _dropout)
                    
                # Convolution Layer
                conv2 = conv2d('conv2', dropout1, _weights['wc2'], _biases['bc2'])
                # Max Pooling (down-sampling)
                pool2 = max_pool('pool2', conv2, k=1)
                # Apply Dropout
                dropout2 = tf.nn.dropout(pool2, _dropout)
                
                #For debugging
                if False:
                    reshape = tf.reshape(dropout2, [in_X.shape[0], -1])
                    dim = reshape.get_shape()[1].value
                    print dim
            
                return dropout2
            
            def classifier_FC_d1(last_conv_output, _weights, _biases, _dropout):
            
                # Fully connected layer
                # Reshape last_conv_output output to fit dense layer input
                dense1 = tf.reshape(last_conv_output, [-1, _weights['d1_wd1'].get_shape().as_list()[0]]) 
                # Relu activation
                dense1 = tf.nn.relu(tf.matmul(dense1, _weights['d1_wd1']) + _biases['d1_bd1'])
                  
                # Relu activation
                dense2 = tf.nn.relu(tf.matmul(dense1, _weights['d1_wd2']) + _biases['d1_bd2']) 
            
                return dense2
            
            
            def classifier_FC_d2(last_conv_output, _weights, _biases, _dropout):
            
                # Fully connected layer
                # Reshape last_conv_output output to fit dense layer input
                dense1 = tf.reshape(last_conv_output, [-1, _weights['d2_wd1'].get_shape().as_list()[0]]) 
                # Relu activation
                dense1 = tf.nn.relu(tf.matmul(dense1, _weights['d2_wd1']) + _biases['d2_bd1'])
                  
                # Relu activation
                dense2 = tf.nn.relu(tf.matmul(dense1, _weights['d2_wd2']) + _biases['d2_bd2']) 
            
                return dense2
            
            
            def classifier_FC_d3(last_conv_output, _weights, _biases, _dropout):
            
                # Fully connected layer
                # Reshape last_conv_output output to fit dense layer input
                dense1 = tf.reshape(last_conv_output, [-1, _weights['d3_wd1'].get_shape().as_list()[0]]) 
                # Relu activation
                dense1 = tf.nn.relu(tf.matmul(dense1, _weights['d3_wd1']) + _biases['d3_bd1'])
                  
                # Relu activation
                dense2 = tf.nn.relu(tf.matmul(dense1, _weights['d3_wd2']) + _biases['d3_bd2']) 
            
                return dense2
                
            def get_sequence_length_logits(last_fc, _weights, _biases):
                # Output, class prediction
                out = tf.matmul(last_fc, _weights['length_out']) + _biases['length_out']
                return out
            
            
            def get_digit1_logits(last_fc, _weights, _biases):
                # Output, class prediction
                out = tf.matmul(last_fc, _weights['digit1_out']) + _biases['digit1_out']
                return out
            
            
            def get_digit2_logits(last_fc, _weights, _biases):
                # Output, class prediction
                out = tf.matmul(last_fc, _weights['digit2_out']) + _biases['digit2_out']
                return out
            
            
            def get_digit3_logits(last_fc, _weights, _biases):
                # Output, class prediction
                out = tf.matmul(last_fc, _weights['digit3_out']) + _biases['digit3_out']
                return out
            
            
            
            def initialise_param(shape, dtype=tf.float32):
                return tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(shape[0])))
            
            # Store layers weight & bias
            weights = {
                'wc1': tf.Variable(initialise_param([3, 3, CHANNELS, INPUT_DIMENSION])),
                'wc2': tf.Variable(initialise_param([3, 3, INPUT_DIMENSION, 128])),
                
                'd1_wd1': tf.Variable(initialise_param([32*32*128, 256])),    
                'd1_wd2': tf.Variable(initialise_param([256, 256])),
                'd2_wd1': tf.Variable(initialise_param([32*32*128, 256])),    
                'd2_wd2': tf.Variable(initialise_param([256, 256])),
                'd3_wd1': tf.Variable(initialise_param([32*32*128, 256])),    
                'd3_wd2': tf.Variable(initialise_param([256, 256])),
                
                'digit1_out': tf.Variable(initialise_param([256, NUMBER_OF_CLASSES])),
                'digit2_out': tf.Variable(initialise_param([256, NUMBER_OF_CLASSES + 1])),
                'digit3_out': tf.Variable(initialise_param([256, NUMBER_OF_CLASSES + 1])),  
                
                #'regr_out' :tf.Variable(initialise_param([256, bb_target_dimensions]))
            }
            biases = {
                'bc1': tf.Variable(initialise_param([INPUT_DIMENSION])),
                'bc2': tf.Variable(initialise_param([128])),
                
                'd1_bd1': tf.Variable(initialise_param([256])),
                'd1_bd2': tf.Variable(initialise_param([256])),
                
                'd2_bd1': tf.Variable(initialise_param([256])),
                'd2_bd2': tf.Variable(initialise_param([256])),
                
                'd3_bd1': tf.Variable(initialise_param([256])),
                'd3_bd2': tf.Variable(initialise_param([256])),
                
                'digit1_out': tf.Variable(initialise_param([NUMBER_OF_CLASSES])),
                'digit2_out': tf.Variable(initialise_param([NUMBER_OF_CLASSES + 1])),
                'digit3_out': tf.Variable(initialise_param([NUMBER_OF_CLASSES + 1])),
            
                #'regr_out' :tf.Variable(initialise_param([bb_target_dimensions]))
            }
    
            
            last_conv_layer =  convolution_layers(self.tf_x, weights, biases, self.keep_prob, INPUT_DIMENSION)
            last_fc_layer_d1 = classifier_FC_d1(last_conv_layer, weights, biases, self.keep_prob)
            last_fc_layer_d2 = classifier_FC_d2(last_conv_layer, weights, biases, self.keep_prob)
            last_fc_layer_d3 = classifier_FC_d3(last_conv_layer, weights, biases, self.keep_prob)
            
            
            digit1_logits = get_digit1_logits(last_fc_layer_d1, weights, biases)
            digit2_logits = get_digit2_logits(last_fc_layer_d2, weights, biases)
            digit3_logits = get_digit3_logits(last_fc_layer_d3, weights, biases)
            
            digit1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(digit1_logits, self.tf_y1))
            digit2_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(digit2_logits, self.tf_y2))
            digit3_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(digit3_logits, self.tf_y3))
            
            overall_loss = digit1_loss + digit2_loss + digit3_loss
            
            self.optimiser = op_algorithm(starting_learning_rate).minimize( overall_loss) 
            
            
            correct_digit1_prediction = tf.equal(tf.argmax(digit1_logits,1) , tf.argmax(self.tf_y1,1))
            self.digit1_accuracy = tf.reduce_mean(tf.cast(correct_digit1_prediction, "float"))
            
            correct_digit2_prediction = tf.equal(tf.argmax(digit2_logits,1) , tf.argmax(self.tf_y2,1))
            self.digit2_accuracy = tf.reduce_mean(tf.cast(correct_digit2_prediction, "float"))
            
            correct_digit3_prediction = tf.equal(tf.argmax(digit3_logits,1) , tf.argmax(self.tf_y3,1))
            self.digit3_accuracy = tf.reduce_mean(tf.cast(correct_digit3_prediction, "float"))
    
        return g
    
    def init_interactive_session(self,graph):
        return tf.InteractiveSession(graph=graph)
    
    def load_saved_model(self,path_to_saved_model, session):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(path_to_saved_model)
        saver.restore(session, ckpt.model_checkpoint_path)
    
    def run_training_epoch(self,session, x, y1, y2, y3, keep_prob):
        l = -1
        feed_dict = { self.tf_length: l, self.tf_x: x, self.tf_y1: y1, self.tf_y2: y2, self.tf_y3: y3, self.keep_prob:keep_prob}
            
        session.run(self.optimiser,feed_dict=feed_dict)
    

    def test_model(self,session, x, y1, y2, y3):
        '''Returns the accuracy of digit1, digit2, and digit3'''
        feed_dict = {self.tf_x: x, self.keep_prob:1.0, self.tf_y1: y1, self.tf_y2: y2, self.tf_y3: y3, self.keep_prob:1.0}
        d1, d2, d3 = session.run([self.digit1_accuracy, self.digit2_accuracy, self.digit3_accuracy], \
                        feed_dict=feed_dict )
    
        #print "Test : d1 acc=%0.3f d2 acc=%0.3f d3 acc=%0.3f" % (d1, d2, d3 )
        return d1, d2, d3
