import tensorflow as tf
from sentiment_analysis_deep_net import create_feature_sets_and_labels
import numpy as np
train_x,train_y,test_x,test_y=create_feature_sets_and_labels('pos.txt','neg.txt')

n_nodes_h1=500
n_nodes_h2=500
n_nodes_h3=500

n_classes=2
batch_size=100

x=tf.placeholder('float',[None,len(train_x[0])])
y=tf.placeholder('float')

def neural_network_model (data):

    hidden_1_layer= {'weigths':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_h1])),
                     'biases':tf.Variable(tf.random_normal([n_nodes_h1]))}

    hidden_2_layer= {'weigths':tf.Variable(tf.random_normal([n_nodes_h1,n_nodes_h2])),
                     'biases':tf.Variable(tf.random_normal([n_nodes_h2]))}

    hidden_3_layer= {'weigths':tf.Variable(tf.random_normal([n_nodes_h2,n_nodes_h3])),
                     'biases':tf.Variable(tf.random_normal([n_nodes_h3]))}

    output_layer= {'weigths':tf.Variable(tf.random_normal([n_nodes_h3,n_classes])),
                     'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1= tf.add(tf.matmul(data,hidden_1_layer['weigths']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2= tf.add(tf.matmul(l1,hidden_2_layer['weigths']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3= tf.add(tf.matmul(l2,hidden_3_layer['weigths']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output= tf.matmul(l3,output_layer['weigths']+ output_layer['biases'])
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimiser= tf.train.AdamOptimizer().minimize(cost)

    hm_epoch= 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epoch):
            epoch_loss=0
            i=0
            while i<len(train_x):
                start=i
                end = i+ batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                i += batch_size
                _, c= sess.run([optimiser,cost],feed_dict={x: batch_x , y: batch_y})
                epoch_loss+=c
            print('Epoch ',epoch+1, 'completed out of ',hm_epoch, 'epoch_loss:', epoch_loss)

        correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy= tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy:', accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)