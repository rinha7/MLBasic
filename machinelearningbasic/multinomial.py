# 여러개의 class가 있을 때 그것을 예측하는 것
# Softmax Classifiery
# 0~1 사이의 확률로 표현해주는 softmax function을 이용한다. ( 모든 레이블을 더하면 1이 됨 )

import tensorflow as tf

#data set
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]

# y_data 는 one-hot 인코딩 방식으로 표현된 것
# one-hot 인코딩이란 하나만 값이 다른 것을 의미
y_data = [[0,0,1], #2 를 의미
          [0,0,1],
          [0,0,1],
          [0,1,0], # 1을 의미
          [0,1,0],
          [0,1,0],
          [1,0,0], # 0을 의미
          [1,0,0]]
X = tf.placeholder("float",[None,4])
Y = tf.placeholder("float",[None,3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

# tensorflow로 구현하는 것은 ㄱ나단함
# tf.matmul(X,W)+bias 를 하면 구현됨
# 여기서는 내장함수를 이용하여 구현한다
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0 :
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    print("after learning")
    a = sess.run(hypothesis, feed_dict={X:[[1,11,7,9],
                                           [1,3,4,3],
                                           [1,1,0,1]]})
    print(a,sess.run(tf.math.argmax(a,1)))