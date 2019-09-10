# linear regression의 간단한 구

import tensorflow as tf

X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# H(x) = Wx + b로 표현할 수 있음
hypothesis = X * W + b


cost = tf.reduce_mean(tf.square(hypothesis - Y)) # reduce_mean을 통해 평균 구함

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)


train = optimizer.minimize(cost) # cost를 miniize함.

# 실행을 위한 session 작성 필요
sess = tf.Session()
sess.run(tf.global_variables_initializer())



for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(cost), sess.run(W), sess.run(b))