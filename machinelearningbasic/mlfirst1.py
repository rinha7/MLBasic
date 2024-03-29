# linear regression의 간단한 구

import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# H(x) = Wx + b로 표현할 수 있음
hypothesis = X * W + b


cost = tf.reduce_mean(tf.square(hypothesis - Y)) # reduce_mean을 통해 평균 구함

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 실행을 위한 session 작성 필요

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val , b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
# placeholder?

print(sess.run(hypothesis, feed_dict={X:[55]})) # X가 5일때 Y의 값은 얼마일까? 를 예측함
print(sess.run(hypothesis, feed_dict={X:[2.5]}))