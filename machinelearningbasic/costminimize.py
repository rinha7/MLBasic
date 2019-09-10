import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W*X-Y) * X)
descent = W - learning_rate * gradient
# optimize의 gradient 를 사용하여 대체할 수 있음. 그 쪽이 좀 더 깔끔함
# 단지, 본인이 radient를 잘라서 써야할 때는 구현하여 사용한다.

# assign은 변수값의 초기화를 뜻함. variable을 통한 초기화
update = W.assign(descent)


sess = tf.Session()

sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update, feed_dict = {X:x_data, Y:y_data})
    print(step, sess.run(cost,feed_dict={X:x_data, Y:y_data}), sess.run(W))