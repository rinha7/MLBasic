import tensorflow as tf


x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]] # 현재 갖고 있는 y의 데이터

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

# hypothesis를 표현하기 위해 sigmoid를 사용함
# sigmoid 는 tf.div(1.,1. + tf.exp(tf.matmul(X,W) + b))와 같다고 보면 된다.

cost = -tf.reduce_mean(Y *tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# 정확도를 계산합니다
# hypothesis가 0.5보다 클때만 True가 되도록 정확도를 계산
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, cost_val)


    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis : ",h,"\n correct(Y) : ",c,"\n Accuracy : ",a)

# 결과를 보면 다음과 같이 나온다
'''

# Hypothesis는 수식을 입력한 후에 나온 값들
# 여기서 0.5 보다 큰 경우에만 1이 된다.
Hypothesis :  [[0.03592602]
 [0.16533053]
 [0.32879117]
 [0.7706564 ]
 [0.9326154 ]
 [0.97786194]] 
 
 # 예측과 y의 값을 비교한다.
 correct(Y) :  [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]] 
 
 # 모두 맞았으므로 정확도는 1이 나옴.
 Accuracy :  1.0
'''