# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

'''
(101, 16) (101, 1)
'''

nb_classes = 7  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
# tf.one_hot과 reshpae는 우리가 사용하는 데이터가 원하는 모양이 아닐때 원하는 모양으로 조정해줌.
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:", Y_one_hot)

'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32)
'''

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# learning rate 은 크면 값을 못찾을 수 있고 작으면 찾기 힘들어진다
# standardization?
# Overfitting

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot,1))
# 정확도는 우리의 예측과 원래 결과를 비교해서 낸다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 예측과 정답과의 차이를 평균을 내서 정확도를 체

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            loss, acc =  sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data})

            print("Step: {:5}\tLoss:{:.3f}\tAcc:{:.2%}".format(step, loss, acc))

    pred = sess.run(prediction, feed_dict={X:x_data})


    # flatten은 [[1],[0]]처럼 생긴 배열을 [1,0]과 같은 형태로 바꾸어주는 역할을 수행한다.
    # zip 은 묶는것, 이거는 나중에 해보자
    for p, y in zip(pred, y_data.flatten()):
        # 예측한 값과 일치하는지 predict와 원래 y값을 비교, 일치하면 True를 출-력
        print("[{}] Prediction : {} True Y: {}".format(p == int(y), p , int(y)))