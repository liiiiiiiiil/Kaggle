import pandas as pd 
import tensorflow as tf
import numpy as np
from  sklearn.preprocessing import LabelEncoder

class Dataset:

    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass
    
    @property
    def data(self):
        return self._data
    
    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)# get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
    
        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
    
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]


mushrooms=pd.read_csv("./mushroom.csv")
data=mushrooms
labelencoder=LabelEncoder()
for col in data.columns:
    data[col]=labelencoder.fit_transform(data[col])

data=data.drop('stalk-root',1)
data=data.values
y_temp=np.zeros([data.shape[0],2])
idx=(data[:,0]==0)
y_temp[idx,0]=1
idx=(data[:,0]==1)
y_temp[idx,1]=1

data=np.delete(data,0,1)
data=np.concatenate((y_temp,data),axis=1)

data_train=data[:7500]
data_test=data[7500:]

learning_rate=0.01
training_epochs=50
batch_size=500

keep_prob=tf.placeholder(tf.float32)
X=tf.placeholder(tf.float32,[None,21])
Y=tf.placeholder(tf.float32,[None,2])

W1=tf.Variable(tf.random_normal([21,100],stddev=0.01))
b1=tf.Variable(tf.constant(value=0.,dtype=tf.float32,shape=[100]))
L1=tf.matmul(X,W1)+b1
L1=tf.nn.relu(L1)
L1=tf.nn.dropout(L1,keep_prob=keep_prob)

W2=tf.Variable(tf.random_normal([100,2],stddev=0.01))
b2=tf.Variable(tf.constant(0.,dtype=tf.float32,shape=[2]))
logits=tf.matmul(L1,W2)+b2

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
dataset_train=Dataset(data_train)
correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(data_train.shape[0]/batch_size)
    for i in range(total_batch):
        data_batch=dataset_train.next_batch(batch_size)
        batch_x=data_batch[:,2:]
        batch_y=data_batch[:,:2]
        feed_dict={X:batch_x,Y:batch_y,keep_prob:0.4}
        c,_=sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost+=c/total_batch
    print('eopch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost))

print('learning_finished')

train_X=data_train[:,2:]
train_y=data_train[:,:2]
print("training accuracy:",sess.run(accuracy,feed_dict={X:train_X,Y:train_y,keep_prob:1}))
test_X=data_test[:,2:]
test_y=data_test[:,:2]
print("accuracy:",sess.run(accuracy,feed_dict={X:test_X,Y:test_y,keep_prob:1}))
    














