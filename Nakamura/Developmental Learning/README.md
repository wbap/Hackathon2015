#Purpose
I wanted to simulate developmental way of learning with deep learning algorithm.
In this case, I made a CNN(based on VGG-mini) which does an image clastering task with adding layer step by step. I took the Cifar-10 for a database to use. An important point is, attaching new layer to input layer. My purpose was, finding a result which is given by an incremental development of learning accuracy.
In order to make comparison, I made another CNN which just outputs result of training for each numbers of layers.

results are as follows.
![dev-test-acc]("https://raw.github.com/wiki/john5a18/Hackathon2015/developmental_result/log-test_acc.png")
![タイトル](リンクURL)
![タイトル](リンクURL)
![タイトル](リンクURL)

There are two points. (1)the difference of curvature. (2)with epoch goes, accuracy and loss of developmental way of CNN training hit a roof.
(1) is obviously caused by the way of learning. it seems that developmental way of learning can get good acuracy with less times of epoch. 
(2) is caused by the first step of learning with minimum layers. In my case, weights are fixed in order from output layer, so output accuracy and loss are limited by the first training.




#How To Train
#Download dataset
sh download.sh

this command is gonnam make 'cifar-10-batches-py' directory which contains 5 data batches, readme.html, and so on.

#Pickle datasets
python dataset.py

this command is gonna make 'data' directory which contains 'test_data.npy', 'test_labels.npy', 'train_data.py', 'train_labels.py'

#Run Developmental Learning
python developmental_train.py

you will see 'result' directory. which has directories starts with 'Developmental_train_' and ends with the date, like 'Developmental_train_2015-10-25_12-09-54_144574259471'.
you can see log.txt in the directory. but it may be dificult to lead those boring lines.

#Draw Graphs
python draw_loss_crvs.pt --logfile --outdir

you can write graphs with this command. --logfile is the log.txt which you made before. you need to specify --outdir also.
you will see four graphs with this command. 'log_test_acc.png','log_test_loss.png','log_train_acc.png','log_train_loss.png'


#Comparison
in order to compare with normal way of training, i made normal_train.py.
you can run this command as developmental_train.py. you will see a directory which starts with 'Normal_train_' in the 'results' directory.
you can also use graph drawing command.


#Requirements
chainer==1.3.2
filelock==2.0.4
funcsigs==0.4
matplotlib==1.4.3
mock==1.3.0
nose==1.3.7
numpy==1.10.1
pbr==1.8.1
protobuf==2.6.1
pyparsing==2.0.3
python-dateutil==2.4.2
pytz==2015.6
six==1.10.0
wheel==0.26.0


