#download dataset
sh download.sh

this command is gonnam make 'cifar-10-batches-py' directory which contains 5 data batches, readme.html, and so on.

#Pickle datasets
python dataset.py

this command is gonna make 'data' directory which contains 'test_data.npy', 'test_labels.npy', 'train_data.py', 'train_labels.py'

#Run Developmental Learning
python developmental_train.py

you will see 'result' directory. which has 

#Draw Graphs
python draw_loss_crvs.pt --logfile --outdir

in order to compare with normal way of training, i made normal_train.py.



