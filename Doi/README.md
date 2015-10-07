# Team Doi
CNN-LSTM based Japanese Sign recognition Neural Network System written in chainer.  

## Team Members
- Yurika Doi(University of Tokyo)
- Takuma Yagi(Tokyo Institute of Technology)
- Tomohito Minakuchi(Keio University)

## Quick use

(1) Install dependencies  
(2) Download learning data  
(3) clone repository  
(4-1) (Using CPU) python train\_new.py --lr 0.001 --datadir /where/you/placed/the/learning/data/  
(4-2) (Using GPU) python verificate.py --gpu 0 --device\_num 0 --lr 0.001 --datadir /where/you/placed/the/learning/data/  
(5) python verificate.py --datadir /where/you/placed/the/learning/data/ --model /trained/chainer/model  

## Dependencies

* Python2.7
* numpy (1.9 >=)
* chainer 1.3.1
* progressbar2
* OpenCV(2 or 3 with XVID codec support)
* argparse
* logging
* CUDA(6.5 >= recommended)
* matplotlib

## Core functionality

* traini\_new.py: training and testing
* verificate.py: sign prediction using trained model

## Sample Dataset

This recognition system uses GrayScale movie(.avi) and Depth movie(.xml, OpenCV matrix) for source input.  
Sample Data is available(1 person, 3 words, 4episodes/word)  

https://drive.google.com/file/d/0B9sgleO5gCDyNUFtY0ZCREpmaUk/view?usp=sharing  

