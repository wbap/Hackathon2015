##The First WBAI Hackathon (2015-09) Team Report

#The Title of the Project:
Comparison of imitation learning using video games

#Members: name, affiliate; name, affiliate; …
・Keigo Nishida(Osaka University)
・Ryodai Tamura(Doshisha University)
・Takaya Tamaki (Doshisha University)
・Takuya Miura(Osaka University)

#Abstract:
Imitation of another person who have special ability enable us to get high skills effectively. 
Imitation learning is to train prediction of expert’s behavior. 
In this hack , we try to implement imitation learning using Deep Q-network and LSTM. This architecture is inspired by songbird neural network model.

#Goal:
We confirm whether the early learning of video game is carried out to combine Deep Q-learning and LSTM

#Method:
1.	Acquiring DQN parameters which were learned enough , you take an expert movie used training data.
2.	LSTM is learned by training data which is output by CNN layer of DQN
3.	Initializing full connected network of DQN, you execute reinforcement learning which coupled DQN with LSTM
#Result:
Judging from comparison between DQN and DQN-LSTM learning, we confirmed imitation learning succeed.
But we couldn’t confirm difference of LSTM learning number of times.

#Dependencies
•	Python 2.7+
•	Numpy
•	Scipy
•	Chainer (1.3.0): https://github.com/pfnet/chainer
•	RL-glue core: https://sites.google.com/a/rl-community.org/rl-glue/Home/rl-glue
•	RL-glue Python codec: https://sites.google.com/a/rl-community.org/rl-glue/Home/Extensions/python-codec
•	Arcade Learning Environment (version ALE 0.4.4): http://www.arcadelearningenvironment.org/


#Reference:
V. Mnih et al., “Human-level control through deep reinforcement learning”
Doya K et al.,  “A computational model of birdsong learning by auditory experience and auditory feedback” 
Ila R. Fiete et al., “Model of Birdsong Learning Based on Gradient Estimation by Dynamic Perturbation of Neural Conductances”
小島 哲 “小鳥のさえずり学習の神経機構：大脳基底核経路と強化学習モデル”
和多 和宏 “小鳥がさえずるとき脳内では何が起こっている？”
DQNの生い立ち　＋　Deep Q-NetworkをChainerで書いたhttp://qiita.com/Ugo-ama/items/08c6a5f6a571335972d5