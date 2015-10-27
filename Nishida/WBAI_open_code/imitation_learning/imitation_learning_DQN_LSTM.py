# -*- coding: utf-8 -*-
"""
Deep Q-network implementation with chainer and rlglue
Copyright (c) 2015  Naoto Yoshida All Right Reserved.
"""

import copy

import pickle
import numpy as np
import scipy.misc as spm

from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action

from lstm import LSTM, make_initial_state

lstm_loss = 0.00053
class LSTM_class:
    #lstm_loss = 0
    def __init__(self):
        self.model_lstm = LSTM(3136, 1045)
        #you should select path of saving parameters
        d_name = 'lstm_params/1epoch_params/'
        self.model_lstm.l1_x.W = np.load(d_name+'l1_x_W.npy')
        self.model_lstm.l1_x.b = np.load(d_name+'l1_x_b.npy')
        self.model_lstm.l1_h.W = np.load(d_name+'l1_h_W.npy')
        self.model_lstm.l1_h.b = np.load(d_name+'l1_h_b.npy')
        self.model_lstm.l6.W = np.load(d_name+'l6_W.npy')
        self.model_lstm.l6.b = np.load(d_name+'l6_b.npy')
        cuda.get_device(0).use()
        self.model_lstm=self.model_lstm.to_gpu()

class DQN_class:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 100#10**4  # Initial exploratoin. original: 5x10^4
    replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**4  # Target update frequancy. original: 10^4
    data_size = 10**5 #10**5  # Data size of history. original: 10^6

    def __init__(self, enable_controller=[0, 3, 4]):
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller  # Default setting : "Pong"

        print "Initializing DQN..."

        print "Model Building"
        self.CNN_model = FunctionSet(
            l1=F.Convolution2D(4, 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
            l2=F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
            l3=F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
            )

        self.model = FunctionSet(
            l4=F.Linear(3136, 512, wscale=np.sqrt(2)),
            q_value=F.Linear(512, self.num_of_actions,
                             initialW=np.zeros((self.num_of_actions, 512),
                                               dtype=np.float32))
        ).to_gpu()
        
        d = 'elite/'
        
        self.CNN_model.l1.W.data = np.load(d+'l1_W.npy')#.astype(np.float32)
        self.CNN_model.l1.b.data = np.load(d+'l1_b.npy')#.astype(np.float32)
        self.CNN_model.l2.W.data = np.load(d+'l2_W.npy')#.astype(np.float32)
        self.CNN_model.l2.b.data = np.load(d+'l2_b.npy')#.astype(np.float32)
        self.CNN_model.l3.W.data = np.load(d+'l3_W.npy')#.astype(np.float32)
        self.CNN_model.l3.b.data = np.load(d+'l3_b.npy')#.astype(np.float32)

        self.CNN_model = self.CNN_model.to_gpu()
        self.CNN_model_target = copy.deepcopy(self.CNN_model)
        self.model_target = copy.deepcopy(self.model)


        
        print "Initizlizing Optimizer"
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
        self.optimizer.setup(self.model.collect_parameters())

        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        self.D = [np.zeros((self.data_size, 4, 84, 84), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, 4, 84, 84), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool),
                  np.zeros((self.data_size, 1), dtype=np.uint8)]
        


    def forward(self, state, action, Reward, state_dash, episode_end):
        num_of_batch = state.shape[0]
        s = Variable(state)
        s_dash = Variable(state_dash)

        Q = self.Q_func(s)  # Get Q-value

        # Generate Target Signals
        tmp = self.Q_func_target(s_dash)  # Q(s',*)
        tmp = list(map(np.max, tmp.data.get()))  # max_a Q(s',a)
        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(Q.data.get(), dtype=np.float32)

        for i in xrange(num_of_batch):
            if not episode_end[i][0]:
                tmp_ = np.sign(Reward[i]) + self.gamma * max_Q_dash[i]
            else:
                tmp_ = np.sign(Reward[i])

            action_index = self.action_to_index(action[i])
            target[i, action_index] = tmp_

        # TD-error clipping
        td = Variable(cuda.to_gpu(target)) - Q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero_val = Variable(cuda.to_gpu(np.zeros((self.replay_size, self.num_of_actions), dtype=np.float32)))
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss, Q

    def stockExperience(self, time,
                        state, action, lstm_reward, state_dash,
                        episode_end_flag, ale_reward):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.D[0][data_index] = state
            self.D[1][data_index] = action
            self.D[2][data_index] = lstm_reward
            self.D[5][data_index] = ale_reward
        else:
            self.D[0][data_index] = state
            self.D[1][data_index] = action
            self.D[2][data_index] = lstm_reward
            self.D[3][data_index] = state_dash
            self.D[5][data_index] = ale_reward
        self.D[4][data_index] = episode_end_flag


    def experienceReplay(self, time):

        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            s_replay = np.ndarray(shape=(self.replay_size, 4, 84, 84), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, 4, 84, 84), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.D[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.D[1][replay_index[i]]
                r_replay[i] = self.D[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.D[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.D[4][replay_index[i]]

            s_replay = cuda.to_gpu(s_replay)
            s_dash_replay = cuda.to_gpu(s_dash_replay)

            # Gradient-based update
            self.optimizer.zero_grads()
            loss, _ = self.forward(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.optimizer.update()

    def Q_func(self, state):
        h1 = F.relu(self.CNN_model.l1(state / 254.0))  # scale inputs in [0.0 1.0]
        h2 = F.relu(self.CNN_model.l2(h1))
        h3 = F.relu(self.CNN_model.l3(h2))
        h4 = F.relu(self.model.l4(h3))
        #test now
        #print h3.data.shape
        Q = self.model.q_value(h4)
        return Q 

    
    def Q_func_LSTM(self, state):
        h1 = F.relu(self.CNN_model.l1(state / 254.0))  # scale inputs in [0.0 1.0]
        h2 = F.relu(self.CNN_model.l2(h1))
        h3 = F.relu(self.CNN_model.l3(h2))
        return h3.data.get()
    
        
    def Q_func_target(self, state):
        h1 = F.relu(self.CNN_model_target.l1(state / 254.0))  # scale inputs in [0.0 1.0]
        h2 = F.relu(self.CNN_model_target.l2(h1))
        h3 = F.relu(self.CNN_model_target.l3(h2))
        h4 = F.relu(self.model.l4(h3))
        Q = self.model_target.q_value(h4)
        return Q
    
    def LSTM_reward(self, lstm_out, state_next):
        lstm_reward = np.sign((self.lstm_loss - (lstm_out - state_next)**2))
        return lstm_reward

    def e_greedy(self, state, epsilon):
        s = Variable(state)
        Q = self.Q_func(s)
        Q = Q.data

        if np.random.rand() < epsilon:
            index_action = np.random.randint(0, self.num_of_actions)
            print "RANDOM"
        else:
            index_action = np.argmax(Q.get())
            print "GREEDY"
        return self.index_to_action(index_action), Q

    def target_model_update(self):
        self.model_target = copy.deepcopy(self.model)

    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]

    def action_to_index(self, action):
        return self.enable_controller.index(action)


class dqn_agent(Agent):  # RL-glue Process
    lastAction = Action()
    policyFrozen = False

    def agent_init(self, taskSpec):
        # Some initializations for rlglue
        self.lastAction = Action()

        self.time = 0
        self.epsilon = 0.0  # Initial exploratoin rate

        # Pick a DQN from DQN_class
        self.DQN = DQN_class()  # default is for "Pong".
        self.lstm_class = LSTM_class()


    def agent_start(self, observation):
        self.lstm_State = make_initial_state(1045)
        for key, value in self.lstm_State.items():
            value.data = cuda.to_gpu(value.data)
        # Preprocess
        tmp = np.bitwise_and(np.asarray(observation.intArray[128:]).reshape([210, 160]), 0b0001111)  # Get Intensity from the observation
        obs_array = (spm.imresize(tmp, (110, 84)))[110-84-8:110-8, :]  # Scaling

        # Initialize State
        self.state = np.zeros((4, 84, 84), dtype=np.uint8)
        self.state[0] = obs_array
        state_ = cuda.to_gpu(np.asanyarray(self.state.reshape(1, 4, 84, 84), dtype=np.float32))

        # Generate an Action e-greedy
        returnAction = Action()
        action, Q_now = self.DQN.e_greedy(state_, self.epsilon)
        returnAction.intArray = [action]

        # Update for next step
        self.lastAction = copy.deepcopy(returnAction)
        self.last_state = self.state.copy()
        self.last_observation = obs_array

        return returnAction

    def agent_step(self, reward, observation):

        # Preproces
        tmp = np.bitwise_and(np.asarray(observation.intArray[128:]).reshape([210, 160]), 0b0001111)  # Get Intensity from the observation
        obs_array = (spm.imresize(tmp, (110, 84)))[110-84-8:110-8, :]  # Scaling
        obs_processed = np.maximum(obs_array, self.last_observation)  # Take maximum from two frames

        # Compose State : 4-step sequential observation
        self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], obs_processed], dtype=np.uint8)
        state_ = cuda.to_gpu(np.asanyarray(self.state.reshape(1, 4, 84, 84), dtype=np.float32))
        
        state_for_lstm = Variable(cuda.to_gpu(np.asanyarray(self.last_state.reshape(1, 4, 84, 84), dtype=np.float32)))
        CNNout = self.DQN.Q_func_LSTM(state_for_lstm).reshape(1,64*7*7)
        now_CNN = self.DQN.Q_func_LSTM(Variable(state_)).reshape(1,64*7*7) 

        lstm_in = cuda.to_gpu(CNNout)
        
        returnAction = Action()                                                                                  
        action, Q_now = self.DQN.e_greedy(state_, self.epsilon)           
        returnAction.intArray = [action] 
        
        self.dqn_reward = reward
        #if dqn-reward is not 0, LSTM reset

        if self.dqn_reward != 0:
            reward = self.dqn_reward
            self.lstm_State=make_initial_state(1045)
            for key, value in self.lstm_State.items():
                value.data = cuda.to_gpu(value.data)
        else:
            LState = self.lstm_State
            self.lstm_State, self.lstm_s_dash = self.lstm_class.model_lstm.predict(lstm_in,LState)
            reward = lstm_loss - F.mean_squared_error(self.lstm_s_dash, Variable(cuda.to_gpu(now_CNN)))
            reward = reward.data.get()
            print reward


        # Exploration decays along the time sequence
        if self.policyFrozen is False:  # Learning ON/OFF
            if self.DQN.initial_exploration < self.time:
                self.epsilon -= 1.0/10**6
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print "Initial Exploration : %d/%d steps" % (self.time, self.DQN.initial_exploration)
                eps = 1.0
        else:  # Evaluation
                print "Policy is Frozen"
                eps = 0.05

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.DQN.stockExperience(self.time, self.last_state, self.lastAction.intArray[0], reward, self.state, False, self.dqn_reward)
            self.DQN.experienceReplay(self.time)

        # Target model update
        if self.DQN.initial_exploration < self.time and np.mod(self.time, self.DQN.target_model_update_freq) == 0:
            print "########### MODEL UPDATED ######################"
            self.DQN.target_model_update()
            np.save('params/DQN_LSTM_epoch1/l4_W.npy',self.DQN.model.l4.W.get())
            np.save('params/DQN_LSTM_epoch1/l4_b.npy',self.DQN.model.l4.b.get())
            np.save('params/DQN_LSTM_epoch1/q_value_W.npy',self.DQN.model.q_value.W.get())
            np.save('params/DQN_LSTM_epoch1/q_value_b.npy',self.DQN.model.q_value.b.get())

        # Simple text based visualization
        print ' Time Step %d /   ACTION  %d  /   REWARD %.1f   / EPSILON  %.6f  /   Q_max  %3f' % (self.time, self.DQN.action_to_index(action), np.sign(reward), eps, np.max(Q_now.get()))

        # Updates for next step
        self.last_observation = obs_array


        if self.policyFrozen is False:
            self.lastAction = copy.deepcopy(returnAction)
            self.last_state = self.state.copy()
            self.time += 1

        return returnAction

    def agent_end(self, reward):  # Episode Terminated
        self.dqn_reward = reward
        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.DQN.stockExperience(self.time, self.last_state, self.lastAction.intArray[0], reward, self.last_state, True, self.dqn_reward)
            self.DQN.experienceReplay(self.time)

        # Target model update
        if self.DQN.initial_exploration < self.time and np.mod(self.time, self.DQN.target_model_update_freq) == 0:
            print "########### MODEL UPDATED ######################"
            self.DQN.target_model_update()
            np.save('params/DQN_LSTM_epoch1/l4_W.npy',self.DQN.model.l4.W.get())
            np.save('params/DQN_LSTM_epoch1/l4_b.npy',self.DQN.model.l4.b.get())
            np.save('params/DQN_LSTM_epoch1/q_value_W.npy',self.DQN.model.q_value.W.get())
            np.save('params/DQN_LSTM_epoch1/q_value_b.npy',self.DQN.model.q_value.b.get())
        # Simple text based visualization
        print '  REWARD %.1f   / EPSILON  %.5f' % (np.sign(reward), self.epsilon)
        # Time count
        if self.policyFrozen is False:
            self.time += 1

    def agent_cleanup(self):
        pass

    def agent_message(self, inMessage):
        if inMessage.startswith("freeze learning"):
            self.policyFrozen = True
            return "message understood, policy frozen"

        if inMessage.startswith("unfreeze learning"):
            self.policyFrozen = False
            return "message understood, policy unfrozen"

        if inMessage.startswith("save model"):
            with open('dqn_model.dat', 'w') as f:
                pickle.dump(self.DQN.model, f)
            return "message understood, model saved"

if __name__ == "__main__":
    AgentLoader.loadAgent(dqn_agent())
