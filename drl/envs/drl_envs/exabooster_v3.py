import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import os
import errno
import sys
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
import requests
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)
np.seterr(divide='ignore', invalid='ignore')
import pickle
from drl.models.sngp_lstm_regression_v2 import SNGP_LSTM_Regression
import sys
#sys.path.append("/home/schram/repositories/fnal_surrogate_utils")
from drl.utils.CreateBoosterTrainingData import LoadHDF5Data
#from utils.CreateBoosterTrainingData import LoadHDF5Data

def regulation(alpha, gamma, error, min_set, beta):
    # calculate the prediction with current regulation rules
    ER = error  # error
    _MIN = min_set  # setting
    for i in range(len(_MIN)):
        if i > 0:
            beta_t = beta[-1] + gamma * ER[i]
            beta.append(beta_t)  # hopefully this will update self.rachael_beta in place
    MIN_pred = _MIN - alpha * ER - np.asarray(beta[-15:]).reshape(15, 1)  # predict the next, shiftting happens in the plotting #check here
    # used to be 15, now 150
    return MIN_pred

def scale(value, min_val, max_val):
    #print('min_val',min_val)
    #print('max_val',max_val)
    return  value*(max_val - min_val) + min_val

class ExaBooster_v3(gym.Env):
    def __init__(self):

        self.save_dir = os.getcwd()
        self.episodes = 0
        self.steps = 0
        self.max_steps = 100
        self.episodic_reward = 0
        self.episodic_data_reward  = 0
        self.rachael_reward = 0
        self.diff = 0
        #
        self.rl_reward_list = []
        self.rl_episodic_reward_list = []

        self.data_reward_list = []
        #self.rachael_reward_list = []
        #
        self.avg_rl_reward_list = []
        #self.avg_rachael_reward_list = []
        self.avg_data_reward_list = []

        self.vimin_list = []
        self.predictions_list = []
        self.pred_error_list = []
        #
        #self.rachael_beta = [0]

        # Read in the data

        #baseline_data_dir = '/Users/schram/processed_data/'
        baseline_data_dir = '/work/data_science/fnal/'
        baseline_data_file = 'ExaBooster_TrainValTest_LSTM_lookback_15step.h5'
        self.X_train, self.Y_train_o, self.X_val, self.Y_val_o, \
        self.X_test, self.Y_test_o, self.Scale, self.In, self.Out = \
            LoadHDF5Data(os.path.join(baseline_data_dir, baseline_data_file))

        print('X_train.shape:', self.X_train.shape)
        print('Y_train.shape:', self.Y_train_o.shape)

        self.nsamples = self.X_train.shape[0]
        self.ntimes = self.X_train.shape[1]
        self.nvariables = self.X_train.shape[2]

        self.batch_id = 500#self.episodes + 4200

        booster_model_dir = '/home/schram/repositories/dnc2s_rl/dnc2s_rl/notebooks'
        #booster_model_dir = '/work/data_science/kishan/FNAL_rl/'
        booster_model_name = 'sngp_reg_model_weights_v204_04_22_100553'

        print('booster_model_dir', booster_model_dir)
        booster_model_pfn = os.path.join(booster_model_dir,booster_model_name)
        print("booster model file=", booster_model_pfn, flush=True)
        #with tf.device('/cpu:0'):
        self.booster_model = SNGP_LSTM_Regression(hidden_size=256,
                                                  num_inputs=self.nvariables,
                                                  num_outputs=1,
                                                  fourier_dim=256,
                                                  drop_percent=0.2)
        load_status = self.booster_model.load_weights(booster_model_pfn)
        print('load_status:', load_status)
        # Load weights
        with open(booster_model_pfn + '.pkl', 'rb') as f:
            reload_cov, reload_W, reload_b = pickle.load(f)

        self.booster_model.cov = reload_cov
        self.booster_model.W = reload_W
        self.booster_model.b = reload_b
        self.booster_model(self.X_train[0,:,:].reshape(1,self.ntimes,self.nvariables))
        print('booster_model:',self.booster_model.summary())
        self.data_state = None

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.nvariables,),
            dtype=np.float64
        )

        # DYNAMIC ACTION SPACE SIZING
        data_diff = self.Y_train_o[0:-1,0].flatten() - self.Y_train_o[1:,0].flatten()
        median = np.median(data_diff)
        q1 = np.percentile(data_diff, 5)
        q2 = np.percentile(data_diff, 95)
        print('median:',median, ' q1:', q1,' q2:', q2 )
        old_values = 0.005
        # Continuous
        self.action_space = spaces.Box(low=q1,
                                       high=q2,
                                       shape=(1,),
                                       dtype=np.float32)

        self.VIMIN = 0
        self.internal_state = np.zeros(shape=(1, self.ntimes, self.nvariables))
        self.predicted_state = np.zeros(shape=(1, 1, self.nvariables))

#         self.rachael_state = np.zeros(shape=(1, self.ntimes, self.nvariables))
#         self.rachael_predicted_state = np.zeros(shape=(1, 1, self.nvariables))

        #
        self.min_iminer = self.Scale['x_min'][0][1]
        #print(self.min_iminer)
        #sys.exit()
        self.max_iminer = self.Scale['x_max'][0][1]
        self.data_vimin, self.pred_vimin, self.data_iminer, self.pred_iminer, self.pred_iminer_error = [], [], [], [] ,[]

        logger.debug('Init pred shape:{}'.format(self.predicted_state.shape))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps += 1
        logger.debug('Episode/State: {}/{}'.format(self.episodes, self.steps))
        done = False

        # Steps:
        # 1) Update VIMIN based on action
        # 2) Predict booster variables
        # 3) Shift state with new values

        # Step 1: Calculate the new B:VIMIN based on policy action

        logger.debug('Step() before action VIMIN:{}'.format(self.VIMIN))
        self.VIMIN = self.VIMIN + action
        #print(type(self.VIMIN))

        logger.debug('Step() state before action VIMIN:{}'.format(self.internal_state))
        self.internal_state[0][self.ntimes - 1][0] = self.VIMIN
        logger.debug('Step() state after action VIMIN:{}'.format(self.internal_state))

        # Step 2: Predict using booster model
        # For testing 
        isTesting = False
        if isTesting:
            self.internal_state[:,:,0] = np.copy(self.X_train[self.batch_id + self.steps,:,0])
            self.VIMIN = np.array(self.internal_state[:,-1:,0].flatten())
        #print(type(self.VIMIN))
        #
        predicted_state, predicted_error = self.booster_model(self.internal_state)
        pred_iminer = float(predicted_state[0][0])
        pred_iminer_error = float(predicted_error[0][0])
        pred_iminer_rdm = np.random.normal(pred_iminer, pred_iminer_error)
        
        # Calibrate the error
        normalize_pred_iminer_error = np.sqrt(pred_iminer_error) / float(self.nvariables)
        normalize_pred_iminer_error = float(normalize_pred_iminer_error)
        
        # Step 3: Shift state by one step
        self.internal_state[0, 0:-1, : ] = self.internal_state[0, 1:, :]  # shift forward

        # Update IMINER
        self.internal_state[0][self.ntimes - 1][1] = pred_iminer

        # Update data state for rendering
        self.data_state = np.copy(self.X_train[self.batch_id + self.steps].reshape(1, self.ntimes, self.nvariables))
        
        # where's data_vimin
        data_iminer = self.data_state[0,-1:,1]
        # Invert 
        scaled_data_iminer = scale(data_iminer, self.min_iminer, self.max_iminer)
        data_reward = -abs(scaled_data_iminer)

        # Use data for everything but the action and prediction variables
        self.internal_state[0, :, 2:self.nvariables] = self.data_state[0, :, 2:self.nvariables]

        scaled_pred_iminer = scale(pred_iminer, self.min_iminer, self.max_iminer)
        scaled_pred_iminer_error = scale(pred_iminer+normalize_pred_iminer_error, self.min_iminer, self.max_iminer)-scaled_pred_iminer
        
        # Reward
        reward = -abs(scaled_pred_iminer)#-abs(scaled_pred_iminer_error)

        if self.steps >= int(self.max_steps):
            done = True

        # Counter "outside the box"
        ood_flag = 0
        if self.VIMIN>1 or self.VIMIN<0:
            ood_flag = 1
            #reward -= 1.0

        self.vimin_list.append(self.VIMIN)
        self.predictions_list.append(predicted_state[0])
        self.pred_error_list.append(scaled_pred_iminer)

        self.data_vimin.append(self.data_state[0,-1:,0])
        self.pred_vimin.append(self.VIMIN)
        self.data_iminer.append(scaled_data_iminer)
        self.pred_iminer.append(scaled_pred_iminer)
        self.pred_iminer_error.append(scaled_pred_iminer_error)
        self.rl_reward_list.append(-abs(scaled_pred_iminer))
        self.episodic_reward += -abs(scaled_pred_iminer)
        self.episodic_data_reward += data_reward
        return self.internal_state[0, -1:, : ].flatten(), reward, done, [scaled_pred_iminer, scaled_pred_iminer_error, ood_flag, data_reward]

    def reset(self):

        if len(self.data_vimin)>0:
            self.ep_render()
        self.data_vimin, self.pred_vimin, self.data_iminer, self.pred_iminer, self.pred_iminer_error = [], [], [], [] ,[]
        self.rl_reward_list = []
        if self.episodes % 5 == 0:
            fig, axs = plt.subplots(1, figsize=(12, 12))
            plt.plot(self.vimin_list, label='VIMIN')
            plt.plot(self.predictions_list, label='predicted iminor')
            plt.plot(self.pred_error_list, label='prediction errors')
            plt.plot()
            plt.legend()
            plt.savefig('results_dir/reward_episode{}.png'.format(self.episodes))
            plt.close('all')

        self.episodes += 1
        self.steps = 0
        self.episodic_data_reward  = 0
        self.episodic_reward = 0
        self.diff = 0
        self.rachael_reward = 0
        self.rachael_beta = [0]

        # Prepare the random sample ##
        self.batch_id = 500 #np.random.randint(low=500, high=1100)  # to train

        logger.info('Resetting env')
        logger.debug('self.state:{}'.format(self.internal_state))
        self.internal_state = None
        self.internal_state = np.copy(self.X_train[self.batch_id].reshape(1, self.ntimes, self.nvariables))

        self.data_state = None
        self.data_state = np.copy(self.X_train[self.batch_id].reshape(1, self.ntimes, self.nvariables))

        self.rachael_state = None
        self.rachael_state = np.copy(self.X_train[self.batch_id].reshape(1, self.ntimes, self.nvariables))

        logger.debug('self.state:{}'.format(self.internal_state))
        logger.debug('reset_data.shape:{}'.format(self.internal_state.shape))
        self.VIMIN = self.internal_state[0, -1:, 0]

        #

        return self.internal_state[0, -1:, :].flatten()

    def ep_render(self):
        #return 0
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['axes.labelweight'] = 'regular'
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['font.family'] = [u'serif']
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = [u'serif']
        plt.rcParams['font.size'] = 16

        this_reward = float(self.episodic_data_reward/self.episodic_reward)
        print('this_reward:',this_reward)
        self.rl_episodic_reward_list.append(this_reward)
        r_fig, r_axs = plt.subplots(3, figsize=(12, 12), constrained_layout=True)

        #r_axs[0].plot(self.data_vimin, "--", color="black", label='Data')
        r_axs[0].set_title('Episode {} Reward {:.2f}'.format(self.episodes,this_reward))
        r_axs[0].plot(self.rl_episodic_reward_list, "--", color="red", label='Agent')
        r_axs[0].set_xlabel("Episodic Reward")
        r_axs[0].set_ylabel("Steps")
        r_axs[0].legend()
        r_axs[0].axhline(y=1, color='black', linestyle='-')
        
#         r_axs[1].plot(self.rl_reward_list, "--", color="red", label='Agent')
#         r_axs[1].set_xlabel("Reward")
#         r_axs[1].set_ylabel("Steps")
#         r_axs[1].legend()
        
        r_axs[1].plot(self.pred_vimin, "--", color="red", label='Agent')
        r_axs[1].set_xlabel("B:VIMIN (Action)")
        r_axs[1].set_ylabel("Steps")
        r_axs[1].legend()

        #r_axs[1].plot(self.data_iminer, "--", color="black", label='Data')
        r_axs[2].plot(self.pred_iminer, "--", color="red", label='Agent')
        _ = plt.fill_between(np.arange(len(self.pred_iminer)),
                     np.array(self.pred_iminer).flatten() - np.array(self.pred_iminer_error).flatten(),
                     np.array(self.pred_iminer).flatten() + np.array(self.pred_iminer_error).flatten(), color='blue',
                     alpha=0.25)
            
        r_axs[2].set_xlabel("B:IMINER")
        r_axs[2].set_ylabel("Steps")
        r_axs[2].legend()
        
        #
        #plt.draw()
        r_fig.savefig('results_dir/episodes/ExaBoosterSNGP_Episode{}_Traces.png'.format(self.episodes))
        plt.close('all')
        plt.clf()
        return 0
        #
        # logger.debug('render()')
        #
        # import seaborn as sns
        # sns.set_style("ticks")
        # nvars = 2  # len(self.variables)> we just want B:VIMIN and B:IMINER
        # fig, axs = plt.subplots(nvars, figsize=(12, 8))
        # logger.debug('self.state:{}'.format(self.state))
        #
        # # Rachael's Eq
        # alpha = 10e-2
        # gamma = 7.535e-5
        # # try dstate
        # BVIMIN_trace = unscale(self.variables[0], self.state[0, 0, 1:-1].reshape(-1, 1), self.scale_dict)
        # BIMINER_trace = unscale(self.variables[1], self.state[0, 1, :].reshape(-1, 1), self.scale_dict)
        #
        # B_VIMIN_trace = unscale(self.variables[2], self.state[0, 2, :].reshape(-1, 1), self.scale_dict)
        #
        # # something is weird with this change... it definitely is predicting 180 value which isn't right
        # BVIMIN_pred = unscale(self.variables[0], self.rachael_state[0, 0, :].reshape(-1, 1), self.scale_dict)
        # rachael_IMINER = unscale(self.variables[1], self.rachael_state[0, 1, :].reshape(-1, 1), self.scale_dict)
        #
        # for v in range(0, nvars):
        #     utrace = self.state[0, v, :]
        #     trace = unscale(self.variables[v], utrace.reshape(-1, 1), self.scale_dict)
        #     if v == 0:
        #         # soemthing seems weird... might need to actually track it above
        #         axs[v].set_title('Raw data reward: {:.2f} - RL agent reward: {:.2f} - PID Eq reward {:.2f}'.format(self.data_total_reward,
        #                                                                                                            self.total_reward, self.rachael_reward))
        #
        #     axs[v].plot(trace, label='RL Policy', color='black')
        #
        #     # if v==1:
        #     data_utrace = self.data_state[0, v, :]
        #     data_trace = unscale(self.variables[v], data_utrace.reshape(-1, 1), self.scale_dict)
        #
        #     if v == 1:
        #         x = np.linspace(0, 14, 15)  # np.linspace(0, 14, 15) #np.linspace(0, 149, 150) #TODO: change this so that it is dynamic for lookback
        #         axs[v].fill_between(x, -data_trace.flatten(), +data_trace.flatten(), alpha=0.2, color='red')
        #
        #     axs[v].plot(data_trace, 'r--', label='Data')
        #     # axs[v].plot()
        #     axs[v].set_xlabel('time')
        #     axs[v].set_ylabel('{}'.format(self.variables[v]))
        #     # axs[v].legend(loc='upper left')
        #
        # # replaced np.linspace(0,14,15)
        # axs[0].plot(np.linspace(0, 14, 15), BVIMIN_pred, label="PID Eq", color='blue', linestyle='dotted')
        # axs[0].legend(loc='upper left')
        # axs[1].plot(np.linspace(0, 14, 15), rachael_IMINER, label="PID Eq", color='blue', linestyle='dotted')
        # axs[1].legend(loc='upper left')
        #
        # plt.savefig('results_dir/episode{}_step{}_v1.png'.format(self.episodes, self.steps))
        # plt.clf()
        #
        # fig, axs = plt.subplots(1, figsize=(12, 12))
        #
        # Y_agent_bvimin = unscale(self.variables[0], self.state[0][0].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        # Y_agent_biminer = unscale(self.variables[1], self.state[0][1].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        # Y_data_bvimin = unscale(self.variables[0], self.data_state[0][0].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        # Y_data_biminer = unscale(self.variables[1], self.data_state[0][1].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        # Y_rachael_bvimin = unscale(self.variables[0], self.rachael_state[0][0].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        # Y_rachael_iminer = unscale(self.variables[1], self.rachael_state[0][1].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        #
        # np_predict = np.concatenate((Y_data_bvimin, Y_data_biminer, Y_agent_bvimin, Y_agent_biminer,
        #                              Y_rachael_bvimin, Y_rachael_iminer), axis=self.concate_axis)
        # df_cool = pd.DataFrame(np_predict, columns=['bvimin_data', 'biminer_data', 'bvimin_agent', 'biminer_agent', 'bvimin_rachael', 'biminer_rachael'])
        #
        # plt.scatter(Y_data_bvimin, Y_data_biminer, color='red', alpha=0.5, label='Data')
        # plt.scatter(Y_agent_bvimin, Y_agent_biminer, color='black', alpha=0.5, label='RL Policy')
        # plt.scatter(Y_rachael_bvimin, Y_rachael_iminer, color='blue', alpha=0.5, label='PID Eq')
        # plt.xlabel('B:VIMIN')
        # plt.ylabel('B:IMINER')
        # plt.legend()
        # plt.savefig('results_dir/corr_episode{}_step{}.png'.format(self.episodes, self.steps))
        #
        # #
        #
        # plt.close('all')
