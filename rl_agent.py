#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import tensorflow as tf
import tf_slim as slim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from ns3gym import ns3env
import time

import argparse
from ns3gym import ns3env
from tcp_base import TcpTimeBased
from tcp_newreno import TcpNewReno

__author__ = "Kenan and Sharif, Modified the code by Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2020, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"

parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')
args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)

port = 5555
simTime = 60 # seconds
stepTime = 0.5  # seconds
seed = 12
simArgs = {"--duration": simTime,}
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
# simpler:
#env = ns3env.Ns3Env()
env.reset()

ob_space = env.observation_space
ac_space = env.action_space

print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)


def get_agent(obs):
    socketUuid = obs[0]
    tcpEnvType = obs[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        if tcpEnvType == 0:
            # event-based = 0
            tcpAgent = TcpNewReno()
        else:
            # time-based = 1
            tcpAgent = TcpTimeBased()
        tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent

# initialize variable
get_agent.tcpAgents = {}
get_agent.ob_space = ob_space
get_agent.ac_space = ac_space

s_size = ob_space.shape[0]
print("State size: ",ob_space.shape[0])

a_size = 3
print("Action size: ", a_size)

model = keras.Sequential()
model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
model.add(keras.layers.Dense(a_size, activation='softmax'))
#model.compile(optimizer=tf.train.AdamOptimizer(0.001),
model.compile(optimizer=tf.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

weights = model.layers[0].get_weights()
print("Weights: ",weights[0])

print(model.summary())

total_episodes = 6
max_env_steps = 1000
env._max_episode_steps = max_env_steps

epsilon = 1.0               # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999

time_history = []
rew_history = []
tp_history = []
cWnd_history=[]
cWnd_history2=[]
rtt_history=[]
segAkc=[]
No_step = 0
t2 =10
t =[]
reward = 0
done = False
info = None
action_mapping = {}
action_mapping[0] = 0
action_mapping[1] = 600
action_mapping[2] = -60
U_new =0
U =0
U_old=0
bw = 2 #Mbps
delta1 = 0.5
delta2=0.5
p=0
throughput=0
training_weights=[]

stepIdx = 0        
currIt = 0

try:
    while True:
        print("Start iteration: ", currIt)
        done = False
        info = None
        obs = env.reset()
        print("Step: ", stepIdx)
        obs = np.reshape(obs, [1, s_size])
        rewardsum = 0

        for e in range(total_episodes):
            print("Episode count: ",e)
            #obs = env.reset()
            #print("Episode obs: ",obs)
            #obs = np.reshape(obs, [1, s_size])
            cWnd = obs[0,5]
            #print("U new: ", U_new)
            #print("U old: ", U_old)
            rewardsum = 0
            for etime in range(max_env_steps):
                # Choose action
                sseconds = time.time()
                #print("Start time seconds since epoch =", sseconds)
                print("Step obs: ",obs)
                if np.random.rand(1) < epsilon:
                    action_index = np.random.randint(3)
                    print ("Action index inside loop: ",action_index)
                    print("Value Initialization ...")
                else:
                    #print("Prediction1 : ",model.predict(obs)[0])
                    action_index = np.argmax(model.predict(obs)[0])
                    #print(action_index)
                new_cWnd = cWnd + action_mapping[action_index]
                new_ssThresh = np.int_(cWnd/2)
                actions = [new_ssThresh, new_cWnd]

                # sign change
                # change weights
                # parameter change
                # time-based, convert 0 to very small value
                U_new = 0.7*(np.log(obs[0,2])) - 0.7*(np.log(obs[0,9] ))
                # U_new = 0

                print("1. Sim time: ",obs[0,2])
                print("2. Time based - Seg ack: ",obs[0,9])
                print("2. Event based - RTT: ",obs[0,9])

                #print("Throughput: ",obs[0,15])
                #print("Delay: ",(obs[0,11]-obs[0,12]))

                #U_new = np.log(obs[0,15]/bw) - delta1 * np.log(obs[0,11]-obs[0,12]) + delta2 * np.log(1-p)
                #print("Log Throughput: ",np.log(obs[0,15]/bw))
                #print("Log Delay: ",np.log(obs[0,11]-obs[0,12]))

                print("U new: ", U_new)
                print("U old: ", U_old)

                U=U_new-U_old

                print("Delta U: ",U)
                
                if U <-0.05:
                    reward=-5
                elif U >0.05:
                    #reward=8
                    reward=1
                else:
                    reward=0
                
                # Step
                next_state, _, done, info = env.step(actions)

                '''
                if U <-0.05:
                    reward+=-5
                elif U >0.05:
                    #reward=8
                    reward+=1
                else:
                    reward+=0'''

                print("reward: ",reward)

                cWnd = next_state[5]
                #print("Next state: ", next_state)
                print("cWnd:",cWnd)

                if done:
                    print("episode: {}/{}, etime: {}, rew: {}, eps: {:.2}"
                        .format(e, total_episodes, etime, rewardsum, epsilon))
                    print("model training done!")
                    break

                #U_old=0.7*(np.log(obs[0,2]))-0.7*(np.log(obs[0,9] ))
                U_old = U_new
                #print("Log error: ",obs[0,9])
                
                next_state = np.reshape(next_state, [1, s_size])
                # Train
                target = reward
                if not done:
                    #print("Prediction2 : ",model.predict(next_state)[0])
                    target = (reward + 0.95 * np.amax(model.predict(next_state)[0]))

                print("target: ", target)
                target_f = model.predict(obs)
                print("target_f: ", target_f)

                target_f[0][action_index] = target
                print("Updated target_f: ", target_f)
                model.fit(obs, target_f, epochs=1, verbose=0)
                weights = model.layers[0].get_weights()
                #training_weights.append(weights[0])

                obs = next_state
                seg=obs[0,7] #for event based
                #seg=obs[0,9] #for time based

                rtt=obs[0,9] #for event based
                #rtt=obs[0,11] #for time based
                
                #throughput = obs[0,15]  #for time based
                rewardsum += reward
                if epsilon > epsilon_min: epsilon *= epsilon_decay
                No_step += 1
        
                print("number of steps :", No_step)
                #print("epsilon :",epsilon)
        
                #print("reward sum", rewardsum)
                segAkc.append(seg)
                rtt_history.append(rtt)
                #tp_history.append(throughput)

                cWnd_history.append(cWnd)
                time_history.append(time)
                rew_history.append(rewardsum)
                eseconds = time.time()
                #print("Total time: ",eseconds-sseconds)
            
            print("Plot Learning Performance: ")
            mpl.rcdefaults()
            mpl.rcParams.update({'font.size': 16})

            #fig, ax = plt.subplots(2, 2, figsize=(4,2))
            #plt.tight_layout(pad=0.3)
            fig = plt.figure(figsize=(10,6))
            plt.plot(range(len(cWnd_history)), cWnd_history, marker="", linestyle="-")
            plt.title('Congestion windows')
            plt.xlabel('Steps')
            plt.ylabel('CWND (segments)')
            plt.savefig('cwnd.png'.format(e))

            #fig = plt.figure(figsize=(10,6))
            #plt.plot(range(len(tp_history)), tp_history, marker="", linestyle="-")
            #plt.title('Throughput over time')
            #plt.xlabel('Steps')
            #plt.ylabel('Throughput (bits)')
            #plt.savefig('throughput.png'.format(e))

            fig = plt.figure(figsize=(10,6))
            plt.plot(range(len(rtt_history)), rtt_history, marker="", linestyle="-")
            plt.title('RTT over time')
            plt.xlabel('Steps')
            plt.ylabel('RTT (microseconds)')
            plt.savefig('rtt.png'.format(e))

            fig = plt.figure(figsize=(10,6))
            plt.plot(range(len(rew_history)), rew_history, marker="", linestyle="-")
            plt.title('Reward sum plot')
            plt.xlabel('Steps')
            plt.ylabel('Accumulated reward')
            plt.savefig('reward.png'.format(e))

            #ax[0, 0].plot(range(len(cWnd_history)), cWnd_history, marker="", linestyle="-")
            #ax[0, 0].set_title('Congestion windows')
            #ax[0, 0].set_xlabel('Steps')
            #ax[0, 0].set_ylabel('CWND (segments)')

            #ax[0, 1].plot(range(len(tp_history)), tp_history, marker="", linestyle="-")
            #ax[0, 1].set_title('Throughput over time')
            #ax[0, 1].set_xlabel('Steps')
            #ax[0, 1].set_ylabel('Throughput (bits)')

            #ax[1, 0].plot(range(len(rtt_history)), rtt_history, marker="", linestyle="-")
            #ax[1, 0].set_title('RTT over time')
            #ax[1, 0].set_xlabel('Steps')
            #ax[1, 0].set_ylabel('RTT (microseconds)')

            #ax[1, 1].plot(range(len(rew_history)), rew_history, marker="", linestyle="-")
            #ax[1, 1].set_title('Reward sum plot')
            #ax[1, 1].set_xlabel('Steps')
            #ax[1, 1].set_ylabel('Accumulated reward')

            #plt.savefig('plots.png')
            #plt.show()
            '''
            plt.grid(True, linestyle='--')
            plt.title('Learning Performance')
            #plt.plot(range(len(rew_history)), rew_history, label='Reward', marker="^", linestyle=":")#, color='red')
            plt.plot(range(len(rew_history)), rew_history, label='Reward', marker="", linestyle="-")#, color='k')

            #plt.plot(range(len(segAkc)), segAkc, label='segAkc', marker="", linestyle="-"),# color='b')
            plt.plot(range(len(Rtt)),Rtt, label='Rtt', marker="", linestyle="-")#, color='y')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.legend(prop={'size': 12})
            plt.savefig('learning.pdf', bbox_inches='tight')
            plt.show()'''

            if done:
                print("episode: {}/{}, etime: {}, rew: {}, eps: {:.2}"
                    .format(e, total_episodes, etime, rewardsum, epsilon))
                print("model training done!")
                break

        #print("Training weights: ",training_weights)
        while True and (not done):
            action_index = np.argmax(model.predict(obs)[0])
            print("Action index outside loop: ",action_index)
            new_cWnd = cWnd + action_mapping[action_index]
            new_ssThresh = np.int_(cWnd/2)
            actions = [new_ssThresh, new_cWnd]

            # Step
            next_state, reward, done, info = env.step(actions)
            print("cWnd:",next_state[5])

            obs = np.reshape(next_state, [1, s_size])

            if done:
                print("episode: {}/{}, etime: {}, rew: {}, eps: {:.2}"
                    .format(e, total_episodes, etime, rewardsum, epsilon))
                print("outside loop done")
                break

        final_weights = model.layers[0].get_weights()
        print("Final weights: ",final_weights)

        currIt += 1
        if currIt == iterationNum:
            break

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
except Exception as e:
    print(e)
finally:
    env.close()
    print("Finally Done")
