#!/usr/bin/env python

'''
Just Another DAgger Implementation
Author: JJ Zhu
MIT license
based on the starter code from Berkeley CS294-112
'''

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    #===========================================================================
    # generate expert data
    #===========================================================================
    # param
    expert_policy_file = 'experts/Humanoid-v1.pkl'
    envname = 'Humanoid-v1'
    render = 1
    num_rollouts = 25
    max_timesteps = 0
    # policy_fn contains expert policy
    policy_fn = load_policy.load_policy(expert_policy_file)
    with tf.Session():
        tf_util.initialize()
        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit
    
        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                # action using expert policy policy_fn
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
    
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        
        # pass observations, actions to imitation learning
        obs_data = np.squeeze(np.array(observations))
        act_data = np.squeeze(np.array(actions))
        
    save_expert_mean = np.mean(returns)
    save_expert_std = np.std(returns)
    
    #===========================================================================
    # set up the network structure for the imitation learning policy function
    #===========================================================================
    # dim for input/output
    obs_dim = obs_data.shape[1]
    act_dim = act_data.shape[1]
    
    # architecture of the MLP policy function
    x = tf.placeholder(tf.float32, shape=[None, obs_dim])
    yhot = tf.placeholder(tf.float32, shape=[None, act_dim])
    is_train = tf.placeholder(tf.bool)
    
    h1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
    # drop1=tf.layers.dropout(inputs=h1,rate=0.5,training=is_train)
    h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu)
    # drop2=tf.layers.dropout(inputs=h2,rate=0.5,training=is_train)
    h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu)
    # drop3=tf.layers.dropout(inputs=h3,rate=0.5,training=is_train)
    yhat = tf.layers.dense(inputs=h3, units=act_dim, activation=None)
    
    loss_l2 = tf.reduce_mean(tf.square(yhot - yhat))
    train_step = tf.train.AdamOptimizer().minimize(loss_l2)

    #===========================================================================
    # run DAgger alg
    #===========================================================================
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # record return and std for plotting
        save_mean = []
        save_std = []
        save_train_size = []
        # loop for dagger alg
        for i_dagger in xrange(50):
            print 'DAgger iteration ', i_dagger
            # train a policy by fitting the MLP
            batch_size = 25
            for step in range(10000):
                batch_i = np.random.randint(0, obs_data.shape[0], size=batch_size)
                train_step.run(feed_dict={x: obs_data[batch_i, ], yhot: act_data[batch_i, ], is_train: 1})
                if (step % 1000 == 0):
                    print 'opmization step ', step
                    print 'obj value is ', loss_l2.eval(feed_dict={x:obs_data, yhot:act_data, is_train: 0})
            print 'Optimization Finished!'
            # use trained MLP to perform
            max_steps = env.spec.timestep_limit
    
            returns = []
            observations = []
            actions = []
            for i in range(num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = yhat.eval(feed_dict={x:obs[None, :], is_train:0})
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1   
                    if render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
    
            # expert labeling
            act_new = []
            for i_label in xrange(len(observations)):
                act_new.append(policy_fn(observations[i_label][None, :]))
            # record training size
            train_size = obs_data.shape[0]
            # data aggregation
            obs_data = np.concatenate((obs_data, np.array(observations)), axis=0)
            act_data = np.concatenate((act_data, np.squeeze(np.array(act_new))), axis=0)
            # record mean return & std
            save_mean = np.append(save_mean, np.mean(returns))
            save_std = np.append(save_std, np.std(returns))
            save_train_size = np.append(save_train_size, train_size)
            
    dagger_results = {'means': save_mean, 'stds': save_std, 'train_size': save_train_size,
                      'expert_mean':save_expert_mean, 'expert_std':save_expert_std}
    print 'DAgger iterations finished!'

if __name__ == '__main__':
    main()
