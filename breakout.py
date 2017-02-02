# import gym
import gym.wrappers
import numpy as np
import time
import tensorflow as tf
# import matplotlib.pyplot as plt
import configparser
import threading

#
config = configparser.ConfigParser()
config.read('breakout.ini')
#
assert int(config['Atari']['SCREEN_X']) % int(config['Atari']['SCREEN_LOW_X']) == 0
assert int(config['Atari']['SCREEN_Y']) % int(config['Atari']['SCREEN_VERY_LOW_Y']) == 0
assert int(config['Atari']['SCREEN_X']) % int(config['Atari']['SCREEN_LOW_X']) == 0
assert int(config['Atari']['SCREEN_Y']) % int(config['Atari']['SCREEN_VERY_LOW_Y']) == 0

#
float_formatter = lambda x: "%+.6f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})


#
class BreakoutPolicyNetwork:

    def __init__(self):

        # Tensorflow Policy network
        #
        activate = tf.nn.relu
        max_v = 0.2
        learning_rate = float(config['Learn']['LEARNING_RATE'])

        #
        n_input_0 = int(config['Atari']['SCREEN_LOW_X']) * int(config['Atari']['SCREEN_LOW_Y'])
        n_input_1 = n_input_0
        n_input_2 = int(config['Atari']['SCREEN_VERY_LOW_X']) * int(config['Atari']['SCREEN_VERY_LOW_Y'])
        n_input_3 = n_input_2
        n_hidden_1 = int(config['NeuralNetwork']['N_HIDDEN_1'])
        n_hidden_2 = int(config['NeuralNetwork']['N_HIDDEN_2'])
        n_output = int(config['Breakout']['ACTION_N'])

        #
        self.training = tf.placeholder(tf.bool)

        #
        def bn(z, axes, n):
            mean, var = tf.nn.moments(z, axes=axes)
            beta = tf.Variable(tf.constant(0.0, shape=[n]))
            gamma = tf.Variable(tf.constant(1.0, shape=[n]))
            epsilon = 1e-3

            ema = tf.train.ExponentialMovingAverage(decay=float(config['Learn']['BN_Decay']))

            def mean_var_with_update():
                ema_apply_op = ema.apply([mean, var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(mean), tf.identity(var)

            mean, var = tf.cond(self.training,
                                mean_var_with_update,
                                lambda: (ema.average(mean), ema.average(var)))

            return tf.nn.batch_normalization(z, mean, var, beta, gamma, epsilon)

        #
        self.l_input_0 = tf.placeholder(tf.float32, [None, n_input_0], name='l_input_0')
        self.l_input_1 = tf.placeholder(tf.float32, [None, n_input_1], name='l_input_1')
        self.l_input_2 = tf.placeholder(tf.float32, [None, n_input_2], name='l_input_2')
        self.l_input_3 = tf.placeholder(tf.float32, [None, n_input_3], name='l_input_3')
        #
        w0_0 = tf.Variable(tf.random_uniform([n_input_0, n_hidden_1], minval=-max_v, maxval=max_v), name='w0_0')
        w0_1 = tf.Variable(tf.random_uniform([n_input_1, n_hidden_1], minval=-max_v, maxval=max_v), name='w0_1')
        w0_2 = tf.Variable(tf.random_uniform([n_input_2, n_hidden_1], minval=-max_v, maxval=max_v), name='w0_2')
        w0_3 = tf.Variable(tf.random_uniform([n_input_3, n_hidden_1], minval=-max_v, maxval=max_v), name='w0_3')
        b0 = tf.Variable(tf.zeros([n_hidden_1]), name='b0')
        l_hidden_1_z = tf.add(tf.add(
            tf.add(tf.matmul(self.l_input_0, w0_0), tf.matmul(self.l_input_1, w0_1)),
            tf.add(tf.matmul(self.l_input_2, w0_2), tf.matmul(self.l_input_3, w0_3))), b0, name='l_hidden_1_z')
        l_hidden_1_bn = bn(l_hidden_1_z, [0], n_hidden_1)
        l_hidden_1 = activate(l_hidden_1_bn, name='l_hidden_1')
        #
        w1 = tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], minval=-max_v, maxval=max_v), name='w1')
        b1 = tf.Variable(tf.zeros([n_hidden_2]), name='b1')
        l_hidden_2_z = tf.add(tf.matmul(l_hidden_1, w1), b1, name='l_hidden_2_z')
        l_hidden_2_bn = bn(l_hidden_2_z, [0], n_hidden_2)
        l_hidden_2 = activate(l_hidden_2_bn, name='l_hidden_2')
        #
        w2 = tf.Variable(tf.random_uniform([n_hidden_2, n_output], minval=-max_v, maxval=max_v), name='w2')
        b2 = tf.Variable(tf.zeros([n_output]), name='b2')
        l_output_z = tf.add(tf.matmul(l_hidden_2, w2), b2, name='l_output_z')
        l_output_bn = bn(l_output_z, [0], n_output)
        self.l_output = tf.nn.softmax(l_output_bn, name='l_output')
        #
        self.l_better_output = tf.placeholder(tf.float32, [None, n_output], name='l_better_output')
        #
        self.f_decays = tf.placeholder(tf.float32, [None], name='f_decays')
        #
        self.cross_entropy = tf.reduce_mean(tf.multiply(
            self.f_decays, -tf.reduce_sum(self.l_better_output * tf.log(tf.clip_by_value(self.l_output, 1e-3, 1.0)),
                                          reduction_indices=[1])), name='decayed_cross_entropy')
        #
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cross_entropy)
        #
        self.saver = tf.train.Saver()
        #
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer(),
                      feed_dict={self.training: True})

        #
        self.restore()

    def save(self):
        print("Saving model...")
        self.saver.save(self.sess, config['Tensorflow']['MODEL_DIR'] + '/' + config['Tensorflow']['MODEL_FILE'])
        print("Successfully saved model")

    def restore(self):

        ckpt = tf.train.get_checkpoint_state(config['Tensorflow']['MODEL_DIR'])
        if config.getboolean('Tensorflow', 'MODEL_LOAD') and ckpt and ckpt.model_checkpoint_path:
            print("Restoring model...")
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Successfully restored model")


#
class GymPlayer:

    @staticmethod
    def screen_shrink(data, x, y, z):
        return data.reshape(
            x, int(data.shape[0] / x),
            y, int(data.shape[1] / y),
            z, int(data.shape[2] / z)).mean(axis=(1, 3, 5))

    def __init__(self, no, pn):

        self.no = no

        #
        self.env = gym.make('Breakout-v0')
        self.pn = pn

        #
        ep_run = int(config['Learn']['EPISODES_PER_RUN'])
        t_max = int(config['Learn']['T_MAX'])

        #
        obs_dim_0 = int(config['Atari']['SCREEN_LOW_X']) * int(config['Atari']['SCREEN_LOW_Y'])
        obs_dim_1 = obs_dim_0
        obs_dim_2 = int(config['Atari']['SCREEN_VERY_LOW_X']) * int(config['Atari']['SCREEN_VERY_LOW_Y'])
        obs_dim_3 = obs_dim_2
        actions_n = int(config['Breakout']['ACTION_N'])

        #
        self.ept = -1
        self.ex_observations_0 = np.empty((ep_run * t_max, obs_dim_0))
        self.ex_observations_1 = np.empty((ep_run * t_max, obs_dim_1))
        self.ex_observations_2 = np.empty((ep_run * t_max, obs_dim_2))
        self.ex_observations_3 = np.empty((ep_run * t_max, obs_dim_3))
        self.ex_actions = np.empty((ep_run * t_max, actions_n))
        self.ex_actions_done = np.empty((ep_run * t_max, actions_n), dtype=np.bool)
        self.ex_rewards = np.empty(ep_run * t_max)
        self.ex_better_actions = np.empty((ep_run * t_max, actions_n))
        self.ex_decays = np.empty(ep_run * t_max)
        self.ep_total_rewards = np.empty(ep_run)

        print("GymWorker #{} created".format(no))

    def run_episodes(self, ep_n=1, render=False):

        print("GymWorker #{} is running {} New Episodes...".format(self.no, ep_n))
        thread = threading.Thread(target=self.run, args=(ep_n, render))
        thread.start()
        return thread

    def run(self, ep_n=1, render=False):

        pn = self.pn

        #
        t_max = int(config['Learn']['T_MAX'])
        s_low_x = int(config['Atari']['SCREEN_LOW_X'])
        s_low_y = int(config['Atari']['SCREEN_LOW_Y'])
        s_vlow_x = int(config['Atari']['SCREEN_VERY_LOW_X'])
        s_vlow_y = int(config['Atari']['SCREEN_VERY_LOW_Y'])
        p_adjust = float(config['Learn']['P_ADJUST'])
        decay = float(config['Learn']['DECAY'])
        action_n = int(config['Breakout']['ACTION_N'])
        action_offset = int(config['Breakout']['ACTION_OFFSET'])

        # cumulative t for all episode
        self.ept = 0 
        
        for ep in range(ep_n):

            observation = self.env.reset()

            # breakout-specific code
            lives = -1
            ept_start = self.ept

            #
            total_reward = 0

            #
            prev_frame_low = np.zeros(s_low_x * s_low_y, dtype=np.int8)
            prev_frame_vlow = np.zeros(s_vlow_x * s_vlow_y, dtype=np.int8)

            for t in range(t_max):

                ept = self.ept

                #
                if render:
                    self.env.render()
                    time.sleep(float(config['Atari']['RENDER_SLEEP']))

                #
                frame_low = np.array(self.screen_shrink(observation, s_low_x, s_low_y, 1).mean(axis=2) > 0,
                                     dtype=np.int8).flatten()
                frame_low_diff = frame_low - prev_frame_low
                prev_frame_low = frame_low
                #
                frame_vlow = np.array(self.screen_shrink(observation, s_vlow_x, s_vlow_y, 1).mean(axis=2) > 0,
                                      dtype=np.int8).flatten()
                frame_vlow_diff = frame_vlow - prev_frame_vlow
                prev_frame_vlow = frame_vlow

                #
                [actions] = pn.sess.run(
                    [pn.l_output],
                    feed_dict={
                        pn.l_input_0: frame_low.reshape((1, -1)),
                        pn.l_input_1: frame_low_diff.reshape((1, -1)),
                        pn.l_input_2: frame_vlow.reshape((1, -1)),
                        pn.l_input_3: frame_vlow_diff.reshape((1, -1)),
                        pn.training: False
                    })

                #
                actions_noise = np.random.random(action_n) * float(config['Breakout']['ACTION_NOISE']) / action_n
                actions_p = actions.flatten() + actions_noise
                actions_p /= np.sum(actions_p)
                #
                action = np.random.choice(action_n, p=actions_p)
                # action = np.random.randint(action_n)

                # start a game
                if ept - ept_start == 0:
                    action = 0

                # save
                self.ex_observations_0[ept, :] = frame_low
                self.ex_observations_1[ept, :] = frame_low_diff
                self.ex_observations_2[ept, :] = frame_vlow
                self.ex_observations_3[ept, :] = frame_vlow_diff
                self.ex_actions[ept, :] = actions
                self.ex_actions_done[ept, :] = False
                self.ex_actions_done[ept, action] = True
                self.ex_rewards[ept] = 0

                #
                observation, reward, done, info = self.env.step(action + action_offset)

                #
                # print("GymWorker #{}: Episode {}/{}: t#{}: reward:{}, info:{} ".format(
                #     self.no, ep, ep_n, t, reward, info))

                total_reward += reward
                info_lives = info['ale.lives']

                # check dead or alive
                if lives != info_lives:  # new life
                    # if dead, give -1 reward
                    if t > 0:
                        reward = -1.0
                    #
                    lives = info_lives

                # got (plus or minus) reward
                if reward != 0:

                    m = ept - ept_start + 1

                    self.ex_rewards[ept_start:ept+1] = reward

                    better_actions = self.ex_actions[ept_start:ept+1].copy()
                    actions_done = self.ex_actions_done[ept_start:ept+1]
                    better_actions[actions_done] += reward * p_adjust
                    better_actions[better_actions < 0] = 0
                    better_actions[:, :] *= np.array(1.0 / better_actions.sum(axis=1)).reshape(-1, 1)

                    # assert np.count_nonzero(np.abs((better_actions.sum(axis=1)) - 1.0) > 0.01) == 0
                    # assert np.count_nonzero(better_actions < 0) == 0
                    # assert np.count_nonzero(better_actions > 1) == 0

                    self.ex_better_actions[ept_start:ept+1, :] = better_actions

                    self.ex_decays[ept_start:ept+1] = np.array([decay ** (m - i - 1) for i in range(m)])

                    ept_start = ept + 1

                if done or t == t_max - 1:

                    self.ep_total_rewards[ep] = total_reward

                    print("GymWorker #{:<2d} has finished episode #{:<3d} for reward {:5.1f} and t#{} ".format(
                        self.no, ep, total_reward, t))
                    break

                #
                self.ept += 1


class GymTrainer:

    def __init__(self, player_n):

        #
        self.pn = BreakoutPolicyNetwork()

        #
        self.gps = []
        #
        for my_i in range(player_n):
            self.gps.append(GymPlayer(my_i, self.pn))

    def play(self, render):

        threads = []
        for i, gp in enumerate(self.gps):
            threads.append(gp.run_episodes(int(config['Learn']['EPISODES_PER_RUN']),
                                           render=render))

        for t in threads:
            t.join()

    def train(self):

        print("Start training for {} iterations".format(int(config['Learn']['ITERATIONS_PER_LEARN'])))

        gps = self.gps
        gp_n = len(gps)

        #
        all_epts = np.array([gps[i].ept for i in range(gp_n)])
        all_avg_reward = np.mean([gps[i].ep_total_rewards for i in range(gp_n)], axis=(0, 1))
        all_min_reward = np.min([gps[i].ep_total_rewards for i in range(gp_n)])
        all_max_reward = np.max([gps[i].ep_total_rewards for i in range(gp_n)])

        #
        print("total experiences: {:,}, reward avg: {:5.2f} min: {} max: {}".format(
            np.sum(all_epts), all_avg_reward, all_min_reward, all_max_reward))

        #
        all_observations_0 = np.concatenate([gps[i].ex_observations_0[0:all_epts[i]+1] for i in range(gp_n)])
        all_observations_1 = np.concatenate([gps[i].ex_observations_1[0:all_epts[i]+1] for i in range(gp_n)])
        all_observations_2 = np.concatenate([gps[i].ex_observations_2[0:all_epts[i]+1] for i in range(gp_n)])
        all_observations_3 = np.concatenate([gps[i].ex_observations_3[0:all_epts[i]+1] for i in range(gp_n)])
        all_better_actions = np.concatenate([gps[i].ex_better_actions[0:all_epts[i]+1] for i in range(gp_n)])
        all_decays = np.concatenate([gps[i].ex_decays[0:all_epts[i]+1] for i in range(gp_n)])

        pn = self.pn

        for i in range(int(config['Learn']['ITERATIONS_PER_LEARN'])):

            [xe, _] = pn.sess.run(
                [pn.cross_entropy, pn.optimize],
                feed_dict={
                    pn.l_input_0: all_observations_0,
                    pn.l_input_1: all_observations_1,
                    pn.l_input_2: all_observations_2,
                    pn.l_input_3: all_observations_3,
                    pn.l_better_output: all_better_actions,
                    pn.f_decays: all_decays,
                    pn.training: True
                })

            if i % int(config['Learn']['PRINT_PER_ITERATIONS']) == 0:
                print("learn #{:<5d}: cross entropy: {:7.4f}".format(i, xe))

        print("Finished training")


play_only = config.getboolean('Atari', 'PLAY_ONLY')

if play_only:
    config['Learn']['PLAYER_N'] = str(1)

trainer = GymTrainer(int(config['Learn']['PLAYER_N']))

for run in range(int(config['Learn']['RUNS'])):

    print("Start to run #{}...".format(run))

    trainer.play(render=play_only)

    if not play_only:
        trainer.train()
        if run % int(config['Learn']['SAVE_MODEL_PER_RUNS']) == 0:
            trainer.pn.save()
