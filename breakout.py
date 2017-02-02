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
        max_v = 0.1
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


class Experiences:
    
    def __init__(self, ex_max):
        
        #
        self.exs = dict()
        self.m_lim = ex_max
        self.m = 0
        self.rollp = 0  # rolling pointer
        
        #
        obsrv_low_dim = int(config['Atari']['SCREEN_LOW_X']) * int(config['Atari']['SCREEN_LOW_Y'])
        obsrv_low_diff_dim = obsrv_low_dim
        obsrv_vlow_dim = int(config['Atari']['SCREEN_VERY_LOW_X']) * int(config['Atari']['SCREEN_VERY_LOW_Y'])
        obsrv_vlow_diff_dim = obsrv_vlow_dim
        actions_n = int(config['Breakout']['ACTION_N'])

        self.exs['obsrv_low'] = np.empty((ex_max, obsrv_low_dim))
        self.exs['obsrv_low_diff'] = np.empty((ex_max, obsrv_low_diff_dim))
        self.exs['obsrv_vlow'] = np.empty((ex_max, obsrv_vlow_dim))
        self.exs['obsrv_vlow_diff'] = np.empty((ex_max, obsrv_vlow_diff_dim))
        self.exs['actions'] = np.empty((ex_max, actions_n), dtype=np.float)
        self.exs['actions_done'] = np.empty((ex_max, actions_n), dtype=np.bool)
        self.exs['rewards'] = np.empty(ex_max)
        self.exs['better_actions'] = np.empty((ex_max, actions_n), dtype=np.float)
        self.exs['decays'] = np.empty(ex_max)

    def set(self, ex_key, ex_i, ex):
        self.exs[ex_key][ex_i] = ex

    def get(self, ex_key, ex_i=None):
        if ex_i is None:
            exs = self.exs[ex_key][0:self.m]
        else:
            exs = self.exs[ex_key][ex_i]
        return exs

    def concatenate(self, exs_list):
        for exs in exs_list:
            m = exs.m
            if self.rollp + m > self.m_lim:
                assert self.rollp + m < self.m_lim * 2  # don't allow case of rolling twice
                do_roll = True
                m1 = self.m_lim - self.rollp
                m2 = m - m1
            else:
                do_roll = False

            for ex_key in self.exs:
                if do_roll:
                    self.set(ex_key, slice(self.rollp, self.rollp + m1), exs.get(ex_key, slice(0, m1)))
                    self.set(ex_key, slice(0, m2), exs.get(ex_key, slice(m1, m)))
                else:
                    self.set(ex_key, slice(self.rollp, self.rollp + m), exs.get(ex_key))

            if do_roll:
                self.m = self.m_lim
                self.rollp = m2
            else:
                if self.m != self.m_lim:
                    self.m += m
                self.rollp += m


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
        ep_n = int(config['Learn']['EPISODES_PER_RUN'])

        #

        #
        self.exs = Experiences(ep_n * int(config['Learn']['T_MAX']))

        #
        self.ept = -1

        #
        self.ep_total_rewards = np.empty(ep_n)

        print("GymWorker #{:<2d} created".format(no))

    def run_episodes(self, ep_n=1, render=False, p_adjust=0.1):

        print("GymWorker #{:<2d} is running {} New Episodes...".format(self.no, ep_n))
        thread = threading.Thread(target=self.run, args=(ep_n, render, p_adjust))
        thread.start()
        return thread

    def run(self, ep_n=1, render=False, p_adjust=0.1):

        pn = self.pn

        #
        t_max = int(config['Learn']['T_MAX'])
        s_low_x = int(config['Atari']['SCREEN_LOW_X'])
        s_low_y = int(config['Atari']['SCREEN_LOW_Y'])
        s_vlow_x = int(config['Atari']['SCREEN_VERY_LOW_X'])
        s_vlow_y = int(config['Atari']['SCREEN_VERY_LOW_Y'])
        decay = float(config['Learn']['DECAY'])
        action_n = int(config['Breakout']['ACTION_N'])
        action_offset = int(config['Breakout']['ACTION_OFFSET'])

        #
        self.ept = 0
        ept_start = 0

        for ep in range(ep_n):

            observation = self.env.reset()

            # breakout-specific code
            lives = -1

            #
            total_reward = 0

            #
            prev_frame_low = np.zeros(s_low_x * s_low_y, dtype=np.int8)
            prev_frame_vlow = np.zeros(s_vlow_x * s_vlow_y, dtype=np.int8)

            for t in range(t_max):

                ept = self.ept
                exs = self.exs

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

                # add noise
                actions_noise = np.random.random(action_n) * float(config['Breakout']['ACTION_NOISE']) / action_n
                actions_p = actions.flatten() + actions_noise
                actions_p /= np.sum(actions_p)
                #                #
                action = np.random.choice(action_n, p=actions_p)
                # action = np.random.randint(action_n)

                # start a game
                if ept - ept_start == 0:
                    action = 0

                #
                actions_done = np.zeros(action_n, dtype=np.bool)
                actions_done[action] = True

                # save
                exs.set('obsrv_low', ept, frame_low)
                exs.set('obsrv_low_diff', ept, frame_low_diff)
                exs.set('obsrv_vlow', ept, frame_vlow)
                exs.set('obsrv_vlow_diff', ept, frame_vlow_diff)
                exs.set('actions', ept, actions)
                exs.set('actions_done', ept, actions_done)
                exs.set('rewards', ept, 0)

                #
                observation, reward, done, info = self.env.step(action + action_offset)

                #
                # print("GymWorker #{:2d}: Episode {:3d}/{:<3d}: t#{:<5d}: action:{}, reward:{}, info:{} ".format(
                #     self.no, ep, ep_n, t, action, reward, info))

                total_reward += reward
                info_lives = info['ale.lives']

                timeout = False
                if t == t_max - 1:
                    timeout = True
                    reward = 0  # TODO: ???

                # check dead or alive
                if lives != info_lives:  # new life
                    # if dead, give -1 reward
                    if t > 0:
                        reward = -1.0
                    #
                    lives = info_lives

                # got (plus or minus) reward
                if reward != 0 or done or timeout:

                    r_m = ept - ept_start + 1

                    exs.set('rewards', slice(ept_start, ept+1), reward)

                    r_better_actions = exs.get('actions', slice(ept_start, ept+1)).copy()
                    r_actions_done = exs.get('actions_done', slice(ept_start, ept+1))

                    # DEBUG
                    # print(np.mean(r_actions_done, axis=0, dtype=np.float))

                    r_better_actions[r_actions_done] += reward * p_adjust
                    r_better_actions[r_better_actions < 0] = 0
                    r_better_actions[:, :] *= np.array(1.0 / r_better_actions.sum(axis=1)).reshape(-1, 1)

                    assert np.count_nonzero(np.abs((r_better_actions.sum(axis=1)) - 1.0) > 0.01) == 0
                    assert np.count_nonzero(r_better_actions < 0) == 0
                    assert np.count_nonzero(r_better_actions > 1) == 0

                    exs.set('better_actions', slice(ept_start, ept+1), r_better_actions)

                    exs.set('decays', slice(ept_start, ept+1),
                            np.array([decay ** (r_m - i - 1) for i in range(r_m)]))

                    ept_start = ept + 1

                #
                self.ept += 1

                if done or timeout:

                    # TODO: automate setting m or rollp
                    self.exs.m = self.ept
                    self.exs.rollp = self.ept

                    assert self.exs.exs['decays'][self.exs.m - 1] == 1
                    assert np.count_nonzero(np.abs(
                        np.sum(self.exs.exs['better_actions'][0:self.exs.m], axis=1) - 1.0) > 0.01) == 0

                    self.ep_total_rewards[ep] = total_reward

                    print("GymWorker #{:<2d} has finished episode #{:<3d} for reward {:5.1f} and t#{} ".format(
                        self.no, ep, total_reward, t))
                    break


class GymTrainer:

    def __init__(self, player_n):

        #
        self.pn = BreakoutPolicyNetwork()

        #
        self.gps = []
        #
        for my_i in range(player_n):
            self.gps.append(GymPlayer(my_i, self.pn))

        self.past_exs = Experiences(int(config['Learn']['HISTORY_EX_MAX']))

    def play(self, render, p_adjust):

        threads = []
        for i, gp in enumerate(self.gps):
            threads.append(gp.run_episodes(int(config['Learn']['EPISODES_PER_RUN']),
                                           render=render,
                                           p_adjust=p_adjust))

        for t in threads:
            t.join()

    def train(self):

        print("Start training for {} iterations".format(int(config['Learn']['ITERATIONS_PER_LEARN'])))

        gps = self.gps
        gp_n = len(gps)

        #
        m = np.sum([gps[i].exs.m for i in range(gp_n)])
        avg_reward = np.mean([gps[i].ep_total_rewards for i in range(gp_n)], axis=(0, 1))
        min_reward = np.min([gps[i].ep_total_rewards for i in range(gp_n)])
        max_reward = np.max([gps[i].ep_total_rewards for i in range(gp_n)])

        #
        print("last run total experiences: {:,}, reward avg: {:5.2f} min: {} max: {}".format(
            m, avg_reward, min_reward, max_reward))

        #
        if config.getboolean('Learn', 'USE_PAST_EX'):
            past_exs = self.past_exs
            past_exs.concatenate([gps[i].exs for i in range(gp_n)])
            exs = past_exs
            #
            print("past total experiences: {:,}".format(past_exs.m))
        else:
            exs = Experiences(m)
            exs.concatenate([gps[i].exs for i in range(gp_n)])

        if exs.m > 0:

            pn = self.pn

            exs_obsrv_low = exs.get('obsrv_low')
            exs_obsrv_low_diff = exs.get('obsrv_low_diff')
            exs_obsrv_vlow = exs.get('obsrv_vlow')
            exs_obsrv_vlow_diff = exs.get('obsrv_vlow_diff')
            exs_better_actions = exs.get('better_actions')
            exs_decays = exs.get('decays')

            for i in range(int(config['Learn']['ITERATIONS_PER_LEARN'])):

                [xe, _] = pn.sess.run(
                    [pn.cross_entropy, pn.optimize],
                    feed_dict={
                        pn.l_input_0: exs_obsrv_low,
                        pn.l_input_1: exs_obsrv_low_diff,
                        pn.l_input_2: exs_obsrv_vlow,
                        pn.l_input_3: exs_obsrv_vlow_diff,
                        pn.l_better_output: exs_better_actions,
                        pn.f_decays: exs_decays,
                        pn.training: True
                    })

                if (i % int(config['Learn']['PRINT_PER_ITERATIONS']) == 0 or
                        i == int(config['Learn']['ITERATIONS_PER_LEARN']) - 1):

                    print("learn #{:<5d}: cross entropy: {:7.4f}".format(i, xe))

        print("Finished training")

play_only = config.getboolean('Atari', 'PLAY_ONLY')

if play_only:
    config['Learn']['PLAYER_N'] = '1'
    config['Breakout']['ACTION_NOISE'] = '0.0' 

trainer = GymTrainer(int(config['Learn']['PLAYER_N']))

my_p_adjust = float(config['Learn']['P_ADJUST_START'])

for run in range(int(config['Learn']['RUNS'])):

    print("Start to run #{}...".format(run))
    print("Set p_adjust to {:4.2f}".format(my_p_adjust))

    trainer.play(render=play_only, p_adjust=my_p_adjust)

    if not play_only:
        trainer.train()
        if run > 0 and run % int(config['Learn']['SAVE_MODEL_PER_RUNS']) == 0:
            trainer.pn.save()

    my_p_adjust = max(my_p_adjust * float(config['Learn']['P_ADJUST_DECAY']),
                      float(config['Learn']['P_ADJUST_END']))
