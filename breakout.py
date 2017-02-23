import gym.wrappers
import numpy as np
import time
import math
import tensorflow as tf
import configparser
import threading

#
config = configparser.ConfigParser()
config.read('breakout.ini')

#
assert int(config['Atari']['SCREEN_X']) % int(config['Atari']['SCREEN_LOW_X']) == 0
assert int(config['Atari']['SCREEN_Y']) % int(config['Atari']['SCREEN_LOW_Y']) == 0

#
np.set_printoptions(linewidth=np.nan, threshold=np.nan, formatter={'float_kind': lambda x: "%4.2f" % x})

#
dim_input = [int(config['Atari']['SCREEN_LOW_Y']),
             int(config['Atari']['SCREEN_LOW_X']),
             int(config['Atari']['SCREEN_LOW_Z'])]


#
class BreakoutPolicyNetwork:

    def __init__(self):

        # Tensorflow Policy network
        #
        max_v = 0.1
        learning_rate = float(config['PolicyNetwork']['LEARNING_RATE'])

        #
        n_conv_1 = int(config['PolicyNetwork']['N_CONV_1'])
        n_conv_2 = int(config['PolicyNetwork']['N_CONV_2'])
        n_conv_3 = int(config['PolicyNetwork']['N_CONV_3'])
        n_conv_4 = int(config['PolicyNetwork']['N_CONV_4'])
        #
        n_fc_1 = int(config['PolicyNetwork']['N_FC_1'])
        n_fc_2 = int(config['PolicyNetwork']['N_FC_2'])
        #
        n_action = int(config['Breakout']['ACTION_N'])

        #
        self.training = tf.placeholder(tf.bool)

        #
        def bn(z, axes, n, name='bn'):
            mean, var = tf.nn.moments(z, axes=axes)
            beta = tf.Variable(tf.constant(0.0, shape=[n]))
            gamma = tf.Variable(tf.constant(1.0, shape=[n]))
            epsilon = 1e-5

            ema = tf.train.ExponentialMovingAverage(decay=float(config['PolicyNetwork']['BN_Decay']))

            def mean_var_with_update():
                ema_apply_op = ema.apply([mean, var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(mean), tf.identity(var)

            mean, var = tf.cond(self.training,
                                mean_var_with_update,
                                lambda: (ema.average(mean), ema.average(var)))

            return tf.nn.batch_normalization(z, mean, var, beta, gamma, epsilon, name=name)

        def conv(l_prv, n_filter, f, s=1, activate=tf.nn.relu, name='conv'):

            n_prv = l_prv.get_shape().as_list()[-1]

            w = tf.Variable(tf.truncated_normal([f, f, n_prv, n_filter]), name=name + '_w')
            b = tf.Variable(tf.ones([n_filter]), name=name + '_b')
            l_z = tf.add(tf.nn.conv2d(l_prv, w, [1, s, s, 1], padding='SAME'), b, name=name + '_z')
            l_a = activate(l_z, name=name + '_a')

            return l_a

        def ccconv(l_prv, n_filter, activate=tf.nn.relu, name='conv'):

            l_a00 = conv(l_prv, n_filter, 1, s=1, activate=activate, name=name)
            l_a01 = conv(l_a00, n_filter, 3, s=2, activate=activate, name=name)
            l_a10 = conv(l_prv, n_filter, 1, s=1, activate=activate, name=name)
            l_a11 = conv(l_a10, n_filter, 5, s=2, activate=activate, name=name)
            l_a20 = conv(l_prv, n_filter, 1, s=1, activate=activate, name=name)
            l_a21 = conv(l_a20, n_filter, 7, s=2, activate=activate, name=name)

            l_c = tf.concat(3, [l_a01, l_a11, l_a21])

            l_bn = bn(l_c, [0, 1, 2], n_filter*3, name=name + '_bn')

            return l_bn

        def fc(l_prv, n_node, activate=tf.identity, name='fc'):

            n_prv = l_prv.get_shape().as_list()[-1]
            w = tf.Variable(tf.random_uniform([n_prv, n_node], minval=-max_v, maxval=max_v), name=name + '_w')
            b = tf.Variable(tf.ones([n_node]), name=name + '_b')
            l_cur_z = tf.add(tf.matmul(l_prv, w), b, name=name + '_z')
            # l_cur_bn = bn(l_cur_z, [0], n_node, name=name + '_bn')
            l_cur = activate(l_cur_z, name=name + '_a')

            return l_cur

        #
        self.l_obsrv_cur0 = tf.placeholder(tf.float32, [None] + dim_input, name='l_obsrv_cur0')
        self.l_obsrv_chg1 = tf.placeholder(tf.float32, [None] + dim_input, name='l_obsrv_chg1')
        self.l_obsrv_chg2 = tf.placeholder(tf.float32, [None] + dim_input, name='l_obsrv_chg2')
        self.l_obsrv_chg3 = tf.placeholder(tf.float32, [None] + dim_input, name='l_obsrv_chg3')

        l_input = tf.concat(
            3, [self.l_obsrv_cur0, self.l_obsrv_chg1, self.l_obsrv_chg2, self.l_obsrv_chg3], name='l_input')

        l_conv_1 = ccconv(l_input, n_conv_1, name='conv1')
        l_conv_2 = ccconv(l_conv_1, n_conv_2, name='conv2')
        l_conv_3 = ccconv(l_conv_2, n_conv_3, name='conv3')
        l_conv_4 = ccconv(l_conv_3, n_conv_4, name='conv4')

        l_conv_flat = tf.reshape(l_conv_4, [-1, np.prod(l_conv_4.get_shape().as_list()[1:4])], name='conv_flat')

        l_fc_1 = fc(l_conv_flat, n_fc_1, activate=tf.nn.relu, name='l_fc_1')
        l_fc_2 = fc(l_fc_1, n_fc_2, activate=tf.nn.relu, name='l_fc_2')

        l_action_z = fc(l_fc_2, n_action, name='l_action_z')

        #
        self.l_output = tf.nn.softmax(l_action_z)
        #
        self.i_actions_done = tf.placeholder(tf.float32, [None, n_action], name='i_actions_done')
        self.i_actions_target = tf.placeholder(tf.float32, [None, n_action], name='i_actions_target')

        self.i_decayed_impacts = tf.placeholder(tf.float32, [None], name='i_decayed_impact')

        # self.loss = tf.reduce_mean(
        #     self.i_decayed_impacts *
        #     tf.reduce_sum(self.i_actions_done * (self.i_actions_target - self.l_output) ** 2,
        #                   reduction_indices=[1]))

        # self.loss = tf.reduce_mean(
        #     self.i_decayed_impacts *
        #     tf.reduce_sum(self.i_actions_done * self.i_actions_target * -tf.log(self.l_output),
        #                   reduction_indices=[1]))
        #
        self.loss = tf.reduce_mean(
            self.i_decayed_impacts *
            tf.reduce_sum(self.i_actions_done * -tf.log(1 - 0.99*tf.abs(self.i_actions_target - self.l_output)),
                          reduction_indices=[1]))

        self.optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)

        #
        self.saver = tf.train.Saver()
        #
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer(), feed_dict={self.training: True})

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
        actions_n = int(config['Breakout']['ACTION_N'])

        self.exs['obsrv_cur0'] = np.empty([ex_max] + dim_input, dtype=np.float16)
        self.exs['obsrv_chg1'] = np.empty([ex_max] + dim_input, dtype=np.float16)
        self.exs['obsrv_chg2'] = np.empty([ex_max] + dim_input, dtype=np.float16)
        self.exs['obsrv_chg3'] = np.empty([ex_max] + dim_input, dtype=np.float16)

        self.exs['actions'] = np.empty((ex_max, actions_n), dtype=np.int8)
        self.exs['actions_done'] = np.empty((ex_max, actions_n), dtype=np.bool)
        self.exs['actions_target'] = np.empty((ex_max, actions_n), dtype=np.int8)

        self.exs['decayed_impacts'] = np.empty(ex_max, dtype=np.float16)

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
    def screen_shrink(data, dim):
        [x, y, z] = dim
        return data.reshape(
            x, int(data.shape[0] / x),
            y, int(data.shape[1] / y),
            z, int(data.shape[2] / z)).mean(axis=(1, 3, 5))

    def __init__(self, no, pn, ep_n=1, play_only=False):

        self.no = no

        #
        self.env = gym.make('Breakout-v0')
        self.play_only = play_only

        if self.play_only:
            self.env = gym.wrappers.Monitor(self.env, 'tmp/Breakout-v0-experiment', force=True)

        self.pn = pn

        #
        self.exs = Experiences(ep_n * int(config['Player']['T_MAX']))

        #
        self.ept = -1

        #
        self.ep_total_rewards = np.empty(ep_n)

        print("GymWorker #{:<2d} created".format(no))

    def run_episodes(self, ep_n=1):

        print("GymWorker #{:<2d} is running {} New Episodes...".format(self.no, ep_n))
        thread = threading.Thread(target=self.run, args=(ep_n,))
        thread.start()
        return thread

    def run(self, ep_n=1):

        pn = self.pn

        #
        t_max = int(config['Player']['T_MAX'])
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
            frame_cur0 = np.zeros(dim_input)
            frame_prv1 = frame_cur0.copy()
            frame_prv2 = frame_cur0.copy()

            #
            for t in range(t_max):

                #
                if self.play_only:
                    self.env.render()

                #
                ept = self.ept
                exs = self.exs

                #
                frame_prv3 = frame_prv2
                frame_prv2 = frame_prv1
                frame_prv1 = frame_cur0
                frame_cur0 = GymPlayer.screen_shrink(observation/255.0, dim_input)

                frame_chg1 = frame_cur0 - frame_prv1
                frame_chg2 = frame_cur0 - frame_prv2
                frame_chg3 = frame_cur0 - frame_prv3

                # no_cpu = 8
                no_gpu = 2
                with tf.device('/gpu:'+str(self.no % no_gpu)):
                    [actions] = pn.sess.run(
                            [pn.l_output],
                            feed_dict={
                                pn.l_obsrv_cur0: frame_cur0.reshape([1] + dim_input),
                                pn.l_obsrv_chg1: frame_chg1.reshape([1] + dim_input),
                                pn.l_obsrv_chg2: frame_chg2.reshape([1] + dim_input),
                                pn.l_obsrv_chg3: frame_chg3.reshape([1] + dim_input),
                                pn.training: False
                            })

                if self.play_only:
                    action_noise = 0.0
                elif ep % 2 == 0:
                    action_noise = float(config['Breakout']['ACTION_NOISE_0'])
                else:
                    action_noise = float(config['Breakout']['ACTION_NOISE_1'])

                if np.random.random() < action_noise:
                    action = np.random.randint(action_n)
                else:
                    action = np.argmax(actions)

                # start a game
                if ept - ept_start < 3:
                    action = 0

                # if self.no == 0:
                #     print(actions, action)
                #
                actions_done = np.zeros(action_n, dtype=np.bool)
                actions_done[action] = True

                # save
                exs.set('obsrv_cur0', ept, frame_cur0)
                exs.set('obsrv_chg1', ept, frame_chg1)
                exs.set('obsrv_chg2', ept, frame_chg2)
                exs.set('obsrv_chg3', ept, frame_chg3)
                exs.set('actions', ept, actions)
                exs.set('actions_done', ept, actions_done)
                exs.set('decayed_impacts', ept, 0)

                #
                observation, reward, done, info = self.env.step(action + action_offset)

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

                    r_actions_target = exs.get('actions', slice(ept_start, ept+1)).copy()
                    r_actions_done = exs.get('actions_done', slice(ept_start, ept+1))

                    if reward > 0:
                        r_actions_target[r_actions_done] = 1.0
                        exs.set('actions_target', slice(ept_start, ept + 1), r_actions_target)
                    elif reward == 0:
                        r_actions_done = False
                        exs.set('actions_done', slice(ept_start, ept + 1), r_actions_done)
                    else:
                        r_actions_target[r_actions_done] = 0.0
                        exs.set('actions_target', slice(ept_start, ept + 1), r_actions_target)

                    ts = ept + 1 - ept_start
                    decay = float(config['Player']['IMPACT_DECAY']) ** np.arange(ts)[::-1]
                    exs.set('decayed_impacts', slice(ept_start, ept + 1), decay / np.sum(decay) * abs(reward))

                    ept_start = ept + 1

                #
                self.ept += 1

                if done or timeout:

                    # TODO: automate setting m or rollp
                    self.exs.m = self.ept
                    self.exs.rollp = self.ept

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
            self.gps.append(GymPlayer(my_i, self.pn, int(config['Trainer']['EPISODES_PER_RUN'])))

        if config.getboolean('Trainer', 'PAST_EX_USE'):
            self.past_exs = Experiences(int(config['Trainer']['PAST_EX_MAX']))

        #
        self.avg_reward_history = []
        self.max_reward_history = []

    def play(self, render=False):

        threads = []
        for i, gp in enumerate(self.gps):
            threads.append(gp.run_episodes(int(config['Trainer']['EPISODES_PER_RUN']), render))

        for t in threads:
            t.join()

    def train(self):

        print("Start training for {} iterations".format(int(config['Trainer']['EPOCHS_PER_LEARN'])))

        gps = self.gps
        gp_n = len(gps)

        #
        m = np.sum([gps[i].exs.m for i in range(gp_n)])
        avg_reward = np.mean([gps[i].ep_total_rewards for i in range(gp_n)], axis=(0, 1))
        min_reward = np.min([gps[i].ep_total_rewards for i in range(gp_n)])
        max_reward = np.max([gps[i].ep_total_rewards for i in range(gp_n)])

        self.avg_reward_history.append(avg_reward)
        self.max_reward_history.append(max_reward)

        #
        print("last run total experiences: {:,}, reward avg: {:5.2f} min: {:5.2f} max: {:5.2f}".format(
            m, avg_reward, min_reward, max_reward))

        #
        def moving_avg(a, n):
            ma = np.cumsum(a, dtype=np.float)
            ma[n:] = (ma[n:] - ma[:-n]) / n
            ma[0:n] = ma[0:n] / np.arange(1, min(len(ma), n) + 1)
            return ma

        #
        print_hist_n = 100
        moving_window = 10

        print("last {} run reward stat:".format(print_hist_n))
        print("avg:")
        print(np.array(self.avg_reward_history[-print_hist_n:]))
        print("last-{} avg:".format(moving_window))
        print(moving_avg(np.array(self.avg_reward_history[-print_hist_n:]), moving_window))
        print("max:")
        print(np.array(self.max_reward_history[-print_hist_n:]))

        #
        if config.getboolean('Trainer', 'PAST_EX_USE'):
            past_exs = self.past_exs
            past_exs.concatenate([gps[i].exs for i in range(gp_n)])
            exs = past_exs

            # sample expriences from past experiences, prefer recent expriences
            mm = float(config['Trainer']['PAST_EX_MM'])
            ex_is0 = (exs.rollp - np.arange(m)) % exs.m
            ex_is1 = (exs.rollp - m -
                      np.abs(np.random.uniform(0, exs.m - m, min(int(m*mm), exs.m-m))).astype(np.int)) % exs.m
            ex_is = np.concatenate((ex_is0, ex_is1))
            ex_m = len(ex_is)

            #
            print("learning past total experiences: {:,}/{:,}".format(ex_m, past_exs.m))
        else:
            exs = Experiences(m)
            exs.concatenate([gps[i].exs for i in range(gp_n)])
            ex_is = np.arange(m)
            ex_m = m

        if ex_m > 0:

            pn = self.pn

            mbs = int(config['Trainer']['MINI_BATCH_SIZE'])

            for i in range(int(config['Trainer']['EPOCHS_PER_LEARN'])):

                mbn = math.ceil(ex_m / mbs)
                loss = np.empty(mbn)

                for j in range(mbn):

                    exs_start = mbs * j
                    exs_end = mbs * (j+1)

                    ex_mb = ex_is[exs_start:exs_end]

                    exs_obsrv_cur0 = exs.get('obsrv_cur0', ex_mb)
                    exs_obsrv_chg1 = exs.get('obsrv_chg1', ex_mb)
                    exs_obsrv_chg2 = exs.get('obsrv_chg2', ex_mb)
                    exs_obsrv_chg3 = exs.get('obsrv_chg3', ex_mb)
                    exs_actions_done = exs.get('actions_done', ex_mb)
                    exs_actions_target = exs.get('actions_target', ex_mb)
                    exs_decayed_impacts = exs.get('decayed_impacts', ex_mb)

                    [loss[j], _] = pn.sess.run(
                        [pn.loss, pn.optimize],
                        feed_dict={
                            pn.l_obsrv_cur0: exs_obsrv_cur0,
                            pn.l_obsrv_chg1: exs_obsrv_chg1,
                            pn.l_obsrv_chg2: exs_obsrv_chg2,
                            pn.l_obsrv_chg3: exs_obsrv_chg3,
                            pn.i_actions_done: exs_actions_done,
                            pn.i_actions_target: exs_actions_target,
                            pn.i_decayed_impacts: exs_decayed_impacts,
                            pn.training: True
                        })

                if (i % int(config['Trainer']['PRINT_PER_EPOCHS']) == 0 or
                        i == int(config['Trainer']['EPOCHS_PER_LEARN']) - 1):

                    print("learn #{:<5d}: loss: {:11.9f}".format(i, np.mean(loss)))

        print("Finished training")


time_prg = time.time()

if config.getboolean('Trainer', 'PLAY_ONLY'):

        my_pn = BreakoutPolicyNetwork()
        my_gp = GymPlayer(0, my_pn, play_only=True)
        while True:
            my_gp.run()

else:

    trainer = GymTrainer(int(config['Trainer']['PLAYER_N']))

    for run in range(int(config['Trainer']['RUNS'])):

        time_run = time.time()
        print("Start to run #{} at {}...".format(run, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())))

        trainer.play()

        trainer.train()
        if run > 0 and run % int(config['Trainer']['SAVE_MODEL_PER_RUNS']) == 0:
            trainer.pn.save()

        print("Finished run #{} for {} secs".format(run, int(time.time() - time_run)))
