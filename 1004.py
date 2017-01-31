# import gym
import gym.wrappers
import numpy as np
import time
import tensorflow as tf
# import matplotlib.pyplot as plt

#
float_formatter = lambda x: "%+.6f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

# Atari 2600 enviroment specific variables
SCREEN_X = 210
SCREEN_Y = 160
SCREEN_Z = 3

# Breakout environment specific variables
SCREEN_LOW_X = int(SCREEN_X / 5)
SCREEN_LOW_Y = int(SCREEN_Y / 5)
SCREEN_LOW_Z = int(SCREEN_Z / 3)
SCREEN_VERY_LOW_X = int(SCREEN_X / 30)
SCREEN_VERY_LOW_Y = int(SCREEN_Y / 20)
SCREEN_VERY_LOW_Z = int(SCREEN_Z / 3)
#
# actions
# 1: start game (maybe ?) or do nothing on playing game
# 2: right
# 3: left
N_ACTIONS = 3
ACTION_OFFSET = 1

# Policy network specific variables
N_INPUT_0 = SCREEN_LOW_X * SCREEN_LOW_Y * SCREEN_LOW_Z  # current frame
N_INPUT_1 = N_INPUT_0  # current frame - previous frame
N_INPUT_2 = SCREEN_VERY_LOW_X * SCREEN_VERY_LOW_Y * SCREEN_VERY_LOW_Z
N_INPUT_3 = N_INPUT_2
N_HIDDEN_1 = 512
N_HIDDEN_2 = 64
N_OUTPUT = N_ACTIONS 

# Trainer specific variables
EP_MAX = 1000000  # max episodes
EP_LEARN = 10  # learn per episodes
T_MAX = 3000  # max time
DECAY = 0.99  # reward decay
EX_MAX = 3 * T_MAX * EP_LEARN  # max experiences to remember
LEARNING_RATE = 1e-4
LEARN_ITERATIONS = 50
LEARN_PRINT = 10
P_ADJUST = 0.5  # adjusting amount of probability of actions
EP_SAVE_MODEL = EP_LEARN  # save model per episodes

#
PLAY_RENDER = False
PLAY_T_DELAY = 0.1

# Tensorflow variabls
TF_CKPT_DIR = 'ckpt/1004/'
TF_CKPT_FILE = 'model.ckpt'
TF_LOAD_MODEL = True

#
env = gym.make('Breakout-v0')

# Check whether an environment is suitable
# assert type(env.action_space) is gym.spaces.discrete.Discrete
# assert type(env.observation_space) is gym.spaces.box.Box
assert env.observation_space.shape == (SCREEN_X, SCREEN_Y, SCREEN_Z)

# Build Tensorflow Policy network 
#
activate = tf.nn.elu
mv = 0.2
#
l_input_0 = tf.placeholder(tf.float32, [None, N_INPUT_0], name='l_input_0')
l_input_1 = tf.placeholder(tf.float32, [None, N_INPUT_1], name='l_input_1')
l_input_2 = tf.placeholder(tf.float32, [None, N_INPUT_2], name='l_input_2')
l_input_3 = tf.placeholder(tf.float32, [None, N_INPUT_3], name='l_input_3')

#
w0_0 = tf.Variable(tf.random_uniform([N_INPUT_0, N_HIDDEN_1], minval=-mv, maxval=mv), name='w0_0')
w0_1 = tf.Variable(tf.random_uniform([N_INPUT_1, N_HIDDEN_1], minval=-mv, maxval=mv), name='w0_1')
w0_2 = tf.Variable(tf.random_uniform([N_INPUT_2, N_HIDDEN_1], minval=-mv, maxval=mv), name='w0_2')
w0_3 = tf.Variable(tf.random_uniform([N_INPUT_3, N_HIDDEN_1], minval=-mv, maxval=mv), name='w0_3')

b0 = tf.Variable(tf.zeros([N_HIDDEN_1]), name='b0')
l_hidden_1 = activate(tf.add(tf.add(
    tf.add(tf.matmul(l_input_0, w0_0), tf.matmul(l_input_1, w0_1)),
    tf.add(tf.matmul(l_input_2, w0_2), tf.matmul(l_input_3, w0_3))), b0), name='l_hidden_1')
#
w1 = tf.Variable(tf.random_uniform([N_HIDDEN_1, N_HIDDEN_2], minval=-mv, maxval=mv), name='w1')
b1 = tf.Variable(tf.zeros([N_HIDDEN_2]), name='b1')
l_hidden_2 = activate(tf.add(tf.matmul(l_hidden_1, w1), b1), name='l_hidden_2')
#
w2 = tf.Variable(tf.random_uniform([N_HIDDEN_2, N_OUTPUT], minval=-mv, maxval=mv), name='w2')
b2 = tf.Variable(tf.zeros([N_OUTPUT]), name='b2')
l_output = tf.nn.softmax(tf.add(tf.matmul(l_hidden_2, w2), b2), name='l_output')
#
l_better_output = tf.placeholder(tf.float32, [None, N_OUTPUT], name='l_better_output')
#
f_decays = tf.placeholder(tf.float32, [None], name='f_decays')
#
cross_entropy = tf.reduce_mean(tf.multiply(
    f_decays, -tf.reduce_sum(l_better_output*tf.log(l_output), reduction_indices=[1])),
    name='decayed_cross_entropy')
#
optimize = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
#
#
saver = tf.train.Saver()
#
sess = tf.Session()
sess.run(tf.global_variables_initializer())


#
ckpt = tf.train.get_checkpoint_state(TF_CKPT_DIR)
if TF_LOAD_MODEL and ckpt and ckpt.model_checkpoint_path:
    print("Restoring model...")
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Successfully restored model")


#
#
def screen_shrink(data, x, y, z):
    return data.reshape(
        x, int(data.shape[0]/x), 
        y, int(data.shape[1]/y), 
        z, int(data.shape[2]/z)).mean(axis=(1, 3, 5))

# experiences to remember (limited by memory)
#
ex = 0
ex_observations_0 = np.empty((EX_MAX, N_INPUT_0))
ex_observations_1 = np.empty((EX_MAX, N_INPUT_1))
ex_observations_2 = np.empty((EX_MAX, N_INPUT_2))
ex_observations_3 = np.empty((EX_MAX, N_INPUT_3))
ex_better_actions = np.empty((EX_MAX, N_ACTIONS))
ex_decays = np.empty(EX_MAX)


def memorize(observations_0, observations_1, observations_2, observations_3,
             actions, actions_done, reward):

    global ex

    #
    m = len(observations_0)

    #
    assert ex + m < EX_MAX

    better_actions = actions.copy()

    better_actions[actions_done] += reward * P_ADJUST
    better_actions[better_actions < 0] = 0
    # noinspection PyTypeChecker
    better_actions[:, :] *= np.array(
        1.0 / better_actions.sum(axis=1)).reshape(-1, 1)

    # assert np.count_nonzero(np.abs((better_actions.sum(axis=1)) - 1.0) > 0.01) == 0
    # assert np.count_nonzero(better_actions < 0) == 0
    # assert np.count_nonzero(better_actions > 1) == 0

    # 
    # print(actions)
    # print(better_actions)

    # decays
    decays = np.array([DECAY ** (t - i) for i in range(m)])

    #
    ex_observations_0[ex:ex+m, :] = observations_0
    ex_observations_1[ex:ex+m, :] = observations_1
    ex_observations_2[ex:ex+m, :] = observations_2
    ex_observations_3[ex:ex+m, :] = observations_3
    ex_better_actions[ex:ex+m, :] = better_actions
    ex_decays[ex:ex+m] = decays
    ex += m


# 
def learn():

    global ex

    print("learning {} experiences...".format(ex))

    for i in range(LEARN_ITERATIONS):
        [xe, _] = sess.run(
            [cross_entropy, optimize],
            feed_dict={
                l_input_0: ex_observations_0[0:ex],
                l_input_1: ex_observations_1[0:ex],
                l_input_2: ex_observations_2[0:ex],
                l_input_3: ex_observations_3[0:ex],
                l_better_output: ex_better_actions[0:ex],
                f_decays: ex_decays[0:ex]})
        if i % LEARN_PRINT == 0 or i == LEARN_ITERATIONS - 1:
            print("iteration #{:4d}, cross entropy:{:7.5f}".format(i, xe))

    # transfered experiences into brain, forget every past experiences
    ex = 0

ep_total_rewards = np.empty(EP_MAX)
# episodes (or epoch?)
for ep in range(EP_MAX):

    print("Episode #{} is starting...".format(ep))

    #
    ep_reward = 0

    #
    ep_observations_0 = np.empty((T_MAX, N_INPUT_0), dtype=np.int8)  # Black & White
    ep_observations_1 = np.empty((T_MAX, N_INPUT_1), dtype=np.int8)
    ep_observations_2 = np.empty((T_MAX, N_INPUT_2), dtype=np.int8)
    ep_observations_3 = np.empty((T_MAX, N_INPUT_3), dtype=np.int8)
    ep_actions = np.zeros((T_MAX, N_ACTIONS))  # probability of actions
    ep_actions_done = np.zeros((T_MAX, N_ACTIONS), dtype=np.bool)  # selected actions

    # noinspection PyRedeclaration
    observation = env.reset()

    # breakout-specific code
    lives = None
    t_life_start = 0
    t_memory_start = 0

    # learn or explore or train and play
    if ep % EP_LEARN == 0:
        if ep > 0:
            print("Rewards of previous {} episodes:".format(EP_LEARN))

            def print_ep_total_rewards(ep_start, ep_end):
                rewards = ep_total_rewards[ep_start:ep_end]
                print("Ep #{:5d}~#{:5d}: Average: {:5.1f}, Min: {:5.1f}, Max: {:5.1f}".format(
                    ep_start, ep_end-1, np.mean(rewards), np.min(rewards), np.max(rewards)))

            print_ep_total_rewards(ep - EP_LEARN, ep)

            print("Rewards average history:")
            for i in range(min(100, int(ep/EP_LEARN)), 0, -1):
                print_ep_total_rewards(ep - EP_LEARN*i, ep-EP_LEARN*(i-1))

            learn()
        play = True
    else:
        play = False

    #
    if ep > 0 and ep % EP_SAVE_MODEL == 0:
        print("Saving model...")
        saver.save(sess, TF_CKPT_DIR + '/' + TF_CKPT_FILE)
        print("Successfully saved model")

    #
    # noinspection PyRedeclaration
    prev_frame_low = np.zeros((SCREEN_LOW_X, SCREEN_LOW_Y), dtype=np.int8)
    # noinspection PyRedeclaration
    prev_frame_very_low = np.zeros((SCREEN_VERY_LOW_X, SCREEN_VERY_LOW_Y), dtype=np.int8)

    for t in range(T_MAX):

        # noinspection PyTypeChecker
        frame_low = np.array(screen_shrink(
            observation, SCREEN_LOW_X, SCREEN_LOW_Y, SCREEN_LOW_Z).mean(axis=2) > 0,
                             dtype=np.int8)
        frame_low_diff = frame_low - prev_frame_low
        prev_frame_low = frame_low
        #
        frame_very_low = np.array(screen_shrink(
            observation, SCREEN_VERY_LOW_X, SCREEN_VERY_LOW_Y, SCREEN_VERY_LOW_Z).mean(axis=2) > 0,
                             dtype=np.int8)
        frame_very_low_diff = frame_very_low - prev_frame_very_low
        prev_frame_very_low = frame_very_low
        #
        # print(frame_low.astype(np.int))
        # print(frame_low_diff.astype(np.int))

        frame_low = frame_low.reshape(N_INPUT_0)
        frame_low_diff = frame_low_diff.reshape(N_INPUT_1)
        frame_very_low = frame_very_low.reshape(N_INPUT_2)
        frame_very_low_diff = frame_very_low_diff.reshape(N_INPUT_3)

        if play and PLAY_RENDER:  
            env.render()
            time.sleep(PLAY_T_DELAY)

        # 
        [t_actions] = sess.run(
            [l_output],
            feed_dict={
                l_input_0: frame_low.reshape((1, N_INPUT_0)),
                l_input_1: frame_low_diff.reshape((1, N_INPUT_1)),
                l_input_2: frame_very_low.reshape((1, N_INPUT_2)),
                l_input_3: frame_very_low_diff.reshape((1, N_INPUT_3)),
                l_better_output: np.zeros((1, N_OUTPUT)),
                f_decays: np.zeros(1)})

        #
        action = np.random.choice(N_OUTPUT, p=t_actions.reshape(N_ACTIONS))

        # breakout-specific
        # breakout game start action
        if t - t_memory_start == 0:
            action = 0

        # save
        ep_observations_0[t, :] = frame_low
        ep_observations_1[t, :] = frame_low_diff
        ep_observations_2[t, :] = frame_very_low
        ep_observations_3[t, :] = frame_very_low_diff
        ep_actions[t, :] = t_actions
        ep_actions_done[t, :] = False
        ep_actions_done[t, action] = True 

        # 
        observation, t_reward, done, info = env.step(action + ACTION_OFFSET)
        ep_reward += t_reward
        info_lives = info['ale.lives']
        
        # print("time: {:4d}, info: {}, action: {}, reward: {}, episode reward: {}".format(
        #     t, info, action, t_reward, ep_reward))

        do_memorize = False

        # got reward
        if t_reward > 0:
            t_reward = int(t_reward)
            t_start = t_life_start
            do_memorize = True

        # check dead or alive
        if lives != info_lives:  # new life
            #
            if t > 0:
                t_reward = -1
                t_start = t_memory_start
                do_memorize = True
            #
            lives = info_lives
            t_life_start = t + 1

        if do_memorize:
            t_reward_sign = "+" if t_reward > 0 else "-"
            print("memorize {} reward: {:+3d} {} experiences (t#{}~t#{})".format(
                t_reward_sign, t_reward, t_reward_sign, t_start, t))
            memorize(
                ep_observations_0[t_start:t], ep_observations_1[t_start:t],
                ep_observations_2[t_start:t], ep_observations_3[t_start:t],
                ep_actions[t_start:t], ep_actions_done[t_start:t],
                t_reward)
            t_memory_start = t + 1

        if done or t == T_MAX - 1:
            ep_total_rewards[ep] = ep_reward
            print("Episode finished for {} rewards at t#{}".format(ep_reward, t))
            print("")
            break
