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
SCREEN_LOW_X = int(SCREEN_X / 10)
SCREEN_LOW_Y = int(SCREEN_Y / 10)
SCREEN_LOW_Z = int(SCREEN_Z / 3)
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
N_HIDDEN_1 = 256
N_HIDDEN_2 = 64
N_OUTPUT = N_ACTIONS 

# Trainer specific variables
EP_MAX = 10000  # max episodes
EP_PLAY = 10  # play per episodes
T_MAX = 1000  # max time
T_SPAN = 50  # experience time span for a reward
DECAY = 0.97  # reward decay
EX_MAX = T_MAX * EP_PLAY  # max experiences to remember
LEARNING_RATE = 1e-4
LEARN_ITERATIONS = 500
LEARN_PRINT = 100
P_ADJUST = 0.5  # adjusting amount of probability of actions 

#
PLAY_T_DELAY = 0.005

#
env = gym.make('Breakout-v0')

# Check whether an environment is suitable
assert type(env.action_space) is gym.spaces.discrete.Discrete
assert type(env.observation_space) is gym.spaces.box.Box
assert env.observation_space.shape == (SCREEN_X, SCREEN_Y, SCREEN_Z)

# Build Tensorflow Policy network 
#
activate = tf.nn.relu
mv = 0.2
#
l_input_0 = tf.placeholder(tf.float32, [None, N_INPUT_0], name='l_input_0')
l_input_1 = tf.placeholder(tf.float32, [None, N_INPUT_1], name='l_input_1')
#
w0_0 = tf.Variable(tf.random_uniform([N_INPUT_0, N_HIDDEN_1], minval=-mv, maxval=mv), name='w0_0')
w0_1 = tf.Variable(tf.random_uniform([N_INPUT_1, N_HIDDEN_1], minval=-mv, maxval=mv), name='w0_1')
b0 = tf.Variable(tf.zeros([N_HIDDEN_1]), name='b0')
l_hidden_1 = activate(tf.add(
    tf.add(tf.matmul(l_input_0, w0_0), tf.matmul(l_input_1, w0_1)), b0), name='l_hidden_1')
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
decays = tf.placeholder(tf.float32, [None], name='decays')
#
cross_entropy = tf.reduce_mean(tf.multiply(
    decays, -tf.reduce_sum(l_better_output*tf.log(l_output), reduction_indices=[1])), 
    name='decayed_cross_entropy')
#
optimize = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
#
#
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#
#
def screen_shrink(data, x, y, z):
    return data.reshape(
        x, int(data.shape[0]/x), 
        y, int(data.shape[1]/y), 
        z, int(data.shape[2]/z)).mean(axis=(1,3,5))

# experiences to remember (limited by memory)
#
ex = 0
ex_observations_0 = np.empty((EX_MAX, N_INPUT_0))
ex_observations_1 = np.empty((EX_MAX, N_INPUT_1))
ex_better_actions = np.empty((EX_MAX, N_ACTIONS))
ex_decays = np.empty((EX_MAX))

def memorize(observations_0, observations_1, actions, actions_done, well_done):

    global ex

    #
    m = len(observations_0)

    #
    assert ex + m < EX_MAX

    better_actions = actions.copy()

    well_sign = 1 if well_done else -1 
    better_actions[actions_done] += well_sign * P_ADJUST
    better_actions[better_actions < 0] = 0
    better_actions[:, :] *= (1.0 / better_actions.sum(axis=1)).reshape(-1, 1)

    # assert np.count_nonzero(np.abs((better_actions.sum(axis=1)) - 1.0) > 0.01) == 0
    # assert np.count_nonzero(better_actions < 0) == 0
    #assert np.count_nonzero(better_actions > 1) == 0

    # 
    # print(actions)
    # print(better_actions)

    # decays
    decays = np.array([DECAY ** (t - i) for i in range(m)])

    #
    ex_observations_0[ex:ex+m, :] = observations_0
    ex_observations_1[ex:ex+m, :] = observations_1
    ex_better_actions[ex:ex+m, :] = better_actions
    ex_decays[ex:ex+m] = decays
    ex += m

# 
def learn():

    global ex

    print("learning {} experiences...".format(ex))
    for i in range(LEARN_ITERATIONS):
        [xe, _] = sess.run([cross_entropy, optimize],
            feed_dict={
                l_input_0:ex_observations_0[0:ex],
                l_input_1:ex_observations_1[0:ex],
                l_better_output: ex_better_actions[0:ex],
                decays: ex_decays[0:ex]})
        if i % LEARN_PRINT == 0:
            print("iteration #{:4d}, cross entropy:{:7.5f}".format(i, xe))

    # transfered experiences into brain, forget every past experiences
    ex = 0


# episodes (or epoch?)
for ep in range(EP_MAX):

    print("Episode #{} is starting...".format(ep))

    #
    ep_reward = 0

    #
    ep_observations_0 = np.empty((T_MAX, N_INPUT_0), dtype=np.bool)  # Black & White
    ep_observations_1 = np.empty((T_MAX, N_INPUT_1), dtype=np.bool)
    ep_actions = np.zeros((T_MAX, N_ACTIONS))  # probability of actions
    ep_actions_done = np.zeros((T_MAX, N_ACTIONS), dtype=np.bool)  # selected actions

    observation = env.reset()

    # breakout-specific code
    lives = None
    t_memory_start = 0

    # learn or explore or train and play
    if ep % EP_PLAY == 0:
        if ep > 0:
            learn()
        play = True
    else:
        play = False

    #
    prev_frame = np.zeros((SCREEN_LOW_X, SCREEN_LOW_Y), dtype=np.bool)

    for t in range(T_MAX):

        frame = np.array(screen_shrink(observation, SCREEN_LOW_X, SCREEN_LOW_Y, SCREEN_LOW_Z).mean(axis=2) > 0, 
            dtype=np.bool)
        frame_diff = frame - prev_frame
        prev_frame = frame

        # print(frame.astype(np.int))
        # print(frame_diff.astype(np.int))

        frame = frame.reshape(N_INPUT_0)
        frame_diff = frame_diff.reshape(N_INPUT_1)

        if play:  
            env.render()
            time.sleep(PLAY_T_DELAY)

        # 
        [actions] = sess.run([l_output], 
            feed_dict={
                l_input_0:frame.reshape((1, N_INPUT_0)),
                l_input_1:frame_diff.reshape((1, N_INPUT_1)),
                l_better_output: np.zeros((1, N_OUTPUT)),
                decays: np.zeros(1)})

        #
        action = np.random.choice(N_OUTPUT, p=actions.reshape((N_ACTIONS)))

        # breakout-specific
        # breakout game start action
        if t - t_memory_start == 0:
            action = 0

        # save
        ep_observations_0[t, :] = frame
        ep_observations_1[t, :] = frame_diff
        ep_actions[t, :] = actions
        ep_actions_done[t, :] = False
        ep_actions_done[t, action] = True 

        # 
        observation, reward, done, info = env.step(action + ACTION_OFFSET)
        ep_reward += reward
        info_lives = info['ale.lives']
        
        # print("time: {:4d}, info: {}, action: {}, reward: {}, episode reward: {}".format(
        #     t, info, action, reward, ep_reward))

        do_memorize = False

        # got reward
        if reward > 0:
            print("memorize * GOOD * experiences (t#{}~t#{})".format(t_memory_start, t))
            do_memorize = True
            well_done = True

        # check dead or alive
        if lives != info_lives:  # new life
            #
            if t > 0:
                print("memorize - bad  - experiences (t#{}~t#{})".format(t_memory_start, t))
                do_memorize = True
                well_done = False
            #
            lives = info_lives

        if do_memorize:
            memorize(
                ep_observations_0[t_memory_start:t], ep_observations_1[t_memory_start:t], 
                ep_actions[t_memory_start:t], ep_actions_done[t_memory_start:t], well_done)
            t_memory_start = t + 1

        if done or t == T_MAX - 1:
            print("Episode finished for {} rewards at t#{}".format(ep_reward, t))
            print("")
            break

