# Random search idea from:
# http://kvfrans.com/simple-algoritms-for-solving-cartpole/

import gym
import gym.wrappers
import numpy as np
import time
import sys

#
float_formatter = lambda x: "%.6f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

#
# https://github.com/openai/gym/wiki/CartPole-v0
#
env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'tmp/cartpole-experiment-1', force=True)
#
T_MAX = 200

#
t_best = 0

for ep in range(200):

    observation = env.reset()

    # params = np.random.random(4) * 2 - 1

    # params found with this algorithm which sometimes ran 200 timesteps
    # params = np.array([0.134581, -0.831711, 0.888523, 0.221410])

    # params found with this algorithm which always(?) ran 200 timesteps
    params = np.array([-0.043532, 0.372516, 0.324205, 0.902193])

    print("ep: {}".format(ep))
    print("params: {}".format(params))

    t = 0

    while True:

        t += 1

        env.render()

        # print("observation: {}".format(observation))
        action = 1 if np.dot(observation, params) > 0 else 0
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t))
            time.sleep(1)
            break

    if t > t_best:
        print("Good! Episode ran more than previous best ({}) timesteps".format(
            t_best))
        t_best = t

    if t == T_MAX:
        break
