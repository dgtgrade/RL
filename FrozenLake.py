import gym
import numpy as np
import time
import os

env = gym.make('FrozenLake8x8-v0')

#
np.set_printoptions(linewidth=np.nan, threshold=np.nan, formatter={'float_kind': lambda x: "%6.4f" % x})

# Q[s,a]
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.01  # learning rate
gamma = 0.990  # decay factor
ep = 0

for ep in range(10000000):

    o = env.reset()

    r = None
    t = 0

    for t in range(200):

        s0 = o

        if ep > 0 and ep % 100000 == 0:
            a = np.argmax(Q[s0])
            time.sleep(1)
        else:
            a = env.action_space.sample()
        # a = 2

        a0 = a

        o, r, d, i = env.step(a)

        if ep % 100000 == 0:
            os.system('cls')
            print(ep, t)
            env.render()

        s1 = o
        Q[s0, a0] += alpha * (r + gamma * np.max(Q[s1, :]) - Q[s0, a0])

        if ep % 100000 == 0:
            print("a:{}, o:{}, r:{}, d:{}, i:{}".format(a, o, r, d, i))
            print(Q)

        if d:
            break

