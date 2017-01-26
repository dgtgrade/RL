# Follows a random search idea from:
# http://kvfrans.com/simple-algoritms-for-solving-cartpole/

# import gym
import gym.wrappers
import numpy as np
import time

#
float_formatter = lambda x: "%+.6f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

#
# https://github.com/openai/gym/wiki/MountainCar-v0
#
env = gym.make('MountainCar-v0')
# env = gym.wrappers.Monitor(env, 'tmp/mountaincar-experiment-1', force=True)

#
EP_MAX = 1000
T_MAX = 100

#
r_best = None
params_best = np.zeros(3)


#
def add_bias(w):
    return np.append([1], w)

for ep in range(EP_MAX):

    # noinspection PyRedeclaration
    observation = env.reset()

    # use only velocity
    params = np.array([0, 0, 1000])

    # random with bias
    # params = (np.random.random(3) - 0.5) * 10

    print("ep: {}".format(ep))
    print("params: {}".format(params))

    t = 0
    r = 0

    while True:

        t += 1

        env.render()

        k = np.dot(add_bias(observation), params)
        action = min(max(int(k), 0), 2)

        print("time: {:3d}, observation: {}, action: {}".format(t, observation, action))

        #
        observation, reward, done, info = env.step(action)
        r += reward

        if done:
            print("Episode finished after {} timesteps".format(t))
            time.sleep(1)
            break

    if r_best is None or r > r_best:
        print("Good! Episode got more reward ({}) than previous one ({})".format(r, r_best))
        r_best = r
