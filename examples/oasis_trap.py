import DeceptiveGym
import numpy
import time

if __name__ == "__main__":

    n_envs = 128

    n_actions = 5
    envs = DeceptiveGym.OasisTrap(n_envs)


    print(envs.observation_shape, envs.actions_count)

    fps = 0

    steps = 0
    while steps < 250000:
        actions = numpy.random.randint(0, n_actions, (n_envs, ))
        
        time_start = time.time()
        states, rewards, dones, infos = envs.step(actions)
        time_stop = time.time()

        dt = time_stop - time_start

        fps = 0.9*fps + 0.1/dt


        dones_idx = numpy.where(dones)[0]

        envs.render()
        
        for n in dones_idx:
            
            envs.reset(n)

            print(steps, round(fps, 2), infos[n])

        steps+= 1

