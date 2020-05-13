from gym import make
import pybullet_envs

from hw05_ant.agent import Agent

if __name__ == "__main__":
    env = make("AntBulletEnv-v0")
    env.render(mode='human')

    agent = Agent()

    for i in range(20):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            env.render(mode='human')
        print(total_reward)