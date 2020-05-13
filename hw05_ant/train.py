from copy import deepcopy

from gym import make
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import deque
import pybullet_envs
import wandb
wandb.init(project="rl-cheetah")

GAMMA = 0.99
CRITIC_LR = 3e-4
ACTOR_LR = 3e-4
CLIP = 0.1
ENTROPY_COEF = 1e-2
TRAJECTORY_SIZE = 512
N_STEP = 6

def transform_state(state):
    return torch.tensor(state)

class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        #self.state_buffer = deque(maxlen=buffer_size)
        #self.action_buffer = deque(maxlen=buffer_size)
        #self.next_state_buffer = deque(maxlen=buffer_size)
        #self.reward_buffer = deque(maxlen=buffer_size)
        #self.done_buffer = deque(maxlen=buffer_size)
        self.state_buffer = torch.tensor([])
        self.action_buffer = torch.tensor([])
        self.next_state_buffer = torch.tensor([])
        self.reward_buffer = torch.tensor([])
        self.done_buffer = torch.tensor([], dtype=torch.bool)
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.state_buffer)

    def append(self, transition):
        state, action, next_state, reward, done = transition
        if state.shape[0] < self.buffer_size:
            self.state_buffer = torch.cat([self.state_buffer, state.reshape(1, -1)])
            self.action_buffer = torch.cat([self.action_buffer, action.reshape(1, -1)])
            self.next_state_buffer = torch.cat([self.next_state_buffer, next_state.reshape(1, -1)])
            self.reward_buffer = torch.cat([self.reward_buffer, torch.tensor([reward])])
            self.done_buffer = torch.cat([self.done_buffer, torch.tensor([done])])
        else:
            self.state_buffer = torch.cat([self.state_buffer[1:], state.reshape(1, -1)])
            self.action_buffer = torch.cat([self.action_buffer[1:], action.reshape(1, -1)])
            self.next_state_buffer = torch.cat([self.next_state_buffer[1:], next_state.reshape(1, -1)])
            self.reward_buffer = torch.cat([self.reward_buffer[1:], torch.tensor([reward])])
            self.done_buffer = torch.cat([self.done_buffer[1:], torch.tensor([done])])

        #self.state_buffer.append(state)
        #self.action_buffer.append(action)
        #self.next_state_buffer.append(next_state)
        #self.reward_buffer.append(reward)
        #self.done_buffer.append(done)

    def sample(self, sample_size=64):
        p = np.random.choice(len(self.state_buffer), size=sample_size, replace=False)
        state_sample = self.state_buffer[p]
        action_sample = self.action_buffer[p]
        next_sample_sample = self.next_state_buffer[p]
        reward_sample = self.reward_buffer[p].reshape(sample_size, -1)
        done_sample = self.done_buffer[p].reshape(sample_size, -1)
        return state_sample, action_sample, next_sample_sample, reward_sample, done_sample

"""class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.actor = None # Torch model
        self.critic = None # Torch model

    def update(self, trajectory):
        state, action, rollouted_reward = zip(*trajectory)

    def get_value(self, state):
        # Should return expected value of the state
        return 0

    def act(self, state):
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        return [0., 0., 0., 0., 0., 0.]

    def save(self):
        torch.save(self.actor, "agent.pkl")"""

class TD3:
    def __init__(self, state_dim, action_dim): #, action_discr):
        self.gamma = GAMMA

        # TODO: change to 32
        hidden_size = 256

        #self.actor = nn.Sequential(
        #    nn.Linear(state_dim, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, action_dim),
        #    nn.Tanh()
        #)
        self.actor = torch.load('agent.pkl')
        #self.critic1 = nn.Sequential(
        #    nn.Linear(state_dim + action_dim, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, 1)
        #)
        self.critic1 = torch.load('critic1.pkl')
        #self.critic2 = nn.Sequential(
        #    nn.Linear(state_dim + action_dim, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, 1)
        #)
        self.critic2 = torch.load('critic2.pkl')
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)
        self.target_actor = deepcopy(self.actor)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.optim_critic = optim.Adam([*self.critic1.parameters(), *self.critic2.parameters()], lr=CRITIC_LR)
        self.mse_loss1 = nn.MSELoss()
        self.mse_loss2 = nn.MSELoss()
        self.tau = 0.05
        self.policy_noise = 0.1
        self.total_it = 0

    def q(self, state, action, target=False):
        input = torch.cat([state, action], dim=1)
        if target:
            return self.target_critic1(input), self.target_critic2(input)
        return self.critic1(input), self.critic2(input)

    def prob(self, state, action):
        t = (action[0] - self.actor(state) * 2) * 2 #[discr_action]
        return torch.exp(-t ** 2 / 2) / np.sqrt(2 * np.pi * 0.5)

    def update(self, transition):
        self.total_it += 1
        state, action, next_state, reward, done = transition

        with torch.no_grad():
            noise = torch.randn_like(action) * 0.2
            noise = noise.clamp(-0.5, 0.5)

            next_action = self.act(next_state, target=True).detach() + noise
            next_action = next_action.clamp(-1, 1)

            target_q1, target_q2 = self.q(next_state, next_action, target=True)
            target_q = self.gamma * torch.min(target_q1, target_q2) * (done.logical_not()) + reward

        # Critic update
        q1, q2 = self.q(state, action)
        delta = self.mse_loss1(q1, target_q) + self.mse_loss2(q2, target_q)
        wandb.log({"Delta": delta})
        self.optim_critic.zero_grad()
        delta.backward()#retain_graph=True)
        self.optim_critic.step()

        if self.total_it % 2 == 0:
            # Actor update
            best_action = self.act(state)
            q1, _ = self.q(state, best_action)
            value = -q1.mean()
            wandb.log({"Value": value})
            self.optim_actor.zero_grad()
            value.backward()
            self.optim_actor.step()

            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    #def update_target(self):
    #    self.target_critic1 = deepcopy(self.critic1)
    #    self.target_critic2 = deepcopy(self.critic2)

    def act(self, state, target=False):
        if target:
            return self.target_actor(state)
        return self.actor(state)

    def save(self, path):
        torch.save(self.actor, path)
        wandb.save(path)
        torch.save(self.critic1, 'critic1.pkl')
        torch.save(self.critic2, 'critic2.pkl')
        wandb.save('critic1.pkl')
        wandb.save('critic2.pkl')


if __name__ == "__main__":
    env = make("AntBulletEnv-v0")


    algo = TD3(state_dim=28, action_dim=8)
    episodes = 10000
    tqdm_episodes = tqdm(range(episodes))
    rb = ReplayBuffer()
    eps = 0.1
    #eps_decay = 0.99

    episodes_to_start = 50
    for i in tqdm_episodes:
        state = transform_state(env.reset())
        total_reward = 0
        done = False
        steps = 0
        while not done:
            if steps > 1010:
                break
            with torch.no_grad():
                action = algo.act(state)
            if i < episodes_to_start:
                action =  env.action_space.sample()
            else:
                action = (action + torch.randn_like(action) * eps * 1).clamp(-1, 1).detach().numpy()
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            rb.append((state, torch.tensor(action, dtype=torch.float32), next_state, reward, done))
            state = next_state

            if i >= episodes_to_start:
                algo.update(rb.sample(256))
            steps += 1
        #algo.update_target()
        #eps *= eps_decay

        wandb.log({"Total reward": total_reward})

        tqdm_episodes.set_description(f'Total reward={total_reward}')
        algo.save("agent.pkl")



