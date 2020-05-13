import numpy as np
import os
import torch
#from .train import transform_state

def transform_state(state):
    return torch.tensor(state)

class Agent:
    def __init__(self):
        self.actor = torch.load(os.path.join(__file__[:-8], "agent.pkl"))
        pass

    def act(self, state):
        state = transform_state(state)
        action = self.actor(state).detach().numpy()
        return list(action)

    def reset(self):
        pass

