import torch
import torch.nn as nn

from simplified_game import *
from agents import *
from dqn_agent import *

dqn_agent = torch.load('dqn_agent.pt')
dqn_agent.mode_eval = True

players = [dqn_agent, Random_Agent(), Random_Agent(), Random_Agent()]
win_counts = play_games(players, 1, verbose=True)
print(win_counts)