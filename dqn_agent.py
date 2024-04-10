import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from simplified_game import *
from agents import *

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 40)
        self.layer2 = nn.Linear(40, 20)
        self.layer3 = nn.Linear(20, n_actions)

    def forward(self, x):
        x.to(device)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

class DQN_Agent(Agent):
  def __init__(self,
                batch_size,
                buffer_size,
                gamma,
                eps_start,
                eps_end,
                eps_decay,
                tau,
                lr,
               ):
    self.memory = ReplayMemory(buffer_size) # initial state, action, next state, reward
    self.policy_net         = DQN(num_players + max_card_num - min_card_num + 3, 2).to(device)
    self.target_net         = DQN(num_players + max_card_num - min_card_num + 3, 2).to(device)
    self.player_num         = 1
    self.optimizer          = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

    self.batch_size = batch_size
    self.gamma = gamma
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_decay = eps_decay
    self.tau = tau

    self.mode_eval = False

    self.steps_done = 0

  def optimize_policy_model(self):
    if len(self.memory) < self.batch_size:
        return
    transitions = self.memory.sample(self.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch   = torch.stack(batch.state).to(device)
    action_batch  = torch.stack(batch.action).to(device)
    reward_batch  = torch.stack(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch).gather(1, action_batch.long().unsqueeze(1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(self.batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

  def update_target_model(self):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
    self.target_net.load_state_dict(target_net_state_dict)

  def select_decisons(self, games):
    self.steps_done += 1

    # network outputs
    network_output = self.policy_net.forward(games)
    decisions = network_output[:, 0] < network_output[:, 1].to(device)

    if self.mode_eval:
      return decisions
    
    # epsilon greedy
    sample = torch.rand(decisions.shape[0]).to(device)
    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        math.exp(-1. * self.steps_done / self.eps_decay)
    decisions[sample > eps_threshold] = torch.rand(decisions[sample > eps_threshold].shape).to(device) > 0.5

    return decisions


  def play_turn(self, games, player_num, verbose = False) -> bool:
    if self.mode_eval:
       with torch.no_grad():
         decisions = self.select_decisons(games)
         handle_decisions(games, player_num, decisions)
         if verbose:
           if decisions[0]:
             print(f"Player {player_num} took a card")
           else:
             print(f"Player {player_num} passed")
         return

    with torch.no_grad():
      old_games = games.clone()
      old_scores = calc_scores(games)
      decisions = self.select_decisons(games)

      handle_decisions(games, player_num, decisions)
      new_scores = calc_scores(games)
      new_games = games.clone()
      #score_deltas = -1 * (new_scores - old_scores)[:, player_num - 1]
      score_deltas = -1 * (new_scores - old_scores)[:, player_num - 1]

    # split the game into individual games, add to memory
    for old_game, new_game, decision, sd in zip(old_games, new_games, decisions, score_deltas):
        self.memory.push(old_game, decision, new_game, sd)

    self.optimize_policy_model()
    self.update_target_model()

if __name__ == "__main__":
  # Hyperparameters
  BATCH_SIZE = 1
  BUFFER_SIZE = 10000
  GAMMA = 0.99
  EPS_START = 0.99
  EPS_END = 0.05
  EPS_DECAY = 1000
  TAU = 0.01
  LR = 1e-4

  num_episodes = 500
  games_per_episode_train = 1000
  games_per_episode_eval = 1000


  dqn_agent = DQN_Agent(BATCH_SIZE, BUFFER_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)

  players = [dqn_agent, Random_Agent(), Random_Agent(), Random_Agent()]
  winrates = []
  dqn_agent.mode_eval = True
  win_counts = play_games(players, games_per_episode_eval, verbose = False)
  dqn_agent.mode_eval = False
  winrate = win_counts[0] / games_per_episode_eval
  print(win_counts)
  winrates.append(winrate)
  for i_episode in range(num_episodes):
      print(f"Episode {i_episode}")

      # play games to update the model
      play_games(players, games_per_episode_train, verbose = False)

      # play games to evaluate the model
      dqn_agent.mode_eval = True
      win_counts = play_games(players, games_per_episode_eval, verbose = False)
      dqn_agent.mode_eval = False
      winrate = win_counts[0] / games_per_episode_eval
      print(win_counts)
      winrates.append(winrate)

  # save the model
  torch.save(dqn_agent, "dqn_agent.pt")

