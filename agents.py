import torch
from abc import ABC, abstractmethod
from simplified_game import *

class Agent(ABC):
  """
  plays a turn in the game
  """
  @abstractmethod
  def play_turn(self, game, player_num) -> None:
    pass

class Random_Agent(Agent):
  def play_turn(self, game, player_num, verbose = False) -> bool:
    take_card = torch.randint(0, 2, (len(game),))
    if verbose:
      if take_card[0]:
        print("Player {} took a card".format(player_num))
      else:
        print("Player {} passed".format(player_num))
    handle_decisions(game, player_num, take_card)

class human_Agent(Agent):
  pass