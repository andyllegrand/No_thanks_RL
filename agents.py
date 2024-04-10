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
    probabilities = torch.tensor([0.75, 0.25])  # 75% chance for 0, 25% chance for 1
    take_card = torch.multinomial(probabilities, len(game), replacement=True)
    if verbose:
      if take_card[0]:
        print("Player {} took a card".format(player_num))
      else:
        print("Player {} passed".format(player_num))
    handle_decisions(game, player_num, take_card)

class Always_Pass_Agent(Agent):
  def play_turn(self, game, player_num, verbose = False) -> bool:
    take_card = torch.zeros(len(game))
    if verbose:
      print("Player {} passed".format(player_num))
    handle_decisions(game, player_num, take_card)

class Human_Agent(Agent):
  def play_turn(self, game, player_num, verbose = False) -> bool:
    assert len(game) == 1
    take_card = torch.tensor([int(input("Take the card? (1 for yes, 0 for no): "))])
    handle_decisions(game, player_num, take_card)
