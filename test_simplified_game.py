import torch
import unittest

from simplified_game import *
from agents import *

class test_game_simplified(unittest.TestCase):
  def test_initialize_game(self):
      # Test case 1: Initialize 1 game
      num_games = 1
      expected_output = torch.zeros((num_games, 38))
      expected_output[:, 1:num_players+1] = starting_chips
      real_output = initialize_games(num_games)
      assert torch.equal(real_output, expected_output)

      # Test case 2: Initialize 3 games
      num_games = 3
      expected_output = torch.zeros((num_games, 38))
      expected_output[:, 1:num_players+1] = starting_chips
      real_output = initialize_games(num_games)
      assert torch.equal(real_output, expected_output)

  def test_flip_card(self):
    # Test case 1: 1 game, flip card
    num_games = 1
    games = initialize_games(num_games)
    flip_cards(games)
    num_1 = len(torch.where(card_locations(games) == -1)[0])
    num_0 = len(torch.where(card_locations(games) == 0)[0])
    assert(num_1 == 1 and num_0 == 32)

    # Test case 2: 100 games. Flag games 50-60 as completed. Games 60-70 already have a flipped card.
    num_games = 100
    games = initialize_games(num_games)
    games[50:60, -1] += 1
    card_locations(games)[60:70, 0] = -1 # flip first card

    flip_cards(games)
    cards = card_locations(games)
    for i in range(num_games):
      num_1 = len(torch.where(cards[i] == -1)[0])
      num_0 = len(torch.where(cards[i] == 0)[0])
      if i >= 50 and i < 60:
        assert(num_1 == 0 and num_0 == 33)
      elif i >= 60 and i < 70:
        assert(num_1 == 1 and num_0 == 32)
        assert(cards[i][0] == -1)
      else:
        assert(num_1 == 1 and num_0 == 32)

  def test_handle_decisions_take(self):
    # Test case 1: 1 game, take card
    num_games = 1
    game = initialize_games(num_games)
    flip_cards(game)
    chip_counts(game)[:, 0] = 1
    player_num = 1
    take_card = torch.tensor([True])
    card_num = torch.where(card_locations(game) == -1)
    handle_decisions(game, player_num, take_card)

    expected_game = initialize_games(num_games)
    card_locations(expected_game)[card_num] = player_num

    # Check that only one element differs (this accounts for the random card flip)
    differing_indices = torch.where(game != expected_game)
    assert len(differing_indices[0]) == num_games and len(torch.unique(differing_indices[0])) == num_games # check that only one element differs per game
    assert game[differing_indices] == torch.ones(num_games) * -1 # check that the differing element is -1

  def test_handle_decisions_pass(self):
    # Test case 2: 1 game, pass
    num_games = 1
    game = initialize_games(num_games)
    flip_cards(game)
    chip_counts(game)[:, 0] = 0
    player_num = 1
    take_card = torch.tensor([False])

    expected_game = game.clone()
    chip_counts(expected_game)[0, player_num] -= 1

    handle_decisions(game, player_num, take_card)

    # Check that only one element differs (this accounts for the random card flip)
    assert torch.equal(game, expected_game)

  def test_handle_decisions_out_of_chips(self):
    num_games = 1
    game = initialize_games(num_games)
    flip_cards(game)
    player_num = 1
    chip_counts(game)[:, player_num] = 0
    take_card = torch.tensor([False])
    card_num = torch.where(card_locations(game) == -1)
    handle_decisions(game, player_num, take_card)

    expected_game = initialize_games(num_games)
    card_locations(expected_game)[card_num] = player_num
    chip_counts(expected_game)[0, player_num] = 0

    # Check that only one element differs (this accounts for the random card flip)
    differing_indices = torch.where(game != expected_game)
    assert len(differing_indices[0]) == num_games and len(torch.unique(differing_indices[0])) == num_games # check that only one element differs per game
    assert game[differing_indices] == torch.ones(num_games) * -1 # check that the differing element is -1

  def test_calc_scores(self):
      import numpy as np

      # make game in numpy format then convert to torch tensor
      game_np = np.array([np.concatenate((np.arange(0, 5), np.repeat(np.arange(1, 5), 9)))[:39]])
      game = torch.tensor(game_np, dtype=torch.float)

      # Calculate the expected score using PyTorch operations
      expected_score = torch.reshape(torch.tensor([
          torch.sum(torch.arange(0, 9)),
          torch.sum(torch.arange(9, 18)),
          torch.sum(torch.arange(18, 27)),
          torch.sum(torch.arange(27, 33))
      ]), (1, 4))

      res = calc_scores(game)
      assert torch.equal(res, expected_score)

  def test_update_over_flag(self):
    # Test case 1: 1 game, flip card
    num_games = 1
    games = initialize_games(num_games)
    flip_cards(games)
    update_over_flag(games)
    assert torch.equal(over_flags(games), torch.zeros((num_games,)))

    # Test case 2: 25 games. Game 0 has been completed. Games 1-25 have [34 - index] card left. Last game should be marked as completed.
    num_games = 25
    games = initialize_games(num_games)
    games[0, -1] += 1
    for i in range(1, num_games):
      card_locations(games)[i, :i] = 1
    update_over_flag(games)

    expected = torch.zeros((num_games,))
    expected[0] = 1
    expected[-1] = 1
    flip_cards(games)
    assert torch.equal(over_flags(games), expected)

  def test_play_games(self):
    num_games = 10000
    players = [Random_Agent(), Random_Agent(), Random_Agent(), Random_Agent()]
    win_counts = play_games(players, num_games, verbose=False)
    print(win_counts)

if __name__ == '__main__':
  with torch.no_grad():
    #test_game_simplified().test_update_over_flag()
    test_game_simplified().test_play_games()