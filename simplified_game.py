import torch

"""
Stores simplified game logic. Written in torch tensors for gpu acceleration.

Simplified game rules:
Same as ordinary no thanks except:
- turns dont chain (easier to train many games in parallel)
- no card strings when scoring (significantly reduces complexity)
- always 4 players
- ai agent is always player 1
- card values are 1-33

Game state stored in vector. Index key:
[0]:    Chip count on current card
[1:5]:  Player chip counts
[5:39]: Card locations. (0: in deck, 1-4: belonging to that player)
[39]: Over flag
"""

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

num_players = 4

min_card_num = 3
max_card_num = 35

cards_left_out = 9

starting_chips = 9

seed = 0
#torch.manual_seed(seed)

# lambdas:
chip_counts     = lambda games: games[:, 0:num_players + 1]
card_locations  = lambda games: games[:, num_players + 1:-1]
over_flags      = lambda games: games[:, -1]
active_games    = lambda games: games[torch.where(over_flags(games) == 0)]
over_games      = lambda games: games[torch.where(over_flags(games) == 1)]

def initialize_games(num_games):
  game = torch.zeros((num_games, num_players + max_card_num - min_card_num + 3)).to(device)
  game[:, 1:num_players+1] = starting_chips
  return game

def flip_cards(games):
  """
  Simulates drawing the top card of the shuffled deck for each game. If a game already has a flipped over card does nothing.
  Does not affect over games.
  """
  def flip_card(game_cards):
    indices = torch.where(game_cards == 0)
    random_index = torch.randint(0, indices[0].shape[0], (1,))
    game_cards[indices[0][random_index]] = -1

  with torch.no_grad():
    flags = over_flags(games)
    cards = card_locations(games)
    for i in range(len(games)):
      if flags[i] == 0 and len(torch.where(cards[i] == -1)[0]) == 0:
        flip_card(cards[i])

def handle_decisions(games, player_num, take_card):
  with torch.no_grad():
    # find games where player is out of chips, and update decisions
    out_of_chips = torch.where(chip_counts(games)[:, player_num] == 0)[0]
    if len(out_of_chips) > 0:
      take_card[out_of_chips] = True

    # divide games into take and pass games
    take_indices = torch.where(take_card)[0]
    pass_indices = torch.where(torch.logical_not(take_card))[0]

    # Apply 'take' operations
    if len(take_indices) > 0:
      #chip_counts(games)[take_indices, player_num] += chip_counts(games)[take_indices, 0]
      chip_counts(games)[take_indices, 0] = 0
      flipped_ind = torch.where((card_locations(games) == -1))[1]
      card_locations(games)[take_indices, flipped_ind[take_indices]] = player_num

    # Apply 'pass' operations
    chip_counts(games)[pass_indices, player_num] -= 1

    # draw new card if necessary
    flip_cards(games)

def update_over_flag(games):
  """
  updates the over flag for each game. The game is over when the number of 0s is less than or equal to cards_left_out.
  """
  with torch.no_grad():
    cards = card_locations(games)
    num_zeros = torch.sum(cards == 0, axis=1)
    over_flags(games)[torch.where(num_zeros <= cards_left_out)] = 1

def calc_scores(games):
  """
  returns tensor of shape num games x num players containing the scores for each player in each game.
  """
  with torch.no_grad():
    game_cards = card_locations(games)
    ret = torch.zeros((game_cards.shape[0], num_players), dtype=int)
    for i, gc in enumerate(game_cards):
      for player in range(1, num_players + 1):
          ret[i, player - 1] = torch.sum(torch.where(gc == player)[0])
    return ret

def prepare_state_for_player(games, player_num):
  """
  "rotates" the game state so that the player is in the first position. This is done by shifting the player's chip count to index 1 while 
  retaining the order of the other players. The card locations are also shifted so that the active card is in the first position.
  """
  ret = torch.copy(games)

  # rotate player chip counts
  torch.roll(chip_counts(ret), -player_num, axis=0)

  # rotate card locations (leave 0 and -1 in place)
  card_locs = card_locations(ret)
  card_locs[card_locs > 0] += player_num
  card_locs[card_locs > 0] %= num_players

  return ret

def play_games(players, num_games, verbose = False):
  games = initialize_games(num_games)
  flip_cards(games)
  num_players = len(players)
  win_counts = torch.zeros(num_players)
  turn_count = 0

  # loop through playing turns. All games are synchronized to allow for vectorization
  while True:
    player_num = turn_count % num_players + 1
    #player_games = prepare_state_for_player(games, player_num)
    player_games = games

    if verbose:
      assert num_games == 1
      print(f"Turn {turn_count}:")
      print(f"player {player_num}'s turn!\n")
      for i in range(1, num_players + 1):
        print(f"Player {i} tokens: {chip_counts(player_games)[:, i]}")
        print(f"Player {i} cards: {torch.where(card_locations(player_games) == i)[1]}\n")

      print(f"Active card: {torch.where(card_locations(player_games) == -1)[1]}\n")
      print(f"Remaining cards: {torch.where(card_locations(player_games) == 0)[1]}\n")

    # call agent handle decisions
    player = players[player_num - 1]
    player.play_turn(player_games, player_num, verbose = verbose)
    turn_count += 1

    # update over flags
    update_over_flag(games)

    # check for and score games that are over
    o_games = over_games(games)
    if o_games.shape[0] > 0:
      scores = calc_scores(o_games)
      winners = torch.argmin(scores, axis=1)
      for w in winners:
        win_counts[w] += 1

    # check if all games are over
    games = active_games(games)
    if games.shape[0] == 0:
      break

  return win_counts