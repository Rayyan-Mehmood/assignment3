import numpy as np
import random

# Constants
EMPTY = 0
X = 1
O = -1
DRAW = 0
MAX_PLAYER = X
MIN_PLAYER = O
WIN_REWARD = 100
DRAW_REWARD = 0
LOSS_REWARD = -100

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = X
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = X
        self.winner = None

    def print_board(self):
        symbols = {EMPTY: ' ', X: 'X', O: 'O'}
        for row in self.board:
            print("|".join([symbols[col] for col in row]))
            print("-----")

    def available_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == EMPTY]

    def make_move(self, move):
        if self.board[move[0]][move[1]] == EMPTY:
            self.board[move[0]][move[1]] = self.current_player
            self.current_player = -self.current_player
            self.check_winner()

    def check_winner(self):
        # Check rows, columns, and diagonals for a winner
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != EMPTY:
                self.winner = self.board[i][0]
                return
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != EMPTY:
                self.winner = self.board[0][i]
                return
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != EMPTY:
            self.winner = self.board[0][0]
            return
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != EMPTY:
            self.winner = self.board[0][2]
            return
        # Check for draw
        if len(self.available_moves()) == 0:
            self.winner = DRAW

def switch_player(player):
    return -player

def initial_state():
    return tuple([tuple(row) for row in np.zeros((3, 3), dtype=int)])

def get_state(board):
    return tuple([tuple(row) for row in board])

def random_action(board):
    return random.choice([(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY])

def apply_move(board, move, player):
    new_board = [list(row) for row in board]
    new_board[move[0]][move[1]] = player
    return tuple([tuple(row) for row in new_board])

def evaluate(board, player):
    game = TicTacToe()
    game.board = np.array(board)
    game.check_winner()
    if game.winner == player:
        return WIN_REWARD
    elif game.winner == DRAW:
        return DRAW_REWARD
    elif game.winner == -player:
        return LOSS_REWARD

def q_learning_train(num_episodes, alpha, gamma, start_epsilon, end_epsilon):
    Q1 = {}
    Q2 = {}
    for episode in range(num_episodes):
        print("Episode: ", episode)
        game = TicTacToe()
        state = initial_state()
        done = False
        num_moves = 0
        player_1_turn = True
        epsilon = start_epsilon - (episode / num_episodes) * (start_epsilon - end_epsilon)
        player1_state = initial_state()
        player2_state = initial_state()
        player1_action = (0, 0)
        player2_action = (0, 0)
        while not done:
            num_moves+=1
            if player_1_turn:  # RL agent chooses actions every second iteration
                if np.random.rand() < epsilon:
                    action = random_action(state)
                else:
                    max_q_value = float('-inf')
                    best_action = None
                    for move in game.available_moves():
                        q_value = Q1.get(apply_move(state, move, X), {}).get(move, 0)
                        if q_value > max_q_value:
                            max_q_value = q_value
                            best_action = move
                    action = best_action
            
                next_state = apply_move(state, action, X)
                reward = evaluate(next_state, X)
            
                if state not in Q1:
                    Q1[state] = {}
                if next_state not in Q1:
                    Q1[next_state] = {}

                if reward is not None:
                    reward = reward / num_moves
                    Q1[state][action] = reward
                    Q2[player2_state][player2_action] = -reward
                    done = True
                else:
                    max_next_q = max(Q1.get(next_state, {}).values(), default=0)
                    Q1[state][action] = Q1.get(state, {}).get(action, 0) + alpha * (0 + gamma * max_next_q - Q1.get(state, {}).get(action, 0))
                    player1_state = state
                    player1_action = action
                    state = next_state

                player_1_turn = False
            
            else:
                if np.random.rand() < epsilon:
                    action = random_action(state)
                else:
                    max_q_value = float('-inf')
                    best_action = None
                    for move in game.available_moves():
                        q_value = Q2.get(apply_move(state, move, O), {}).get(move, 0)
                        if q_value > max_q_value:
                            max_q_value = q_value
                            best_action = move
                    action = best_action
            
                next_state = apply_move(state, action, O)
                reward = evaluate(next_state, O)
            
                if state not in Q2:
                    Q2[state] = {}
                if next_state not in Q2:
                    Q2[next_state] = {}

                if reward is not None:
                    reward = reward / num_moves
                    Q2[state][action] = reward
                    Q1[player1_state][player1_action] = -reward
                    done = True
                else:
                    max_next_q = max(Q2.get(next_state, {}).values(), default=0)
                    Q2[state][action] = Q2.get(state, {}).get(action, 0) + alpha * (0 + gamma * max_next_q - Q2.get(state, {}).get(action, 0))
                    player2_state = state
                    player2_action = action
                    state = next_state
                player_1_turn = True
    return Q1, Q2

def q_learning_move(state, Q):
    max_q_value = float('-inf')
    best_action = None
    for move in game.available_moves():
        q_value = Q.get(state, {}).get(move, 0)
        if q_value > max_q_value:
            max_q_value = q_value
            best_action = move
    return best_action

class DefaultOpponent:
    def __init__(self, player):
        self.player = player

    def find_winning_move(self, board):
        # Check rows, columns, and diagonals for a winning move
        for i in range(3):
            if board[i][0] == board[i][1] == self.player and board[i][2] == 0:
                return (i, 2)
            if board[i][1] == board[i][2] == self.player and board[i][0] == 0:
                return (i, 0)
            if board[i][0] == board[i][2] == self.player and board[i][1] == 0:
                return (i, 1)
            if board[0][i] == board[1][i] == self.player and board[2][i] == 0:
                return (2, i)
            if board[1][i] == board[2][i] == self.player and board[0][i] == 0:
                return (0, i)
            if board[0][i] == board[2][i] == self.player and board[1][i] == 0:
                return (1, i)
        if board[0][0] == board[1][1] == self.player and board[2][2] == 0:
            return (2, 2)
        if board[1][1] == board[2][2] == self.player and board[0][0] == 0:
            return (0, 0)
        if board[0][0] == board[2][2] == self.player and board[1][1] == 0:
            return (1, 1)
        if board[0][2] == board[1][1] == self.player and board[2][0] == 0:
            return (2, 0)
        if board[1][1] == board[2][0] == self.player and board[0][2] == 0:
            return (0, 2)
        if board[0][2] == board[2][0] == self.player and board[1][1] == 0:
            return (1, 1)
        return None

    def find_blocking_move(self, board):
        opponent = 1 if self.player == -1 else -1
        return self.find_winning_move(board) if self.player == 1 else self.find_winning_move(
            [[-1 if cell == 1 else 1 if cell == -1 else 0 for cell in row] for row in board])

    def random_move(self, board):
        available_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
        return random.choice(available_moves) if available_moves else None

    def select_move(self, board):
        winning_move = self.find_winning_move(board)
        if winning_move:
            return winning_move
        blocking_move = self.find_blocking_move(board)
        if blocking_move:
            return blocking_move
        return self.random_move(board)

def minimax(board, player):
    if evaluate(board) is not None:
        return None, evaluate(board)

    if player == MAX_PLAYER:
        max_eval = float('-inf')
        best_move = None
        for move in TicTacToe().available_moves():
            new_board = board.copy()
            new_board[move[0]][move[1]] = MAX_PLAYER
            _, eval = minimax(new_board, switch_player(player))
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return best_move, max_eval
    else:
        min_eval = float('inf')
        best_move = None
        for move in TicTacToe().available_moves():
            new_board = board.copy()
            new_board[move[0]][move[1]] = MIN_PLAYER
            _, eval = minimax(new_board, switch_player(player))
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return best_move, min_eval

def minimax_with_alpha_beta(board, player, alpha, beta):
    if evaluate(board) is not None:
        return None, evaluate(board)

    if player == MAX_PLAYER:
        max_eval = float('-inf')
        best_move = None
        for move in TicTacToe().available_moves():
            new_board = board.copy()
            new_board[move[0]][move[1]] = MAX_PLAYER
            _, eval = minimax_with_alpha_beta(new_board, switch_player(player), alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return best_move, max_eval
    else:
        min_eval = float('inf')
        best_move = None
        for move in TicTacToe().available_moves():
            new_board = board.copy()
            new_board[move[0]][move[1]] = MIN_PLAYER
            _, eval = minimax_with_alpha_beta(new_board, switch_player(player), alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return best_move, min_eval

# Example usage
if __name__ == "__main__":
    game = TicTacToe()

    # Training Q-learning agent as player 1 
    Q_player1, Q_player2 = q_learning_train(num_episodes=50000, alpha=0.3, gamma=0.99, start_epsilon=0.99, end_epsilon=0.99)

    # Default opponent
    default_opponent = DefaultOpponent(player=X)
    # Playing against Q-learning agent as player 1
    game.reset()

    num_games = 200
    num_p1_wins = 0
    num_p2_wins = 0
    num_draws = 0
    for i in range(0,num_games):
        print(i)
        game.reset()
        while game.winner is None:
            if game.current_player == O:
                action = q_learning_move(get_state(game.board), Q_player2)
            else:
                # action = random_action(game.board)  # random opponent
                action = default_opponent.select_move(game.board)
            game.make_move(action)
            # game.print_board()
            # print()
        if game.winner == X:
            num_p1_wins += 1
        elif game.winner == O:
            num_p2_wins += 1
        else:
            num_draws += 1
    
    print("Player 1 Wins: ", num_p1_wins, "\nPlayer 2 Wins: ", num_p2_wins, "\nDraws", num_draws)
        
