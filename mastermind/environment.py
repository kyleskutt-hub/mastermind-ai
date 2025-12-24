import numpy as np
from typing import Tuple, Optional

class MastermindEnv:
    def __init__(self, code_length: int = 4, num_colors: int = 6, max_guesses: int = 10):
        self.code_length = code_length
        self.num_colors = num_colors
        self.max_guesses = max_guesses
        self.secret_code = None
        self.guess_history = []
        self.feedback_history = []
        self.current_step = 0
        
    def reset(self) -> np.ndarray:
        self.secret_code = np.random.randint(0, self.num_colors, size=self.code_length)
        self.guess_history = []
        self.feedback_history = []
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        guess = self._decode_action(action)
        black_pegs, white_pegs = self._evaluate_guess(guess)
        self.guess_history.append(guess)
        self.feedback_history.append((black_pegs, white_pegs))
        self.current_step += 1
        done = black_pegs == self.code_length or self.current_step >= self.max_guesses
        reward = self._calculate_reward(black_pegs, white_pegs, done)
        return self._get_observation(), reward, done, {"black": black_pegs, "white": white_pegs}
    
    def _decode_action(self, action: int) -> np.ndarray:
        guess = []
        for _ in range(self.code_length):
            guess.append(action % self.num_colors)
            action //= self.num_colors
        return np.array(guess)
    
    def _evaluate_guess(self, guess: np.ndarray) -> Tuple[int, int]:
        black_pegs = sum(g == s for g, s in zip(guess, self.secret_code))
        secret_counts = {}
        guess_counts = {}
        for i in range(self.code_length):
            if guess[i] != self.secret_code[i]:
                secret_counts[self.secret_code[i]] = secret_counts.get(self.secret_code[i], 0) + 1
                guess_counts[guess[i]] = guess_counts.get(guess[i], 0) + 1
        white_pegs = sum(min(secret_counts.get(c, 0), guess_counts.get(c, 0)) for c in guess_counts)
        return black_pegs, white_pegs
    
    def _calculate_reward(self, black: int, white: int, done: bool) -> float:
        if black == self.code_length:
            return 10.0 - self.current_step * 0.5
        if done:
            return -5.0
        return black * 0.5 + white * 0.1 - 0.1
    
    def _get_observation(self) -> np.ndarray:
        obs_size = self.max_guesses * (self.code_length + 2)
        obs = np.zeros(obs_size, dtype=np.float32)
        for i, (guess, feedback) in enumerate(zip(self.guess_history, self.feedback_history)):
            start = i * (self.code_length + 2)
            obs[start:start + self.code_length] = guess / self.num_colors
            obs[start + self.code_length] = feedback[0] / self.code_length
            obs[start + self.code_length + 1] = feedback[1] / self.code_length
        return obs
    
    @property
    def observation_size(self) -> int:
        return self.max_guesses * (self.code_length + 2)
    
    @property
    def action_size(self) -> int:
        return self.num_colors ** self.code_length
