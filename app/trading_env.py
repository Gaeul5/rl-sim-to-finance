import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, prices, window_size=20, initial_balance=10_000.0):
        super().__init__()
        self.prices = np.array(prices).flatten().astype(np.float32)
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self._returns = np.diff(self.prices) / (self.prices[:-1] + 1e-8)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(window_size+2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def _get_obs(self):
        ret_window = self._returns[self.current_step-self.window_size:self.current_step]
        return np.concatenate([ret_window,
            [float(self.shares_held > 0),
             self.cash_balance / self.initial_balance]]).astype(np.float32)

    def _portfolio_value(self):
        return self.cash_balance + self.shares_held * float(self.prices[self.current_step])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash_balance = self.initial_balance
        self.shares_held = 0
        self.prev_portfolio_value = self.initial_balance
        self.trade_log = []
        return self._get_obs(), {}

    def step(self, action):
        price = float(self.prices[self.current_step])
        if action == 1 and self.cash_balance >= price:
            shares = int(self.cash_balance // price)
            self.shares_held += shares
            self.cash_balance -= shares * price
            self.trade_log.append({"step": self.current_step, "action": "BUY", "price": price})
        elif action == 2 and self.shares_held > 0:
            self.cash_balance += self.shares_held * price
            self.trade_log.append({"step": self.current_step, "action": "SELL", "price": price})
            self.shares_held = 0
        pv = self._portfolio_value()
        reward = (pv - self.prev_portfolio_value) / (self.prev_portfolio_value + 1e-8) * 100.0
        self.prev_portfolio_value = pv
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        return self._get_obs(), float(reward), terminated, False, {"portfolio_value": pv}
