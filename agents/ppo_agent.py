import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor


class RewardLoggerCallback(BaseCallback):
    """에피소드 종료마다 포트폴리오 가치를 기록하는 콜백."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


class PPOTradingAgent:
    """stable-baselines3 PPO를 트레이딩 환경에 맞게 래핑한 클래스."""

    def __init__(
        self,
        env_fn,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        verbose: int = 0,
        device: str = "auto",
    ):
        self.env = DummyVecEnv([lambda: Monitor(env_fn())])
        self.model = PPO(
            policy,
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=verbose,
            device=device,
        )
        self.reward_callback = RewardLoggerCallback()

    def train(self, total_timesteps: int = 100_000):
        self.model.learn(total_timesteps=total_timesteps, callback=self.reward_callback)

    def predict(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path, env=self.env)

    def evaluate(self, env, n_episodes: int = 5) -> list[float]:
        """env 위에서 n_episodes 실행 후 에피소드별 포트폴리오 최종 가치를 반환."""
        portfolio_values = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = self.predict(obs)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            portfolio_values.append(info.get("portfolio_value", 0.0))
        return portfolio_values
