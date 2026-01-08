import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime

class SingleStockTradingEnv(gym.Env):
    """
    单股票连续仓位交易环境（含仓位状态 + 开盘成交逻辑）
    """

    def __init__(
        self,
        df,
        lookback_n=20,
        initial_cash=1_000_000,
        max_drawdown_penalty=1.0,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.lookback_n = lookback_n
        self.initial_cash = initial_cash
        self.max_drawdown_penalty = max_drawdown_penalty

        # ---------- 动作空间 ----------
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ---------- 状态空间 ----------
        # n 天 OHLCV + 当前仓位
        obs_dim = lookback_n * 5 + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.env_file_name = f"EvnOneStock_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv"
        pd.DataFrame(columns=["reward","terminated","truncated","portfolio_value","position","drawdown","current_step","prev_close","open_price","close_price","prev_position","delta_position"]).to_csv(self.env_file_name, index=False, header=True, mode="w")
        
        self._reset_internal_state()

    def _reset_internal_state(self):
        self.current_step = self.lookback_n
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.max_portfolio_value = self.initial_cash

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal_state()
        return self.get_observation(), {}

    # ---------- observation ----------
    def get_observation(self):
        price_window = self.df.loc[
            self.current_step - self.lookback_n : self.current_step - 1,
            ["open", "high", "low", "close", "volume"],
        ].values.flatten()

        obs = np.concatenate(
            [price_window, np.array([self.position])]
        )

        return obs.astype(np.float32)

    # ---------- step ----------
    def step(self, action):
        target_position = float(np.clip(action[0], 0.0, 1.0))
        prev_position = self.position

        open_price = self.df.loc[self.current_step, "open"]
        close_price = self.df.loc[self.current_step, "close"]
        prev_close = self.df.loc[self.current_step-1, "close"]

        # ---------- 收益计算 ----------
        portfolio_return = 0.0

        # 1️⃣ 原有仓位：昨日收盘 → 今日收盘
        portfolio_return += prev_position * (
            close_price - prev_close
        ) / prev_close

        # 2️⃣ 仓位变化部分：今日开盘 → 今日收盘
        delta_position = target_position - prev_position

        if delta_position > 0:
            # 买入
            portfolio_return += delta_position * (
                close_price - open_price
            ) / open_price
        elif delta_position < 0:
            # 卖出：减少仓位，等价于少亏 / 少赚
            portfolio_return += delta_position * (
                close_price - open_price
            ) / open_price

        # ---------- 更新资产 ----------
        self.portfolio_value *= (1.0 + portfolio_return)
        self.position = target_position

        # ---------- 回撤 ----------
        self.max_portfolio_value = max(
            self.max_portfolio_value, self.portfolio_value
        )
        drawdown = (
            self.max_portfolio_value - self.portfolio_value
        ) / self.max_portfolio_value

        reward = portfolio_return - self.max_drawdown_penalty * drawdown

        # ---------- done ----------
        terminated = self.portfolio_value < 0.5 * self.initial_cash
        truncated = self.current_step >= len(self.df)

        

        info = {
            "reward":[reward],
            "terminated":terminated,
            "truncated":truncated,
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "drawdown": drawdown,
            "current_step":self.current_step,
            "prev_close":prev_close,
            "open_price":open_price,
            "close_price":close_price,
            "prev_position":prev_position,
            "delta_position":delta_position,
        }

        pd.DataFrame(info).to_csv(self.env_file_name, index=False, header=False, mode="a")

        self.current_step += 1
        
        return self.get_observation(), reward, terminated, truncated, info

    def render(self):
        print(
            f"Step={self.current_step}, "
            f"Value={self.portfolio_value:.2f}, "
            f"Position={self.position:.2f}"
        )
