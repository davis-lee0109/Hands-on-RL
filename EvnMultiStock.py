import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class MultiStockTradingEnv(gym.Env):
    """
    多股票强化学习交易环境（共享资金池）
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
            self,
            df: pd.DataFrame,
            stock_list,
            lookback_n=20,
            initial_balance=1.0,
            cost_rate=0.001,
            mdd_penalty=0.2,
            max_position_per_stock=0.3,
            training=True,
    ):
        super().__init__()

        self.df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        self.stock_list = stock_list
        self.n_stock = len(stock_list)
        self.lookback_n = lookback_n
        self.initial_balance = initial_balance
        self.cost_rate = cost_rate
        self.mdd_penalty = mdd_penalty
        self.max_position_per_stock = max_position_per_stock
        self.training = training

        self.features = [
            "adj_open", "adj_close", "adj_preclose",
            "adj_high", "adj_low", "pctchange",
            "volume", "amount"
        ]

        # === 构造 [date x stock x feature] ===
        self.dates = sorted(self.df.trade_date.unique())
        self.T = len(self.dates)

        self.data = {
            d: self.df[self.df.trade_date == d]
            .set_index("ts_code")
            .loc[self.stock_list]
            for d in self.dates
        }

        # === action: 每只股票一个权重 ===
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_stock,),
            dtype=np.float32
        )

        obs_dim = self.n_stock * self.lookback_n * len(self.features) + self.n_stock + 2

        self.observation_space = spaces.Box(
            low=-5,
            high=5,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.reset()

    # ================= reset =================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.training:
            self.current_step = np.random.randint(
                self.lookback_n, self.T - 2
            )
        else:
            self.current_step = self.lookback_n

        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance

        self.weights = np.zeros(self.n_stock)  # 当前仓位
        self.history_values = [self.initial_balance]

        return self._get_observation(), {}

    # ================= observation =================
    def _get_observation(self):
        obs = []

        for i in range(self.current_step - self.lookback_n, self.current_step):
            day_data = self.data[self.dates[i]]
            price_scale = day_data["adj_preclose"].values.reshape(-1, 1)

            feat = day_data[self.features].values / price_scale
            obs.append(feat.flatten())

        obs = np.concatenate(obs)

        mdd = (self.portfolio_value - self.max_portfolio_value) / self.max_portfolio_value

        obs = np.concatenate([
            obs,
            self.weights,
            np.array([self.portfolio_value, mdd])
        ])

        return obs.astype(np.float32)

    # ================= step =================
    def step(self, action):
        # --- 目标权重处理 ---
        target_w = np.clip(action, 0, self.max_position_per_stock)
        if target_w.sum() > 1:
            target_w = target_w / target_w.sum()

        prev_day = self.data[self.dates[self.current_step - 1]]
        curr_day = self.data[self.dates[self.current_step]]

        # --- 股票收益 ---
        stock_ret = (
                curr_day["adj_close"].values /
                prev_day["adj_close"].values - 1
        )

        # --- 组合收益 ---
        gross_ret = np.dot(self.weights, stock_ret)

        # --- 交易成本 ---
        turnover = np.sum(np.abs(target_w - self.weights))
        cost = self.cost_rate * turnover

        daily_return = gross_ret - cost

        # --- 更新净值 ---
        self.portfolio_value *= (1 + daily_return)
        self.history_values.append(self.portfolio_value)

        self.weights = target_w

        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value

        # --- 奖励函数 ---
        mdd = (self.portfolio_value - self.max_portfolio_value) / self.max_portfolio_value

        reward = (
                daily_return
                + self.mdd_penalty * mdd
        )

        reward = np.clip(reward, -5, 5)

        # --- step 推进 ---
        self.current_step += 1

        terminated = self.current_step >= self.T - 1
        truncated = False

        info = {
            "portfolio_value": self.portfolio_value,
            "daily_return": daily_return,
            "turnover": turnover,
            "mdd": mdd
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        print(
            f"Step {self.current_step} | "
            f"Value {self.portfolio_value:.4f} | "
            f"MDD {(self.portfolio_value / self.max_portfolio_value - 1):.2%}"
        )
