from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class StockTradingEnv(gym.Env):
    """
    基于 Gymnasium 标准定制的股票强化学习环境
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, lookback_n=20, initial_balance=1000000, mdd_penalty=2.0, red_line=-0.2,
                 evn_file="EvnSingleStock"):
        super(StockTradingEnv, self).__init__()

        # 1. 整理数据
        self.df = df.sort_values('trade_date').reset_index(drop=True)
        self.lookback_n = lookback_n
        self.initial_balance = initial_balance
        self.mdd_penalty = mdd_penalty
        self.red_line = red_line

        # 提取特征列名（排除日期等非数值列）
        self.features = ['adj_preclose', 'adj_open', 'adj_high',
                         'adj_low', 'adj_close', 'pctchange', 'adj_volumne', 'amount', 'adj_avgprice', 'tradestatuscode'
                         ]

        # 2. 定义动作空间：单一连续值 [0, 1]，代表仓位
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # 3. 定义观测空间：(lookback_n, 字段数量)
        # 注意：实际使用时建议对特征进行归一化，归一化在 get_observation 中处理
        obs_dim = self.lookback_n * len(self.features) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 状态变量初始化
        self.current_step = 0
        self.position = 0.0
        self.net_worths = []
        self.max_net_worth = initial_balance

        self.env_file_name = f"./env_log/{evn_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(
            columns=['date', 'target_position', 'actual_position', 'adj_old_position', 'adj_open', 'adj_close',
                     'adj_preclose', 'high', 'low', 'limit_price', 'stop_price', 'tradestatuscode', 'daily_return',
                     'current_net_worth', 'max_net_worth', 'mdd', 'total_return', 'reward', 'terminated',
                     'truncated', ]).to_csv(
            self.env_file_name, index=False, header=True, mode="w")

    def reset(self, seed=None, options=None):
        # 按照 Gymnasium 标准处理 seed
        super().reset(seed=seed)

        self.current_step = self.lookback_n
        self.position = 0.0
        self.net_worths = [self.initial_balance]
        self.max_net_worth = self.initial_balance

        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        # 截取当前时刻前 lookback_n 天的数据特征
        obs = self.df.loc[self.current_step - self.lookback_n: self.current_step - 1, self.features].reset_index(
            drop=True)

        adj_preclose_value = obs["adj_preclose"].values[0]
        for col_i in ['adj_preclose', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_avgprice']:
            obs[col_i] = obs[col_i].values / adj_preclose_value

        for col_i in ['pctchange', 'adj_volumne', 'amount']:
            obs[col_i] = (max(obs[col_i]) - obs[col_i]) / (max(obs[col_i]) - min(obs[col_i]))

        obs = obs.values.flatten()

        result = np.concatenate([obs, np.array([self.position])])

        return result.astype(np.float32)

    def step(self, action):
        # Gymnasium 传入的 action 通常是 array，取第一个元素
        target_position = float(action[0])

        # 获取当前行情（第 current_step 行）
        pre_row = self.df.iloc[self.current_step - 1]
        pre_adj_close = pre_row['adj_close']
        pre_adj_open = pre_row['adj_open']
        row = self.df.iloc[self.current_step]
        adj_open = row['adj_open']
        adj_close = row['adj_close']
        adj_preclose = row['adj_preclose']
        high = row['high']
        low = row['low']
        limit_price = row['limit']
        stop_price = row['stopping']
        tradestatuscode = row['tradestatuscode']

        # 涨跌停判定 (基于开盘价)
        is_limit_up = low >= limit_price
        is_limit_down = high <= stop_price

        # --- 交易执行逻辑 ---
        old_position = self.position

        # 如果没有涨跌停，则允许调仓
        if tradestatuscode == 0:
            actual_position = old_position
        elif target_position > old_position:
            if is_limit_up:
                actual_position = old_position
            else:
                actual_position = target_position
        elif target_position < old_position:
            if is_limit_down:
                actual_position = old_position
            else:
                actual_position = target_position
        else:
            actual_position = old_position

        adj_old_position = (pre_adj_close / pre_adj_open * old_position) / (
                pre_adj_close / pre_adj_open * old_position + 1 - old_position)

        # --- 收益计算逻辑 ---
        # 调仓日收益 = 旧仓位在(昨收-今开)的变动 + 新仓位在(今开-今收)的变动
        pre_trade_ret = (adj_open / adj_preclose)
        post_trade_ret = (adj_close / adj_open)

        daily_return = (pre_trade_ret * adj_old_position + 1 - adj_old_position) * (
                    post_trade_ret * actual_position + 1 - actual_position) - 1

        # 更新净值
        current_net_worth = self.net_worths[-1] * (1 + daily_return)
        self.net_worths.append(current_net_worth)
        self.position = actual_position

        # --- 计算最大回撤 (MDD) ---
        if current_net_worth > self.max_net_worth:
            self.max_net_worth = current_net_worth

        mdd = (current_net_worth - self.max_net_worth) / self.max_net_worth if self.max_net_worth != 0 else 0

        # --- 奖励函数 ---
        # 累计收益
        total_return = (current_net_worth / self.initial_balance) - 1

        # 长期奖励考虑最大回撤：回撤越大，奖励衰减越快
        reward = total_return + (mdd * self.mdd_penalty)  # 这里的系数2.0可根据对风险的厌恶程度调整

        # --- 更新步数与结束判定 ---
        self.current_step += 1
        # terminated: 触发清盘线，退出。
        terminated = mdd < self.red_line
        # truncated: 异常中断（此处暂无）
        truncated = self.current_step >= len(self.df) - 1

        observation = self._get_observation()
        info = {
            "date": [row['trade_date']],
            "target_position": target_position,
            "actual_position": actual_position,
            'adj_old_position': adj_old_position,
            'adj_open': adj_open,
            'adj_close': adj_close,
            'adj_preclose': adj_preclose,
            'high': high,
            'low': low,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'tradestatuscode': tradestatuscode,
            'daily_return': daily_return,
            'current_net_worth': current_net_worth,
            'max_net_worth': self.max_net_worth,
            'mdd': mdd,
            'total_return': total_return,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        pd.DataFrame(info).to_csv(self.env_file_name, index=False, header=False, mode="a")

        return observation, reward, terminated, truncated, info

    def render(self):
        # 可视化净值或打印
        print(f"Step: {self.current_step}, Net Worth: {self.net_worths[-1]:.2f}, Position: {self.position:.2f}")
