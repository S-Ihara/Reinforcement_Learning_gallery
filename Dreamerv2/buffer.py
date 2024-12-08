import numpy as np 

class ReplayBuffer:
    """
    リプレイバッファ
    """
    def __init__(self, capacity, state_shape, action_size, seq_len, batch_size):
        """
        args:
            capacity (int): メモリのサイズ
            state_shape (tuple[int]): 状態の次元数
            action_size (int): 行動の次元数
            seq_len (int): シーケンスの長さ
            batch_size (int): バッチサイズ
        """

        self.capacity = capacity
        self.state_shape = state_shape
        self.action_size = action_size
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.point = 0
        self.full = False

        self.states = np.empty((self.capacity, *self.state_shape), dtype=np.uint8)
        self.action = np.empty((self.capacity, self.action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.terminal = np.empty((self.capacity, 1), dtype=bool)
    
    def add(self, state, action, reward, terminal):
        """
        args:
            state (np.array): 状態
            action (np.array): 行動
            reward (float): 報酬
            terminal (bool): 終了判定
        """

        self.states[self.point] = state
        self.action[self.point] = action
        self.reward[self.point] = reward
        self.terminal[self.point] = terminal

        self.point = (self.point + 1) % self.capacity
        self.full = self.full or self.point == 0
    
    def _sample_idx(self, L):
        """
        args:
            L (int): シーケンスの長さ
        returns:
            np.array: ランダムな位置からLsteps分のインデックス
        """
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.capacity if self.full else self.point - L)
            idxs = np.arange(idx, idx + L) % self.capacity
            valid_idx = not self.point in idxs[1:] 
        return idxs

    def _retrieve_batch(self, idxs, n, l):
        """
        args:
            idxs (np.array): インデックス
            n (int): バッチサイズ
            l (int): シーケンスの長さ
        returns:
            tuple: 状態、行動、報酬、終了判定
        """
        vec_idxs = idxs.transpose().reshape(-1)
        state = self.states[vec_idxs]

        return state.reshape(l, n, *self.state_shape), self.action[vec_idxs].reshape(l, n, -1), self.reward[vec_idxs].reshape(l, n), self.terminal[vec_idxs].reshape(l, n)

    def _shift_sequences(self, states, actions, rewards, terminals):
        """
        args:
            states (np.array): 状態
            actions (np.array): 行動
            rewards (np.array): 報酬
            terminals (np.array): 終了判定
        note:
            tに対してそれぞれ, s_{t+1}, a_{t}, r_{t}, d_{t}を返す
        """
        states = states[1:]
        actions = actions[:-1]
        rewards = rewards[:-1]
        terminals = terminals[:-1]

        return states, actions, rewards, terminals

    def sample(self):
        """
        returns:
            tuple: 状態、行動、報酬、終了判定
        notes: shape=(Length, Batch, *Shape)
        """
        n = self.batch_size
        l = self.seq_len
        states,actions,rewards,terminals = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        states,actions,rewards,terminals = self._shift_sequences(states, actions, rewards, terminals)

        states = states.astype(np.float32) / 255.0

        return states, actions, rewards, terminals



if __name__ == "__main__":
    pass 