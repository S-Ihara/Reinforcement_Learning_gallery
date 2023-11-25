import numpy as np

# class SimpleReplayBuffer(object):
#     """
#     経験（遷移情報）を貯めておくクラス
#     """
#     def __init__(self,state_shape: tuple, action_shape: int, size: int=100000):
#         """
#         Args:
#             state_shape (tuple): 状態の次元数　画像なら(3,84,84)とか、ベクトルなら(4,)とか
#             action_shape (int): 行動の次元数
#             size (int): リプレイバッファの最大数
#         Note:
#             行動の次元数は離散値なら確実に1
#             メモリをかなり食うので画像などを貯めておくならdtypeをuint8などにする
#         """
#         self.size        = size
#         self.states      = np.empty((self.size,*state_shape),dtype=np.float32)
#         self.actions     = np.empty((self.size,action_shape),dtype=np.float32)
#         self.rewards     = np.empty((self.size,1),dtype=np.float32)
#         self.next_states = np.empty((self.size,*state_shape),dtype=np.float32)
#         self.dones       = np.empty((self.size,1),dtype=np.float32)
#         self.point       = 0
#         self.full        = False

#     def __len__(self):
#         return self.size if self.full else self.point

#     def store(self,state,action,reward,next_state,done):
#         """trajectoryを貯める関数
#         Args:
#             state (np.ndarray): 状態
#             action (int): 行動
#             reward (Union[int,float]): 報酬
#             next_state (np.ndarray): 次の状態
#             done (bool): 終了判定
#         """
#         #state,action,reward,next_state,done = trajectory
#         np.copyto(self.states[self.point],state)
#         np.copyto(self.actions[self.point],action)
#         np.copyto(self.rewards[self.point],reward)
#         np.copyto(self.next_states[self.point],next_state)
#         np.copyto(self.dones[self.point],done)

#         self.point = (self.point+1) % self.size
#         self.full = self.full or self.point == 0

#     def get_minibatch(self,batch_size: int):
#         """バッファからbatch_size分trajectoryを得る関数
#         Args:
#             batch_size (int): いくつtrajectoryを取り出すか
#         Returns:
#             tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]
#         """
#         idexs = np.random.randint(0,
#                                   self.size if self.full else self.point,
#                                   size=batch_size)

#         states = self.states[idexs]
#         actions = self.actions[idexs]
#         rewards = self.rewards[idexs]
#         next_states = self.next_states[idexs]
#         dones = self.dones[idexs]

#         return states, actions, rewards, next_states, dones


# データセットバッファ周りどうしようか
# 実装がちょっと大変そうなので、とりあえず貪欲な実装をする
from collections import deque
import numpy as np
class SequenceBuffer:
    def __init__(self,max_episode_size: int = 1000):
        self.episodes = deque(maxlen=max_episode_size)
        self.size = 0
        self.max_episode_size = max_episode_size
    
    def get_batch(self,batch_size: int, context_length: int):
        """
        Args:
            batch_size int: バッチサイズ
            context_length int: コンテキスト長
        Returns:
            tuple: (states,actions,rewards,dones,timesteps) ((B,L,observation_space),(B,L,1),(B,L,1),(B,L,1),(B,1,1))
        """
        episode_idxs = np.random.randint(0,self.size,size=batch_size)
        # それぞれのエピソードで連続したL個分のデータを取り出す
        for episode_idx in episode_idxs:
            episode = self.episodes[episode_idx]
            states,actions,rewards,dones = episode
            episode_length = len(episode)
            idx = np.random.randint(0,episode_length - context_length) if episode_length - context_length > 0 else 0 # start index
            
            states = states[idx:idx+context_length]
            actions = actions[idx:idx+context_length]
            rewards = rewards[idx:idx+context_length]
            dones = dones[idx:idx+context_length]
            timesteps = idx

        