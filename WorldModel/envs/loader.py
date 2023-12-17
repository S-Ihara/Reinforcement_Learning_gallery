from bisect import bisect
from pathlib import Path

from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np

class _RolloutDataset(torch.utils.data.Dataset):
    def __init__(self,root,transform,buffer_size: int=200,train: bool=True):
        self._transform = transform
        root_dir = Path(root)
        self._files = list(root_dir.glob("*.npz"))
        assert len(list(self._files)) > 0, f"no files found in {root_dir}"

        size = len(self._files)
        if train:
            self._files = list(self._files)[:int(size*0.8)]
        else:
            self._files = list(self._files)[int(size*0.8):]

        self._buffer_size = buffer_size
        self._buffer_index = 0
        self._buffer = None
        self._cum_size = None

        print(f"Loaded {len(list(self._files))} files")
    
    def load_next_buffer(self):
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(total=len(self._buffer_fnames),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with np.load(f) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] +
                                   self._data_per_sequence(data['rewards'].shape[0])]
            pbar.update(1)
        pbar.close()
    
    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self,idx):
        # binary search through cum_size
        file_index = bisect(self._cum_size, idx) - 1
        seq_index = idx - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self,data,seq_index):
        pass

    def _data_per_sequence(self, data_length):
        pass

class RolloutSequenceDataset(_RolloutDataset):
    """ Encapsulates rollouts.

    Args:
        root (Union[str,Path]): root directory of data sequences
        seq_len (int): number of timesteps extracted from each rollout
        transform (Callable): transformation of the observations
        buffer_size (int): number of sequences loaded at once
        train (bool): if True, train data, else test
    Note:
        loaderからデータを取り出すと
        (batch_size,seq_len,channel,height,width)の形になる
    """
    def __init__(self, root, seq_len, transform, buffer_size=200, train=True):
        super().__init__(root, transform, buffer_size, train)
        self._seq_len = seq_len

    def _get_data(self, data, seq_index):
        obs_data = data['observations'][seq_index:seq_index + self._seq_len + 1]
        observations = []
        for img in obs_data:
            img = self._transform(img)
            observations.append(img)
        obs_data = torch.stack(observations)
        #obs_data = self._transform(obs_data.astype(np.float32))
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][seq_index+1:seq_index + self._seq_len + 1]
        action = action.astype(np.float32)
        reward, terminal = [data[key][seq_index+1:
                                      seq_index + self._seq_len + 1].astype(np.float32)
                            for key in ('rewards', 'dones')]
        # data is given in the form
        # (obs, action, reward, terminal, next_obs)
        return obs, action, reward, terminal, next_obs

    def _data_per_sequence(self, data_length):
        return data_length - self._seq_len
    
class RolloutObservationDataset(_RolloutDataset):
    """ Encapsulates rollouts.
    Args:
        root (Union[str,Path]): root directory of data sequences
        seq_len (int): number of timesteps extracted from each rollout
        transform (Callable): transformation of the observations
        train (bool): if True, train data, else test
    Note:
        loaderからデータを取り出すと
        (batch_size,channel,height,width)の形になる
    """
    def _data_per_sequence(self, data_length):
        return data_length

    def _get_data(self, data, seq_index):
        return self._transform(data['observations'][seq_index])

if __name__ == "__main__":
    """test"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    #dataset = RolloutSequenceDataset("data", 100, transform,train=True)
    dataset = RolloutObservationDataset("data",transform,train=True)
    loader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False)
    print(len(dataset))
    
    data = next(iter(loader))
    print(data.shape)
    # batch size数のdataをそれぞれ可視化してみて確認する
    import matplotlib.pyplot as plt
    for i in range(32):
        plt.imshow(data[i].permute(1,2,0))
        plt.pause(0.01)
        plt.clf()