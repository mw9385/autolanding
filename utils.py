import numpy as np
import torch

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, args):
        self.storage = []
        self.args = args        
        self.ptr = 0
        
    
    # replay buffer에 데이터가 꽉 찬 경우를 대비하는 것 같음
    def push(self, data):
        if len(self.storage) == self.args.capacity:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.args.capacity
        else:
            self.storage.append(data)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage)- self.args.time_step, size = batch_size) # 0부터 storage 길이 범위에서 batch_size 갯수 만큼만 뽑아서 사용
        total_states, total_true_states = [], []
        for i in ind:            
            # s = state, a = action, r = reward, ns = next_state, d = done
            states, true_states = [], []                
            for j in range(self.args.time_step):
                # get states
                s, ts = self.storage[i+j]                                
                s = torch.from_numpy(s).type(torch.float32)
                ts = torch.from_numpy(ts).type(torch.float32)
                states.append(s)
                if j == self.args.time_step - 1:
                    true_states.append(ts)
            states = torch.stack(states, dim=0)
            true_states = torch.stack(true_states, dim = 0)

            total_states.append(states)
            total_true_states.append(true_states)
        total_states = torch.stack(total_states, dim=0) #size = [batch_size, time_step, input_length]
        total_true_states = torch.stack(total_true_states, dim= 0).squeeze(1) # size = [batch_size, position vector + velocity vector (6)]  
        return total_states, total_true_states
