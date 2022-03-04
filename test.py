import argparse
from turtle import pos
import numpy as np
import rospy
import torch

from utils import Replay_buffer
from model import Linear_Model, RNN_Model
from env import ENV

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode='train' or 'test'
parser.add_argument('--train_mode', default='lstm', type=str)
parser.add_argument('--state_dim', default=36, type=int)
parser.add_argument('--time_step', default=6, type=int)
parser.add_argument('--n_output', default=6, type=int)
parser.add_argument('--n_hidden', default=256, type=int)
parser.add_argument('--n_lstm', default=64, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--max_step', default=1000000, type=int)
parser.add_argument('--capacity', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--eval_period', default=2400, type=int)
parser.add_argument('--model_directory', default='./model/lstm/', type=str)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
directory = args.model_directory + 'actor.pth'
# define model
if args.train_mode == 'linear':
    model = Linear_Model(args.state_dim, args.time_step, args.n_hidden, args.n_output)
elif args.train_mode == 'lstm':
    model = RNN_Model(args.state_dim, args.n_hidden, args.n_lstm, args.n_output)
model.load_state_dict(torch.load(directory))
model = model.to(device)

# get replay buffer
replay_buffer = Replay_buffer(args)

def main():    
    env = ENV()
    rospy.sleep(2.)
    rate = rospy.Rate(15)   
    position_matrix = []
    velocity_matrix = []
    # env.reset() 
    # Generate dataset in gazebo environment. Wait until a buffer is filled.
    global_step = 0
    for step in range(args.max_step): 
        if step < args.capacity: print("steps:{}".format(step))
        states = env.get_states()  
        true_states = env.get_true_states()      
        replay_buffer.push((states, true_states))        
        
        if step == args.capacity:
            print('------------------------')
            print('START Testing')
        if args.capacity < step:                                
            # get samples from the buffer
            test_X, test_Y = replay_buffer.sample(args.batch_size)
            # load model to cuda and flatten for the use of network input
            if args.train_mode == 'linear':
                test_X = test_X.to(device).view(-1, args.state_dim * args.time_step)
                test_Y = test_Y.to(device).view(-1, args.n_output)        
            elif args.train_mode == 'lstm':
                test_X = test_X.to(device).view(-1, args.time_step, args.state_dim)
                test_Y = test_Y.to(device).view(-1, args.n_output)

            # update parameters            
            if args.train_mode == 'linear':
                pred_Y = model(test_X)
            elif args.train_mode == 'lstm':
                pred_Y, pred_hidden = model(test_X)
                pred_Y = pred_Y[:, -1, :]            
            position_error = pred_Y[:, :3] - test_Y[:, :3]
            p_error = torch.pow(position_error[:, 0], 2) + torch.pow(position_error[:, 1], 2)
            velocity_error = pred_Y[:, 3:] - test_Y[:, 3:]
            v_error = torch.pow(velocity_error[:, 0], 2) + torch.pow(velocity_error[:, 1], 2)
            # append error
            position_matrix.append(torch.sqrt(p_error.mean()))
            velocity_matrix.append(torch.sqrt(v_error.mean()))            
            global_step +=1        
            # write down on tensorboard            
            if global_step % args.eval_period == 0:
                position_matrix = torch.stack(position_matrix, dim=0)
                velocity_matrix = torch.stack(velocity_matrix, dim=0)
                print('global_step:{}'.format(global_step))        
                print("Position Error:{}, Velocity Error:{}".format(torch.mean(position_matrix), torch.mean(velocity_matrix)))   
                                                             
                print("Prediction:{}".format(pred_Y[0]))
                print("True:{}".format(test_Y[0]))
                position_matrix = []
                velocity_matrix = []

        rate.sleep()
if __name__ == '__main__':
    main()

# 오차 산출 방식
#   - 오차 측정 방식: RMSE로 추정함 
#   - 고도에 따른 차이가 있을 수 있음: 고도 변화는 5m ~ 35m 사이
#   - 측정치 취득 주기는 15hz로 고정한 상태
#   - 추정값: 위치 및 선형 속도 (x,y,z)의 3차원 값을 추정함

# LSTM test 결과 (1200개의 측정치에 대해서):
#   - 위치 오차: 0.06 ~ 0.1 사이
#   - 속도 오차: 0.07 ~ 0.09 사이

# Linear test 결과 (1200개의 측정치에 대해서):
#   - 위치 오차: 0.069
#   - 속도 오차: 0.075


# Linear model
# 고도 75m, target 정지 상태, 1주기 동안 측정
#   - 위치 오차: 3.84m --> 아직 학습 하지 않은 상태라 나쁘게 나옴
#   - 속도 오차: 0.09m/s
# 고도 20m, target 정지 상태, 1주기 동안 측정
#   - 위치 오차: 0.245m
#   - 속도 오차: 0.067m/s
# 고도 3m, target 정지 상태, 1주기 동안 측정
#   - 위치 오차: 0.046m
#   - 속도 오차: 0.029m/s

# LSTM model
# 고도 75m, target 정지 상태, 1주기 동안 측정
#   - 위치 오차: 0.820m 
#   - 속도 오차: 0.133m/s
# 고도 20m, target 정지 상태, 1주기 동안 측정
#   - 위치 오차: 0.09m
#   - 속도 오차: 0.029m/s
# 고도 3m, target 정지 상태, 1주기 동안 측정
#   - 위치 오차: 0.1076m
#   - 속도 오차: 0.025m/s








