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
parser.add_argument('--state_dim', default=24, type=int)
parser.add_argument('--time_step', default=6, type=int)
parser.add_argument('--n_output', default=9, type=int)
parser.add_argument('--n_hidden', default=256, type=int)
parser.add_argument('--n_lstm', default=64, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--max_step', default=1000000, type=int)
parser.add_argument('--capacity', default=7, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--eval_period', default=1, type=int)
parser.add_argument('--model_directory', default='./model/lstm/', type=str)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load model
directory = args.model_directory + 'actor.pth'
# define model
if args.train_mode == 'linear':
    model = Linear_Model(args.state_dim, args.time_step, args.n_hidden, args.n_output)
elif args.train_mode == 'lstm':
    model = RNN_Model(args.state_dim, args.n_hidden, args.n_lstm, args.n_output)
model.load_state_dict(torch.load(directory))
model = model.to(device)
print("Model has been loaded.")

# get replay buffer
replay_buffer = Replay_buffer(args)

def main():    
    env = ENV()
    rospy.sleep(2.)
    rate = rospy.Rate(15)   
    position_matrix = []
    velocity_matrix = []
    euler_matrix = []
    
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
            print("Current Steps:{}".format(step))                            
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
            position_error[:, 0] = position_error[:, 0] * -10.0
            position_error[:, 2] = position_error[:, 2] * -75.0
            p_error = torch.pow(position_error[:, 0], 2) + torch.pow(position_error[:, 1], 2)
            velocity_error = pred_Y[:, 3:6] - test_Y[:, 3:6]
            v_error = torch.pow(velocity_error[:, 0], 2) + torch.pow(velocity_error[:, 1], 2)
            euler_error = pred_Y[:, 6:] - test_Y[:, 6:]
            e_error = torch.pow(euler_error[:, 0], 2) + torch.pow(euler_error[:, 1], 2)
            
            # append error
            position_matrix.append(torch.sqrt(p_error.mean()))
            velocity_matrix.append(torch.sqrt(v_error.mean()))            
            euler_matrix.append(torch.sqrt(e_error.mean()))
            global_step +=1  
            
            # write down on tensorboard            
            if global_step % args.eval_period == 0:
                position_matrix = torch.stack(position_matrix, dim=0)
                velocity_matrix = torch.stack(velocity_matrix, dim=0)
                euler_matrix = torch.stack(euler_matrix, dim=0)
                
                print('global_step:{}'.format(global_step))        
                print("Position Error:{}, Velocity Error:{}, Euler Error:{}".format(torch.mean(position_matrix), torch.mean(velocity_matrix), torch.mean(euler_matrix)))
                                                             
                print("Prediction:{}".format(pred_Y[0]))
                print("True:{}".format(test_Y[0]))
                position_matrix = []
                velocity_matrix = []
                euler_matrix = []

                # publish
                env.PubPredState(pred_Y[0])
                env.PubTrueState(test_Y[0])
                
                ######sh#######
                long_time=env.long_header.stamp.to_sec()+1e-9*env.long_header.stamp.to_nsec()
                wide_time=env.wide_header.stamp.to_sec()+1e-9*env.wide_header.stamp.to_nsec()
                print({"long_time":long_time})
                print({"wide_time":wide_time})
                
                if long_time > wide_time:
                    time=env.long_header
                else:
                    time=env.wide_header
                
                env.state_time_pub.publish(time)

        rate.sleep()
if __name__ == '__main__':
    main()
#=======================================================#    
#                       오차 계산                         #
#=======================================================#
# - linear model
# - testing env: sinusoidal하게 움직이는 상황 (140 * 15= 2100, 한 주기 동안 평가함)
# - error: 
#       - Position Error:0.20948487520217896
#       - Velocity Error:0.16029773652553558 
#       - Euler Error:0.3437506854534149

# ------------------------------------------------------#
# - lstm model
# - testing env: sinusoidal하게 움직이는 상황 (140 * 15= 2100, 한 주기 동안 평가함)
# - error: 
#       - Position Error:0.16542430222034454
#       - Velocity Error:0.07910799235105515
#       - Euler Error:0.11627911031246185

# 정지 고도 상황 / landing pad는 움직이고 있음
# - 3m
#       - linear:
#           - Position Error:1.3765994310379028
#           - Velocity Error:0.6093629002571106 
#           - Euler Error:1.0849817991256714
#       -   lstm:
#           - Position Error:0.4114551246166229
#           - Velocity Error:0.17357929050922394
#           - Euler Error:2.11881685256958
# - 20m
#       - linear:
#               - Position Error:0.11212673038244247
#               - Velocity Error:0.08339332044124603 
#               - Euler Error:0.1438852995634079
#       -   lstm:
#               - Position Error:0.23287321627140045 
#               - Velocity Error:0.20646965503692627
#               - Euler Error:0.36421042680740356
# - 75m
#       - linear: 
#           - Position Error:0.22082877159118652
#           - Velocity Error:0.08135247230529785
#           - Euler Error:1.504388451576233
#       -   lstm:
#           - Position Error:1.8251227140426636
#           - Velocity Error:0.08852539211511612
#           - Euler Error:0.07126738131046295









