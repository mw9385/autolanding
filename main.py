import argparse
import rospy
import torch

from utils import Replay_buffer
from model import Linear_Model, RNN_Model
from env import ENV

from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('--capacity', default=10000, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--eval_period', default=1000, type=int)
parser.add_argument('--model_directory', default='./model/lstm/', type=str)
parser.add_argument('--log_directory', default='./logs/lstm', type=str)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define model
if args.train_mode == 'linear':
    model = Linear_Model(args.state_dim, args.time_step, args.n_hidden, args.n_output)
elif args.train_mode == 'lstm':
    model = RNN_Model(args.state_dim, args.n_hidden, args.n_lstm, args.n_output)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion =  torch.nn.MSELoss()

# get replay buffer
replay_buffer = Replay_buffer(args)
writer = SummaryWriter(args.log_directory)

def main():    
    env = ENV()
    rospy.sleep(2) # 이게 없으면 처음에 데이터가 쌓이지 않아서 에러가 뜬다.
    rate = rospy.Rate(15)   
    # Generate dataset in gazebo environment. Wait until a buffer is filled.
    global_step = 0
    for step in range(int(args.max_step)): 
        if step < args.capacity: 
            print("steps:{}".format(step))
        states = env.get_states()  
        true_states = env.get_true_states()      
        replay_buffer.push((states, true_states))        
        
        if step == args.capacity:
            print('------------------------')
            print('START TRAINING')
            
        if args.capacity < step:
            print('Current Steps:{}'.format(step))        
            # get samples from the buffer
            train_X, train_Y = replay_buffer.sample(args.batch_size)
            # load model to cuda and flatten for the use of network input
            if args.train_mode == 'linear':
                train_X = train_X.to(device).view(-1, args.state_dim * args.time_step)
                train_Y = train_Y.to(device).view(-1, args.n_output)        
            elif args.train_mode == 'lstm':
                train_X = train_X.to(device).view(-1, args.time_step, args.state_dim)
                train_Y = train_Y.to(device).view(-1, args.n_output)

            # update parameters
            optimizer.zero_grad()
            if args.train_mode == 'linear':
                pred_Y = model(train_X)
            elif args.train_mode == 'lstm':
                pred_Y, pred_hidden = model(train_X)
                pred_Y = pred_Y[:, -1, :]
            loss = criterion(pred_Y, train_Y)
            loss.backward()
            optimizer.step()            
            global_step +=1        

            # write down on tensorboard            
            if global_step % args.eval_period == 0:
                print('global_step:{}'.format(global_step))
                print('loss:{}'.format(loss))
                writer.add_scalar('Loss', loss, global_step=global_step)
                torch.save(model.state_dict(), args.model_directory + 'actor.pth')            

        rate.sleep()
if __name__ == '__main__':
    main()
















