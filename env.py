import numpy as np
import math
import rospy
import torch
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Header
from gazebo_msgs.msg import ModelState

class ENV(object):
    def __init__(self):
        # initizalization
        rospy.init_node('auto_landing_env', anonymous=True)
        # define publisher
        self.state_pred_pub = rospy.Publisher('pred_states', Float64MultiArray, queue_size=100)
        self.state_true_pub = rospy.Publisher('true_states', Float64MultiArray, queue_size=100)
        # wait for message
        rospy.wait_for_message('/cam_long_array', Float64MultiArray)
        rospy.wait_for_message('/cam_wide_array', Float64MultiArray)        
        rospy.wait_for_message('/marker_gt', Odometry)
        
        ######sh#####
        rospy.wait_for_message('/cam_long_array_time', Header)
        rospy.wait_for_message('/cam_wide_array_time', Header)
        ###########
        
        # define subscriber
        rospy.Subscriber('/cam_long_array', Float64MultiArray, self.callback_long_pose, queue_size=5)        
        rospy.Subscriber('/cam_wide_array', Float64MultiArray, self.callback_wide_pose, queue_size=5)
        rospy.Subscriber('/marker_gt', Odometry, self.callback_true_states, queue_size=5)
        
        ######sh#####
        rospy.Subscriber('/cam_long_array_time', Header, self.callback_long_time, queue_size=5)        
        rospy.Subscriber('/cam_wide_array_time', Header, self.callback_wide_time, queue_size=5)
        self.state_time_pub = rospy.Publisher('pred_time', Header, queue_size=100)
        ###########
        self.ground_vehicle_name = 'my_robot'
        
        self.drone_state = ModelState()
        self.drone_state.model_name = 'iris'
        self.drone_state.reference_frame = 'world'

        self.target_state = ModelState()
        self.target_state.model_name = self.ground_vehicle_name
        self.target_state.reference_frame = 'world'

    # define callback functions        
    def callback_long_pose(self, msg):
        self.long_pose_state = msg.data
        self.long_pose_state = np.asarray(self.long_pose_state)
        
    def callback_wide_pose(self, msg):
        self.wide_pose_state = msg.data
        self.wide_pose_state = np.asarray(self.wide_pose_state)
        
    #####sh#####
    def callback_long_time(self, msg):
        self.long_header = msg

    def callback_wide_time(self, msg):
        self.wide_header = msg    
    ###########
    
    def callback_true_states(self, msg):  
        # 신경망에서 추론된 값을 사용할때는 position.x 값에는 -10을 곱해주고, position.y값에는 -75를 곱해주어야 한다.
        self.true_position = np.zeros([3])
        self.true_vel = np.zeros([3])
        self.true_ori = np.zeros([4])
        self.true_euler = np.zeros([3])
                                    
        self.true_position[0] = msg.pose.pose.position.x / 20.0
        self.true_position[1] = msg.pose.pose.position.y / 20.0
        self.true_position[2] = msg.pose.pose.position.z / -25.0       
        
        self.true_ori[0] = msg.pose.pose.orientation.x
        self.true_ori[1] = msg.pose.pose.orientation.y
        self.true_ori[2] = msg.pose.pose.orientation.z
        self.true_ori[3] = msg.pose.pose.orientation.w
        # transform quaternion to euler
        X,Y,Z = self.quaternion_to_euler(self.true_ori[0], self.true_ori[1], self.true_ori[2],self.true_ori[3])
        # append X, Y, Z to euler
        self.true_euler[0] = X
        self.true_euler[1] = Y
        self.true_euler[2] = Z
        
        self.true_vel[0] = msg.twist.twist.linear.x
        self.true_vel[1] = msg.twist.twist.linear.y
        self.true_vel[2] = msg.twist.twist.linear.z

    def get_states(self):        
        # 갯수가 안맞네?
        self.pose_states = np.stack((self.long_pose_state, self.wide_pose_state))
        self.pose_states = np.reshape(self.pose_states, [-1])
        # normalization
        self.pose_states = self.pose_states
        return self.pose_states
    
    def get_true_states(self):
        self.true_states = np.stack((self.true_position, self.true_vel, self.true_euler))
        self.true_states = np.reshape(self.true_states, [-1])
        return self.true_states

    def quaternion_to_euler(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))
        return X, Y, Z
    
    def PubPredState(self, states):   
        data_to_send = Float64MultiArray()
        data_to_send.data = states.detach().cpu().numpy()
        self.state_pred_pub.publish(data_to_send)
        
    def PubTrueState(self, states):
        data_to_send = Float64MultiArray()
        data_to_send.data = states.detach().cpu().numpy()
        self.state_true_pub.publish(data_to_send)        

def main():
    env = ENV()
    rospy.sleep(2.)
    rate = rospy.Rate(10)
    while True:
        pose_states = env.get_states()
        print('pose_states size:{}'.format(np.shape(pose_states)))
        print('pose_states size:{}'.format((pose_states)))
        true_states = env.get_true_states()
        print('true_states size:{}'.format(np.shape(true_states)))
        print('true_states size:{}'.format((true_states)))
        rate.sleep()

if __name__ == '__main__':
    main()