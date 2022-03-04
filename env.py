import math
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class ENV(object):
    def __init__(self):
        # initizalization
        rospy.init_node('auto_landing_env', anonymous=True)
        # define publisher
        self.state_pub = rospy.Publisher('states', Float64MultiArray, queue_size=100)
        # wait for message
        rospy.wait_for_message('/measurement_topic/long_focal_pixel_list', Float64MultiArray)
        rospy.wait_for_message('/measurement_topic/long_focal_pose_list', Float64MultiArray)
        rospy.wait_for_message('/measurement_topic/wide_focal_pixel_list', Float64MultiArray)        
        rospy.wait_for_message('/measurement_topic/wide_focal_pose_list', Float64MultiArray)
        rospy.wait_for_message('/groundTruth_topic/gt_position', Float64MultiArray)
        rospy.wait_for_message('/groundTruth_topic/gt_velocitiy', Float64MultiArray)
        
        # define subscriber
        rospy.Subscriber('/measurement_topic/long_focal_pixel_list', Float64MultiArray, self.callback_long_pixel, queue_size=5)
        rospy.Subscriber('/measurement_topic/long_focal_pose_list', Float64MultiArray, self.callback_long_pose, queue_size=5)
        rospy.Subscriber('/measurement_topic/wide_focal_pixel_list', Float64MultiArray, self.callback_wide_pixel, queue_size=5)
        rospy.Subscriber('/measurement_topic/wide_focal_pose_list', Float64MultiArray, self.callback_wide_pose, queue_size=5)
        rospy.Subscriber('/groundTruth_topic/gt_position', Float64MultiArray, self.callback_position, queue_size=5)
        rospy.Subscriber('/groundTruth_topic/gt_velocitiy', Float64MultiArray, self.callback_velocity, queue_size=5)        
        
        self.g_set_state = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.ground_vehicle_name = 'my_robot'
        
        self.drone_state = ModelState()
        self.drone_state.model_name = 'iris'
        self.drone_state.reference_frame = 'world'

        self.target_state = ModelState()
        self.target_state.model_name = self.ground_vehicle_name
        self.target_state.reference_frame = 'world'

    # define callback functions        
    def callback_long_pixel(self, msg):
        self.long_pixel_state = msg.data
        # covert tuple to array
        self.long_pixel_state = np.asarray(self.long_pixel_state)        

    def callback_long_pose(self, msg):
        self.long_pose_state = msg.data
        self.long_pose_state = np.asarray(self.long_pose_state)
    
    def callback_wide_pixel(self, msg):
        self.wide_pixel_state = msg.data
        self.wide_pixel_state = np.asarray(self.wide_pixel_state)

    def callback_wide_pose(self, msg):
        self.wide_pose_state = msg.data
        self.wide_pose_state = np.asarray(self.wide_pose_state)

    def callback_position(self, msg):
        self.true_position = msg.data
        self.true_position = np.asarray(self.true_position)

    def callback_velocity(self, msg):
        self.true_velocity = msg.data
        self.true_velocity = np.asarray(self.true_velocity)
    
    # def callback_model(self, msg):  
    #     self.true_pose = np.zeros([3])
    #     self.true_ori = np.zeros([4])
    #     self.true_linear_vel = np.zeros([3])
    #     self.true_angular_vel = np.zeros([3])
                
    #     name_index = msg.name.index(self.ground_vehicle_name)                    
    #     self.true_pose[0] = msg.pose[name_index].position.x
    #     self.true_pose[1] = msg.pose[name_index].position.y
    #     self.true_pose[2] = msg.pose[name_index].position.z        

    #     self.true_ori[0] = msg.pose[name_index].orientation.x
    #     self.true_ori[1] = msg.pose[name_index].orientation.y
    #     self.true_ori[2] = msg.pose[name_index].orientation.z
    #     self.true_ori[3] = msg.pose[name_index].orientation.w
        
    #     self.true_linear_vel[0] = msg.twist[name_index].linear.x
    #     self.true_linear_vel[1] = msg.twist[name_index].linear.y
    #     self.true_linear_vel[2] = msg.twist[name_index].linear.z
        
    #     self.true_angular_vel[0] = msg.twist[name_index].angular.x
    #     self.true_angular_vel[1] = msg.twist[name_index].angular.y
    #     self.true_angular_vel[2] = msg.twist[name_index].angular.z        

    def get_states(self):        
        # 갯수가 안맞네?
        self.pose_states = np.stack((self.long_pose_state, self.wide_pose_state))
        self.pose_states = np.reshape(self.pose_states, [-1])
        # normalization
        self.pose_states = self.pose_states / (-15.)
        return self.pose_states
    
    def get_true_states(self):
        self.true_states = np.stack((self.true_position, self.true_velocity))
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

    def euler_to_quaternion(self, yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

    def reset(self):
        [drone_ox, drone_oy, drone_oz, drone_ow] = self.euler_to_quaternion(0, 0, 0)
        
        # move the drone to the initial position
        self.drone_state.pose.position.x = 0.0
        self.drone_state.pose.position.y = 0.0
        self.drone_state.pose.position.z = 10.0
        self.drone_state.pose.orientation.x = drone_ox
        self.drone_state.pose.orientation.y = drone_oy
        self.drone_state.pose.orientation.z = drone_oz
        self.drone_state.pose.orientation.w = drone_ow
        self.drone_state.twist.linear.x = 5.0
        self.drone_state.twist.linear.y = 0
        self.drone_state.twist.linear.z = 0
        self.drone_state.twist.angular.x = 0
        self.drone_state.twist.angular.y = 0
        self.drone_state.twist.angular.z = 0

        # move the target to the initial position
        [t_ox, t_oy, t_oz, t_ow] = self.euler_to_quaternion(0, 0, 0)
        self.target_state.pose.position.x = 0.0
        self.target_state.pose.position.y = 0.0
        self.target_state.pose.position.z = 0.0
        self.target_state.pose.orientation.x = t_ox
        self.target_state.pose.orientation.y = t_oy
        self.target_state.pose.orientation.z = t_oz
        self.target_state.pose.orientation.w = t_ow
        self.target_state.twist.linear.x = 5.0
        self.target_state.twist.linear.y = 0
        self.target_state.twist.linear.z = 0
        self.target_state.twist.angular.x = 0
        self.target_state.twist.angular.y = 0
        self.target_state.twist.angular.z = 0

        # send the drone and target to the origin point
        self.g_set_state(self.drone_state)
        self.g_set_state(self.target_state)

        # get states
        self.pose_states = self.get_states()
        return self.pose_states

    # def step(self, action):
    #     next_states = self.get_states()
    #     reward = self.get_reward(action)
    #     done = False        
    #     return next_states, reward, done

    # def get_reward(self, action):
    #     action = np.array(action)           
    #     reward_pose = np.sum(action[:3] - self.true_pose)
    #     reward_velocity = np.sum(action[3:] - self.true_linear_vel)
    #     reward = np.stack((reward_pose, reward_velocity))
    #     return reward

# def main():
#     env = ENV()
#     rospy.sleep(2.)
#     rate = rospy.Rate(10)
#     while True:
#         a,b = env.get_states()
#         action = np.zeros([6])
#         b,n,m = env.step(action)

#         rate.sleep()

# if __name__ == '__main__':
#     main()