#!/usr/bin/env python2

import tf
import rospy
import time
import numpy as np
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point
from scipy.spatial.transform import Rotation
from tf import TransformListener, TransformerROS, transformations
from tf import LookupException, ConnectivityException, ExtrapolationException
from medical_msgs.msg import *
from medical_msgs.srv import *
from std_srvs.srv import Trigger, TriggerRequest



class drl_transform(object):
	def __init__(self):
		self.old_predict_pose = Point()
		self.switch = False

		# Subscriber
		rospy.Subscriber('DRL/hand_object_pose', HandObjectPose, self.callback, queue_size=1)

		# Publisher
		self.target_pos = rospy.Publisher("/target_pose", HandObjectPose, queue_size=1)

		# service
		self.algorithm_switch = rospy.Service("~algorithm_switch_server", algorithm_switch, self.switch_callback)

		self.count = 0
		self.count_thres = 1
		self.dis_thres = 0.05

	def callback(self, msg):

		try:
			listener.waitForTransform(
				'vx300s/base_link', 'camera_link', rospy.Time(0), rospy.Duration(1.0))
			(trans, rot) = listener.lookupTransform(
				'vx300s/base_link', 'camera_link', rospy.Time(0))
		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
			print("Error TF listening")
			return

		self.tf_pose = HandObjectPose()

		pose = tf.transformations.quaternion_matrix(np.array(
						[msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]))

		pose[0, 3] = msg.pose.position.x
		pose[1, 3] = msg.pose.position.y
		pose[2, 3] = msg.pose.position.z

		offset_to_world = np.matrix(transformations.quaternion_matrix(rot))
		offset_to_world[0, 3] = trans[0]
		offset_to_world[1, 3] = trans[1]
		offset_to_world[2, 3] = trans[2]

		tf_pose_matrix = np.array(np.dot(offset_to_world, pose))

		# Create a rotation object from Euler angles specifying axes of rotation
		rot = Rotation.from_dcm([[tf_pose_matrix[0, 0], tf_pose_matrix[0, 1], tf_pose_matrix[0, 2]], [tf_pose_matrix[1, 0], tf_pose_matrix[1, 1], tf_pose_matrix[1, 2]], [tf_pose_matrix[2, 0], tf_pose_matrix[2, 1], tf_pose_matrix[2, 2]]])

		# Convert to quaternions and print
		rot_quat = rot.as_quat()

		if tf_pose_matrix[0, 3] >= 0.15 and tf_pose_matrix[0, 3] <= 1.5:
			
			self.tf_pose.pose.position.x = tf_pose_matrix[0, 3]
			self.tf_pose.pose.position.y = tf_pose_matrix[1, 3]
			self.tf_pose.pose.position.z = tf_pose_matrix[2, 3]
			self.tf_pose.pose.orientation.x = rot_quat[0]
			self.tf_pose.pose.orientation.y = rot_quat[1]
			self.tf_pose.pose.orientation.z = rot_quat[2]
			self.tf_pose.pose.orientation.w = rot_quat[3]
			print(self.tf_pose)

			if not self.switch:
				return
			self.matching(self.tf_pose.pose.position)

	def matching(self, real_world_point):
		lock = 1
		p_s = real_world_point
		p_t = self.old_predict_pose
		if np.sqrt((p_s.x - p_t.x)**2 + (p_s.y - p_t.y)**2 + (p_s.z - p_t.z)**2) < self.dis_thres:
			self.count = self.count + 1
			real_world_point.x = (p_s.x*self.count + p_t.x) / (self.count+1)
			real_world_point.y = (p_s.y*self.count + p_t.y) / (self.count+1)
			real_world_point.z = (p_s.z*self.count + p_t.z) / (self.count+1)	
		
		else:
			self.count = 0		

		self.old_predict_pose = real_world_point

		if self.count == self.count_thres:
			lock = 0

		if lock == 0:
			self.count = 0
			
			target_pose = HandObjectPose()
			target_pose.pose.position.x = real_world_point.x
			target_pose.pose.position.y = real_world_point.y
			target_pose.pose.position.z = real_world_point.z 
			target_pose.pose.orientation.x = self.tf_pose.pose.orientation.x
			target_pose.pose.orientation.y = self.tf_pose.pose.orientation.y
			target_pose.pose.orientation.z = self.tf_pose.pose.orientation.z
			target_pose.pose.orientation.w = self.tf_pose.pose.orientation.w

			self.target_pos.publish(target_pose)
			print(target_pose)
			time.sleep(0.5)

			try:
				do_grasping = rospy.ServiceProxy('handover_pick', Trigger) # call service
				resp = do_grasping()

			except rospy.ServiceException as exc:
				print("service did not process request: " + str(exc))

			time.sleep(5)

	def switch_callback(self, req):
		resp = algorithm_switchResponse()
		self.switch = req.data		

	def onShutdown(self):
		rospy.loginfo("Shutdown.")


if __name__ == '__main__':
	rospy.init_node('drl_transform', anonymous=False)
	drl_transform = drl_transform()
	listener = TransformListener()
	transformer = TransformerROS()
	rospy.on_shutdown(drl_transform.onShutdown)
	rospy.spin()
