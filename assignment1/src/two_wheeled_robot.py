import numpy as np
import matplotlib as mpl
from math import sin, cos, atan2, inf, pi
from ir_sim.world import RobotBase
import copy

class RobotTwoWheel(RobotBase):

	student_id = 3035770800

	robot_type = 'custom'
	appearance = 'circle'
	state_dim  = (3, 1) # the state dimension, x, y, theta(heading direction),
	vel_dim = (2, 1)    # the angular velocity of right and left wheel
	goal_dim = (3, 1)	# the goal dimension, x, y, theta 
	position_dim = (2, 1) # the position dimension, x, y

	def __init__(self, id, state, vel = np.zeros((2, 1)), goal=np.zeros((3, 1)), 
				 step_time = 0.01, **kwargs):
		self.shape  = kwargs.get('shape', [4.6, 1.6, 3, 1.6]) # Only for rectangle shape
		self.radius = kwargs.get('radius', 0.25)
		self.radius_wheel = kwargs.get('radius_wheel', 1)
		self.control_mode = kwargs.get('control_mode', 'auto')
		self.noise_mode   = kwargs.get('noise_mode', 'none')
		self.noise_amplitude = kwargs.get('noise_amplitude', np.c_[[0.005, 0.005, 0.001]])
		super(RobotTwoWheel, self).__init__(id, state, vel, goal, step_time, **kwargs)

	def dynamics(self, state, vel, **kwargs):
		r"""
		Choose dynamics function based on different noise mode.
		"""
		state_, vel_ = copy.deepcopy(state), copy.deepcopy(vel)
		if self.control_mode == 'keyboard':
			vel_ = self.keyboard_to_angular(vel_)

		if self.noise_mode == 'none':
			return self.dynamics_without_noise(state_, vel_, **kwargs)
		elif self.noise_mode == 'linear':
			return self.dynamics_with_linear_noise(state_, vel_, **kwargs)
		else:
			return self.dynamics_with_nonlinear_noise(state_, vel_, **kwargs)

	def dynamics_without_noise(self, state, vel, **kwargs):
		r"""
		Question 1
		The dynamics of two-wheeled robot, be careful with the defined direction of the robot.
		
		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel:   2*1 matrix, the angular velocity of the right and left wheels, [omega1, omega2]
		@param state: 3*1 matrix, the state dimension, [x, y, theta (heading direction)]
		
		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		
		"*** YOUR CODE STARTS HERE ***"
		# 获取状态参数
		x, y, theta = state.flatten()
		omega1, omega2 = vel.flatten()
    
		# 计算线速度
		v1 = omega1 * r
		v2 = omega2 * r
    
		# 处理特殊情况
		if omega1 == omega2:
        	# 直线运动
			v = v1  # or v2, since v1 = v2
			next_x = x + v * np.cos(theta) * dt
			next_y = y + v * np.sin(theta) * dt
			next_theta = theta  # heading direction stays the same
		elif omega1 == -omega2:
        	# 原地旋转
			omega0 = (omega1 - omega2) * r / (2 * l)
			next_x = x
			next_y = y
			next_theta = theta + omega0 * dt
		else:
        	# 计算 r_ICR 和 omega0
			r_ICR = (l * (omega1 + omega2)) / (omega2 - omega1)
			omega0 = (omega1 - omega2) * r / (2 * l)

			# 计算 ICR 位置
			ICR_x = x + r_ICR * np.sin(theta)
			ICR_y = y - r_ICR * np.cos(theta)
        
			# 更新状态
			next_x = np.cos(omega0 * dt) * (x - ICR_x) - np.sin(omega0 * dt) * (y - ICR_y) + ICR_x
			next_y = np.sin(omega0 * dt) * (x - ICR_x) + np.cos(omega0 * dt) * (y - ICR_y) + ICR_y
			next_theta = theta + omega0 * dt

		next_state = np.array([[next_x], [next_y], [next_theta]])

		"*** YOUR CODE ENDS HERE ***"
		return next_state

	def dynamics_with_linear_noise(self, state, vel, **kwargs):
		r"""
		Question 2(a)
		The dynamics of two-wheeled robot, be careful with the defined direction.
		
		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel:   2*1 matrix, the angular velocity of the right and left wheels, [omega1, omega2]
		@param state: 3*1 matrix, the state dimension, [x, y, theta (heading direction)]
		@param noise: 3*1 matrix, noises of the additive Gaussian disturbances 
						for the state, [epsilon_x, epsilon_y, epsilon_theta]
		
		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		R  = self.noise_amplitude
		noise = np.random.normal(0, R)
		
		"*** YOUR CODE STARTS HERE ***"
		# First, calculate the next state without noise
		next_state = self.dynamics_without_noise(state, vel, **kwargs)

		# Add Gaussian noise to the next state
		next_state = next_state + noise

		"*** YOUR CODE ENDS HERE ***"
		
		return next_state

	def dynamics_with_nonlinear_noise(self, state, vel, **kwargs):
		r"""
		Question 2(b)
		The dynamics of two-wheeled robot, be careful with the defined direction.
		
		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel:   2*1 matrix, the angular velocity of the right and left wheels, [omega1, omega2]
		@param state: 3*1 matrix, the state dimension, [x, y, theta (heading direction)]
		@param noise: 2*1 matrix, noises of the additive Gaussian disturbances 
						for the angular velocity of wheels and heading, [epsilon_omega, epsilon_theta]. 
						Assume that the noises for omega1 and omega2 are the same.
		
		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		R  = self.noise_amplitude
		# noise = np.random.normal(0, R)
		
		"*** YOUR CODE STARTS HERE ***"

    	# Generate Gaussian noise for omega1 and omega2 (same noise for both)
		epsilon_omega = np.random.normal(0, R[0])  # Noise for omega
		vel[0] += epsilon_omega  # Apply noise to omega1
		vel[1] += epsilon_omega  # Apply same noise to omega2
    
    	# Generate Gaussian noise for theta
		epsilon_theta = np.random.normal(0, R[1])  # Noise for theta
		state[2] += epsilon_theta  # Apply noise to theta

		# 获取状态参数
		x, y, theta = state.flatten()
		omega1, omega2 = vel.flatten()
    
		# 计算线速度
		v1 = omega1 * r
		v2 = omega2 * r
    
		# 处理特殊情况
		if omega1 == omega2:
        	# 直线运动
			v = v1  # or v2, since v1 = v2
			next_x = x + v * np.cos(theta) * dt
			next_y = y + v * np.sin(theta) * dt
			next_theta = theta  # heading direction stays the same
		elif omega1 == -omega2:
        	# 原地旋转
			omega0 = (omega1 - omega2) * r / (2 * l)
			next_x = x
			next_y = y
			next_theta = theta + omega0 * dt
		else:
        	# 计算 r_ICR 和 omega0
			r_ICR = (l * (omega1 + omega2)) / (omega2 - omega1)
			omega0 = (omega1 - omega2) * r / (2 * l)

			# 计算 ICR 位置
			ICR_x = x + r_ICR * np.sin(theta)
			ICR_y = y - r_ICR * np.cos(theta)

			# 更新状态
			next_x = np.cos(omega0 * dt) * (x - ICR_x) - np.sin(omega0 * dt) * (y - ICR_y) + ICR_x
			next_y = np.sin(omega0 * dt) * (x - ICR_x) + np.cos(omega0 * dt) * (y - ICR_y) + ICR_y
			next_theta = theta + omega0 * dt

    	# 更新状态矩阵
		next_state = np.array([[next_x], [next_y], [next_theta]])

		"*** YOUR CODE ENDS HERE ***"
		
		return next_state

	def policy(self):
		r"""
		Question 3
		A simple policy for steering.

		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center

		Return:
		@instructions: A list containing instructions of wheels' angular velocities. 
					   Form: [[omega1_t1, omega2_t1], [omega1_t2, omega2_t2], ...]
		@timepoints: A list containing the duration time of each instruction.
					   Form: [t1, t2, ...], then the simulation time is \sum(t1+t2+...).
		@path_length: The shortest trajectory length after calculation by your hand.
		"""

		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		instructions = []
		timepoints = []
		path_length = 0
		
		"*** YOUR CODE STARTS HERE ***"
		max_angular_velocity = 0.5  # rad/s, as per problem constraint
		precise_angular_velocity = 0.4  # rad/s, for fine adjustments
		current_time = 0
		instructions = []


		# Initial state
		x, y, theta = 0.5, 2.0, np.pi / 2
		target_x, target_y, target_theta = 4.5, 2.0, np.pi / 2

		# Set time intervals and control loop
		remaining_time = 10  # we need to complete in less than 10s
		current_time = 0

		# Initial state
		x, y, theta = 0.5, 2.0, np.pi / 2
		target_x, target_y, target_theta = 4.5, 2.0, np.pi / 2

		# Step 1: Turn to face the target direction
		delta_x = target_x - x
		delta_y = target_y - y
		target_angle = np.arctan2(delta_y, delta_x)
		angle_diff = target_angle - theta

		# Ensure angle_diff is within -pi to pi
		angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Normalize to [-pi, pi]

		# Adjust to target angle with minimum commands
		if abs(angle_diff) > 0.01:
			omega1 = max_angular_velocity if angle_diff > 0 else -max_angular_velocity
			omega2 = -omega1  # Opposite wheel movement to turn
			turn_time = abs(angle_diff) / (r * max_angular_velocity / l)  # Time to complete the turn
			
			instructions.append([omega1, omega2])
			timepoints.append(turn_time)
			current_time += turn_time
    
			# Update theta after turning
			theta += omega1 * r / l * turn_time


		# Step 2: Move forward towards the target
		distance_to_target = np.hypot(delta_x, delta_y)
    
		# Calculate the time needed to move forward
		forward_time = distance_to_target / (r * max_angular_velocity)
    
		# Add the forward instruction
		instructions.append([max_angular_velocity, max_angular_velocity])
		timepoints.append(forward_time)
		current_time += forward_time

		# Update position after moving
		x += np.cos(theta) * r * max_angular_velocity * forward_time
		y += np.sin(theta) * r * max_angular_velocity * forward_time

		# Add a small buffer to ensure robot moves slightly over the target
		extra_time = 0.01  # Extend the duration slightly to ensure overshoot
		instructions[-1] = [max_angular_velocity, max_angular_velocity]
		timepoints[-1] += extra_time
		current_time += extra_time
    
		# Step 3: Adjust heading at the goal
		final_heading_diff = target_theta - theta
		final_heading_diff = np.arctan2(np.sin(final_heading_diff), np.cos(final_heading_diff))  # Normalize

		if abs(final_heading_diff) > 0.01:
			omega1 = precise_angular_velocity if final_heading_diff > 0 else -precise_angular_velocity
			omega2 = -omega1
			adjust_time = abs(final_heading_diff) / (r * precise_angular_velocity / l)
			instructions.append([omega1, omega2])
			timepoints.append(adjust_time)
			current_time += adjust_time

    	# Step 4: Calculate the total path length manually
		path_length = distance_to_target


		"*** YOUR CODE ENDS HERE ***"
		
		return instructions, timepoints, path_length

	
	def plot_robot(self, ax, robot_color = 'g', goal_color='r', show_goal=True, show_text=False, show_traj=False, show_uncertainty=False, traj_type='-g', fontsize=10, **kwargs):
		x = self.state[0, 0]
		y = self.state[1, 0]

		goal_x = self.goal[0, 0]
		goal_y = self.goal[1, 0]

		robot_circle = mpl.patches.Circle(xy=(x, y), radius = self.radius, color = robot_color)
		robot_circle.set_zorder(3)

		ax.add_patch(robot_circle)
		if show_text: ax.text(x - 0.5, y, 'r'+ str(self.id), fontsize = fontsize, color = 'r')
		self.plot_patch_list.append(robot_circle)

		# arrow
		theta = self.state[2][0]
		arrow = mpl.patches.Arrow(x, y, 0.5*cos(theta), 0.5*sin(theta), width = 0.6)
		arrow.set_zorder(3)
		ax.add_patch(arrow)
		self.plot_patch_list.append(arrow)

		if show_goal:
			goal_circle = mpl.patches.Circle(xy=(goal_x, goal_y), radius = self.radius, color=goal_color, alpha=0.5)
			goal_circle.set_zorder(1)

			ax.add_patch(goal_circle)
			if show_text: ax.text(goal_x + 0.3, goal_y, 'g'+ str(self.id), fontsize = fontsize, color = 'k')
			self.plot_patch_list.append(goal_circle)

		if show_traj:
			x_list = [t[0, 0] for t in self.trajectory]
			y_list = [t[1, 0] for t in self.trajectory]
			self.plot_line_list.append(ax.plot(x_list, y_list, traj_type))

	def keyboard_to_angular(self, vel):
		r"""

		Change the velocity from [v, omega] to [omega1, omega2].
		
		Some parameters that you may use:
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel: 2*1 matrix, the forward velocity and the rotation velocity [v, omega]

		Return:
		@param vel_new: 2*1 matrix,the angular velocity of right and left wheel, [omega1, omega2]
		"""
		l  = self.radius
		r  = self.radius_wheel
		
		vel_new = np.c_[[vel[0, 0]+vel[1, 0]*l, vel[0, 0]-vel[1, 0]*l]] / r
		return vel_new
	

