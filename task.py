import numpy as np
from physics_sim import PhysicsSim
import math

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
  
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def vel_angle(current_pos, target_pos, velocity):
    target_dir = target_pos - current_pos

    if np.linalg.norm(target_dir) == 0:
        return -1

    return angle_between(target_dir, velocity)

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.prev_dist = np.inf
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""
        angle_to_targetpos = vel_angle(self.sim.pose[:3], self.target_pos, self.sim.v) / np.pi
        distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        reward = (5. - distance_to_target) * 0.3 - angle_to_targetpos * 0.5
        
        if self.sim.pose[2] >= self.target_pos[2]:  # agent has crossed the target height
            reward += 20.0  # bonus reward
        
        if self.sim.time > 4:
            reward += 20.0  # extra reward, agent made it last
       
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
       
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state