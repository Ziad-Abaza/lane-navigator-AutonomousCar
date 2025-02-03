class RewardSystem:
    def __init__(self):
        self.lane_center_weight = 2.0   
        self.speed_weight = 0.1         
        self.collision_penalty = -50    
        self.goal_reward = 100          
        self.out_of_lane_penalty = -20  
        
        self.max_lane_offset = 1.0      
        self.safe_distance = 0.5        
    
    def calculate_reward(self, state, collision=False, goal_reached=False):
        """
        Calculate the reward based on the car's state.
        
        :param state: Car state from CarPositioning.process_frame
        :param collision: Indicates if a collision occurred
        :param goal_reached: Indicates if the goal was reached
        :return: Total reward value
        """
        reward = 0
        
        offset = abs(state['offset'])
        if offset < self.max_lane_offset:
            reward += self.lane_center_weight * (1 - offset/self.max_lane_offset)
        else:
            reward += self.out_of_lane_penalty
        
        if 'speed' in state and offset < 0.3:
            reward += self.speed_weight * state['speed']
        
        if collision:
            reward += self.collision_penalty
        
        if goal_reached:
            reward += self.goal_reward
        
        min_distance = min(state['left_distance'], state['right_distance'])
        if min_distance < self.safe_distance:
            reward += (min_distance - self.safe_distance) * 10  
        
        return reward
