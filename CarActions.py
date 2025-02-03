class CarActions:
    def __init__(self):
        self.steering_step = 5  # change steering
        self.acceleration_step = 1  # change speed
        self.max_steering = 30  
        self.max_speed = 20  # max speed (m/s)

    def apply_action(self, action_id, current_steering, current_speed):
        """
        Apply the specified action to the current car state.
        
        :param action_id: Action ID (0-4)
        :param current_steering: Current steering angle
        :param current_speed: Current speed
        :return: New steering angle, new speed
        """
        new_steering = current_steering
        new_speed = current_speed
        
        if action_id == 0:
            new_steering = min(current_steering + self.steering_step, self.max_steering)
        elif action_id == 1:
            new_steering = max(current_steering - self.steering_step, -self.max_steering)
        elif action_id == 2:
            new_speed = min(current_speed + self.acceleration_step, self.max_speed)
        elif action_id == 3:
            new_speed = max(current_speed - self.acceleration_step, 0)
        elif action_id == 4:
            new_speed = 0
        else:
            raise ValueError("Invalid action_id")
        
        return new_steering, new_speed

    def get_action_space(self):
        """Retrieve the list of all available actions."""
        return [0, 1, 2, 3, 4]
