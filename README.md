### Lane Navigator - Autonomous Car

This project is the beginning of an autonomous car system that uses for Reinforcement Learning (RL). It focuses on detecting lanes, calculating the vehicle's position within the lane, and applying actions such as steering and speed adjustments. This serves as an early stage for developing a fully self-driving car.

![Lane Navigator](https://github.com/Ziad-Abaza/lane-navigator-AutonomousCar/blob/main/screenshots/screenshot.png)

## Technologies Used

- **Python**: Programming language for implementing the system.
- **OpenCV**: Used for image processing, including line detection, frame manipulation, and the Region of Interest (ROI) selection.
- **Numpy**: Used for mathematical operations, particularly in line fitting and distance calculations.

### Reward System

The `RewardSystem.py` script defines the rewards and penalties the car receives based on its behavior:

- **Lane Centering**: The car is rewarded for staying in the center of the lane.
- **Speed**: A reward is given for maintaining a safe speed when near the center.
- **Collision Penalty**: A penalty is applied if the car collides with any obstacles.
- **Goal Achievement**: A reward is provided when the car reaches the goal.
- **Lane Departure**: A penalty is imposed if the car moves out of the lane.

### Line Detection

The `LineDetection.py` script detects the lanes using OpenCV’s Hough Transform. The lines detected in the frame are filtered based on their slope to determine left and right lanes. The class also handles ROI selection using mouse events and averages the detected lines to improve accuracy.

### Car Actions

The `CarActions.py` script allows the car to take actions based on the state of the environment. These actions include steering adjustments and speed changes. The car can apply one of five predefined actions that influence the steering and speed.

### Data Structure and Usage

The state of the car is represented in the following dictionary:

```python
state = {
    'left_lane': left_lane_coordinates,
    'right_lane': right_lane_coordinates,
    'offset': lane offset (m),
    'offset_rate': change in offset per second (m/s),
    'lane_width': width of the lane (m),
    'left_distance': distance to the left lane (m),
    'right_distance': distance to the right lane (m),
    'steering_angle': calculated steering angle (degrees)
}
```

This state is continuously updated as the car navigates the lane, and it is used to calculate the rewards and take actions.

### Focus Area

The focus area for lane detection is selected manually through mouse clicks on the region of interest (ROI). A screenshot of the selection process is shown below:

![Select Focus Area](https://github.com/Ziad-Abaza/lane-navigator-AutonomousCar/blob/main/screenshots/select_point.png)

### How to Use

1. Clone this repository:
    ```bash
    git clone https://github.com/Ziad-Abaza/lane-navigator-AutonomousCar.git
    ```

2. Run the main script:
    ```bash
    python main.py
    ```

3. Use mouse clicks to define the Region of Interest (ROI) in the frame for lane detection.

4. The car will begin navigating within the detected lanes, and you can observe its actions and rewards through the output.

### Future Development

This is just the first step toward building a fully autonomous car. Future improvements will include:

- Integration of deeper Reinforcement Learning models for better decision-making.
- More sophisticated lane detection algorithms using deep learning.
- Simulation of real-world conditions like road curves and obstacles.
- Integration with self-driving car platforms for real-time testing.

Feel free to contribute to the project. Any improvements, ideas, or suggestions are welcome!

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
