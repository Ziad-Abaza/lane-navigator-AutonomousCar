import cv2
import numpy as np
import time

from LineDetection import LineDetection
from CarActions import CarActions
from RewardSystem import RewardSystem

def main():
    """
    Main function to process video frames, detect lanes, compute vehicle state, 
    determine actions based on lane offset, and display the results with annotated text.
    """
    cap = cv2.VideoCapture("track.mp4")
    if not cap.isOpened():
        print("Unable to open video file.")
        return

    lane_detection = LineDetection(pixel_to_meter=0.01)
    car_actions = CarActions()
    reward_system = RewardSystem()

    current_steering = 0
    current_speed = 0

    selecting_roi = True
    roi_polygon = None

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", lane_detection.mouse_callback)

    prev_left = None
    prev_right = None
    prev_offset = None
    prev_time = time.time()

    # Mapping from action ID to directional letter
    action_map = {
        0: "L",  # Left turn
        1: "R",  # Right turn
        2: "F",  # Forward (accelerate)
        3: "B",  # Brake (decelerate) - not used in this example
        4: "S"   # Stop - not used in this example
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        if selecting_roi:
            instruction = "Click left mouse button to select ROI points. Right button to finish."
            cv2.putText(frame, instruction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for point in lane_detection.polygon_points:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if lane_detection.drawing:
                selecting_roi = False
                roi_polygon = np.array([lane_detection.polygon_points], dtype=np.int32)
                print("ROI selected:", roi_polygon)
            if key == 27:
                break
            continue

        processed_frame, state, prev_left, prev_right, prev_offset, prev_time = lane_detection.process_frame(
            frame, roi_polygon, prev_left, prev_right, prev_offset, prev_time
        )

        reward = reward_system.calculate_reward(state, collision=False, goal_reached=False)

        offset = state.get('offset', 0)
        if offset > 0.1:
            action_id = 0  # Turn left
        elif offset < -0.1:
            action_id = 1  # Turn right
        else:
            action_id = 2  # Accelerate (move forward)

        new_steering, new_speed = car_actions.apply_action(action_id, current_steering, current_speed)
        current_steering = new_steering
        current_speed = new_speed

        action_letter = action_map.get(action_id, "?")
        cv2.putText(processed_frame, f"Action: {action_letter}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Steering: {current_steering}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Speed: {current_speed}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Reward: {reward:.2f}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Frame", processed_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
