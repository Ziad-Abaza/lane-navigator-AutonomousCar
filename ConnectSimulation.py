print("Initializing System...")

import os
import socketio
import eventlet
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from flask import Flask
import base64
import cv2

# Reduce TensorFlow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup Socket.IO and Flask
sio = socketio.Server()
app = Flask(__name__)

# Set maximum speed
MAX_SPEED = 10

def img_preprocess(img):
    """Preprocess the input image for the self-driving model."""
    try:
        print("Starting image preprocessing...")
        img = img[60:135, :, :]  
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) 
        img = cv2.GaussianBlur(img, (3, 3), 0) 
        img = cv2.resize(img, (200, 66)) 
        img = img / 255.0 
        print("Image preprocessing completed.")
        return img
    except Exception as e:
        print(f"Error in image processing: {e}")
        return np.zeros((66, 200, 3))

@sio.on('telemetry')
def telemetry(sid, data):
    """Receive telemetry data from the simulator and process the image."""
    try:
        print("Received telemetry data...")
        speed = float(data['speed'])
        img_b64 = data['image']
        if not img_b64:
            print("No image received.")
            return
        
        # Decode image from base64 and convert to array
        image = Image.open(BytesIO(base64.b64decode(img_b64)))
        image = np.asarray(image)
        print("Image received and decoded.")
        
        image = img_preprocess(image)
        image = np.array([image], dtype=np.float32)

        # Predict steering angle
        steering_angle = float(model.predict(image, verbose=0)[0][0])

        # Calculate throttle
        throttle = max(0.1, 1.0 - speed / MAX_SPEED)

        print(f"Steering: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}")

        send_control(steering_angle, throttle)

    except Exception as e:
        print(f"Error in telemetry: {e}")

@sio.on('connect')
def connect(sid, environ):
    """Handle a new connection from the simulator."""
    print('Car connected!')
    send_control(0, 1)

@sio.on('disconnect')
def disconnect(sid):
    """Handle disconnections."""
    print(f"Client {sid} disconnected")

@sio.on('*')
def catch_all(event, sid, data):
    """Log any event received (for debugging)."""
    print(f"Received event: {event} from {sid} with data: {data}")

def send_control(steering_angle, throttle):
    """Send control commands to the simulator."""
    try:
        print(f"Sending control: Steering Angle = {steering_angle}, Throttle = {throttle}")
        sio.emit('steer', data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        })
    except Exception as e:
        print(f"Error sending control: {e}")

if __name__ == '__main__':
    try:
        print("Loading model...")
        model = load_model('trained_model.keras', safe_mode=False)
        print("Model loaded successfully.")

        app = socketio.Middleware(sio, app)
        print("Starting server on port 4567...")
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
        print("Server started successfully.")
    except Exception as e:
        print(f"Failed to start server: {e}")
