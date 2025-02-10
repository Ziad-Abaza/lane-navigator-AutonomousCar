# Lane Navigator using CNN

## Overview
This project implements a convolutional neural network (CNN) for lane navigation in autonomous driving. It includes data preprocessing, augmentation, model training, evaluation, and visualization.

## Project Structure
```
LaneNavigator/
├── utils.py           # Utility functions for data handling, augmentation, and model training
├── training.py        # Training pipeline for the CNN model
├── trained_model.keras # Saved trained model in Keras format
├── Dataset/          # Folder containing training data (images and CSV log)
│   ├── IMG/         # Image dataset
│   ├── driving_log.csv # CSV file containing driving data
```

## Dependencies
Ensure you have the following dependencies installed:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn imgaug opencv-python
```

## Data Preparation
The dataset consists of images and a CSV file (`driving_log.csv`) containing:
- **Center**: Path to the center camera image
- **Left & Right**: Paths to side camera images
- **Steering**: Steering angle
- **Throttle, Brake, Speed**: Additional driving parameters

### Steps:
1. **Load Dataset**: `import_data_info(path)` extracts image paths and steering angles.
2. **Balance Data**: `balance_data(data)` reduces bias in steering angles.
3. **Load Data**: `load_data(path, data)` loads images and labels.
4. **Split Dataset**: `split_data(images_path, steerings)` splits into training/validation sets.

## Model Architecture
The CNN model follows a similar structure to NVIDIA’s self-driving car model:
- **Input Layer**: Normalization (`Lambda(lambda x: x / 255.0 - 0.5)`) 
- **Convolutional Layers**: Feature extraction with activation `elu`
- **Dropout Layer**: Prevents overfitting
- **Fully Connected Layers**: Predicts steering angle

### Model Creation
```python
model = create_model()
```

## Training the Model
### Steps:
1. **Data Augmentation**: Random flipping, brightness adjustment, noise addition.
2. **Batch Generation**: `batch_generator()` dynamically loads images and applies augmentation.
3. **Train Model**:
```python
history = model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=val_generator,
    validation_steps=len(x_val) // batch_size,
    epochs=10
)
```
4. **Loss Visualization**:
```python
plot_loss(history)
```

## Model Saving & Loading
- **Save Model**: `save_model(model, 'trained_model')`
- **Load Model**: `loaded_model = load_trained_model('trained_model')`

## Visualization
- **Sample Augmented Images**:
```python
sample_images, sample_labels = next(train_generator)
display_images(sample_images[:5], sample_labels[:5], 'Sample Augmented Images')
```
- **Steering Angle Distribution**:
```python
plot_steering_distribution(y_train)
```

## Running the Project
To train the model, execute:
```bash
python training.py
```
This will load data, train the CNN model, visualize results, and save the trained model.

## Future Improvements
- Integrate real-time testing with a simulator.
- Implement reinforcement learning for adaptive steering.
- Optimize model for embedded deployment (Raspberry Pi, Jetson Nano).