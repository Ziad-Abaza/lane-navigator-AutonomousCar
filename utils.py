from matplotlib import pyplot as plt
from matplotlib import image as IMG
from imgaug import augmenters as iaa
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import os
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import cv2
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input

def get_name(file_path):
    """Extract the file name from a given file path."""
    return os.path.basename(file_path)

def import_data_info(path):
    """Load driving data from CSV and extract necessary columns."""
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    file_path = os.path.join(path, 'driving_log.csv')

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    data = pd.read_csv(file_path, names=columns)

    if 'Center' in data.columns and data['Center'].notnull().all():
        data['Center'] = data['Center'].apply(get_name)

    return data

def balance_data(data, display=True):
    """Balance the steering angle distribution to reduce bias."""
    number_bins = 31
    samples_per_bin = 2500
    hist_value, bins = np.histogram(data['Steering'], number_bins)
    center = (bins[:-1] + bins[1:]) * 0.5

    remove_index_list = []
    for i in range(number_bins):
        bin_indices = data.index[(data['Steering'] >= bins[i]) & (data['Steering'] <= bins[i+1])].tolist()
        bin_indices = shuffle(bin_indices)
        remove_index_list.extend(bin_indices[samples_per_bin:])

    print(f'Removed Repeated Images: {len(remove_index_list)}')
    data.drop(remove_index_list, inplace=True)
    print(f'Final Data Size: {len(data)}')

    if display:
        hist_value, _ = np.histogram(data['Steering'], number_bins)
        plt.figure(figsize=(10, 5))
        plt.bar(center, hist_value, width=0.06)
        plt.xlabel("Steering Angle")
        plt.ylabel("Frequency")
        plt.title("Steering Angle Distribution")
        plt.show()

    return data

def load_data(path, data):
    """Load image paths and steering values."""
    print("Loading images...")
    images_path = np.array([os.path.join(path, 'IMG', data.iloc[i, 0]) for i in range(len(data))])
    steerings = data.iloc[:, 3].astype(float).values
    print(f"Loaded {len(images_path)} images successfully.")
    return images_path, steerings

def split_data(images_path, steerings, test_size=0.2, random_state=5):
    """Split the dataset into training and validation sets."""
    return sklearn_train_test_split(images_path, steerings, test_size=test_size, random_state=random_state)

def augment_image(image_path, steering):
    """Apply random augmentation to the given image."""
    image = IMG.imread(image_path)
    augmenter = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(
            scale=(0.8, 1.2),
            rotate=(-20, 20),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
        ),
        iaa.Multiply((0.7, 1.3)),
        iaa.GaussianBlur(sigma=(0, 1.5)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.07 * 255)),
    ])
    augmented_image = augmenter(image=image)
    if np.random.rand() < 0.5:
        augmented_image = np.fliplr(augmented_image)
        steering = -steering
    return augmented_image, steering

def preprocess_image(image):
    """Preprocess the image by cropping, resizing, and normalizing."""
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image

def batch_generator(image_paths, steerings, batch_size, is_training):
    """Generate training and validation batches with preprocessing."""
    while True:
        images = []
        labels = []
        for _ in range(batch_size):
            index = np.random.randint(0, len(image_paths))
            image_path = image_paths[index]
            steering = steerings[index]
            image = IMG.imread(image_path)
            if is_training:
                image, steering = augment_image(image_path, steering)
            image = preprocess_image(image)
            images.append(image)
            labels.append(steering)
        yield np.array(images), np.array(labels)

def display_images(images, steerings, title):
    """Display images with corresponding steering angles."""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(f'Steering: {steerings[i]}')
        ax.axis('off')
    plt.show()

def save_model(model, model_name):
    """Save the trained model in the native Keras format."""
    model.save(f'{model_name}.keras')
    print(f'Model saved as {model_name}.keras')

def load_trained_model(model_name):
    """Load a pre-trained model from a file."""
    return load_model(f'{model_name}.keras', safe_mode=False)

def plot_loss(history):
    """Plot the training and validation loss."""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()

def plot_steering_distribution(steerings):
    """Plot the distribution of steering angles using log scale."""
    plt.figure(figsize=(10, 5))
    plt.hist(steerings, bins=50, color='skyblue', log=True)
    plt.title('Steering Angle Distribution (Log Scale)')
    plt.xlabel('Steering Angle')
    plt.ylabel('Log Count')
    plt.show()

def normalize(x):
    return x / 255.0 - 0.5

def create_model():
    """Create the model architecture."""
    model = Sequential([
        Input(shape=(66, 200, 3)),  
        Lambda(lambda x: x / 255.0 - 0.5),  
        Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Dropout(0.5),
        Flatten(),
        Dense(100, activation='elu'),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model