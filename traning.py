from utils import  *
# --- Data Preparation ---

# Set the path to your dataset folder
path = 'Dataset'

# Import driving data information from the CSV file
data = import_data_info(path)
if data is None:
    print("Error: Failed to load dataset.")
    exit()

# Balance the data to reduce steering bias
data = balance_data(data)
if data.empty:
    print("Error: No data available after balancing.")
    exit()

# Load image paths and steering angles
images_path, steerings = load_data(path, data)

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = split_data(images_path, steerings, test_size=0.2, random_state=5)

# --- Model Creation & Training ---

# Create the model architecture
model = create_model()

# Create data generators for training and validation
batch_size = 32
train_generator = batch_generator(x_train, y_train, batch_size, is_training=True)
val_generator = batch_generator(x_val, y_val, batch_size, is_training=False)

# Train the model using the generators
history = model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=val_generator,
    validation_steps=len(x_val) // batch_size,
    epochs=10  # adjust epochs as needed
)

# --- Evaluation & Saving ---

# Plot training and validation loss
plot_loss(history)

# Plot the distribution of steering angles (using log scale) for the training set
plot_steering_distribution(y_train)

# Save the trained model (native Keras format)
save_model(model, 'trained_model')

# Load the model back later
loaded_model = load_trained_model('trained_model')

# --- Visualize Sample Images ---

# Get a batch of training images from the generator
sample_images, sample_labels = next(train_generator)
# Display a few sample augmented images along with their steering angles
display_images(sample_images[:5], sample_labels[:5], 'Sample Augmented Images')
