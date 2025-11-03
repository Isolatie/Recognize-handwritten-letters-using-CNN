import matplotlib.pyplot as plt
import numpy as np
import math
import random
import tensorflow as tf
from Download import load_data

def plot_tensor(tensor, title_string, channel=0, colormap='gist_yarg', colorbar_flag=False, subplot_title_strings=''):
    number_of_plots = tensor.shape[0]
    grid_size = math.ceil(math.sqrt(number_of_plots))
    fig, axes = plt.subplots(grid_size, grid_size)
    fig.suptitle(title_string)
    
    for i, ax in enumerate(axes.flat):
        if i < number_of_plots:
            image = tensor[i, :, :, channel]
            h = ax.imshow(image, vmin=np.min(image), vmax=np.max(image), cmap=colormap)
            if colorbar_flag:
                cbar = fig.colorbar(h, ax=ax)
                cbar.minorticks_on()
            if subplot_title_strings:
                ax.set_title(subplot_title_strings[i])
        else:
            ax.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

def digits_to_letters(y):
    return np.array([chr(64 + int(i)) for i in y])  # 65 is ASCII for 'A', though for some reason EMNIST dataset has 27 labels for a letter that doesn't exist, so we start at 24

# Load data, split between train and test sets
X_train, y_train, X_test, y_test = load_data()
print(f'X_train shape = {X_train.shape}')
print(f'X_test shape = {X_test.shape}')

num_classes = len(np.unique(y_train))

# Scale images to the [0, 1] range
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert to 4-dimensional tensor
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Flip horizontally and rotate 90 degrees counterclockwise
X_train = np.rot90(np.flip(X_train, axis=2), k=1, axes=(1, 2))
X_test = np.rot90(np.flip(X_test, axis=2), k=1, axes=(1, 2))

print(f'X_train transformed shape = {X_train.shape}')
print(f'X_test transformed shape = {X_test.shape}')

# Convert class vectors to one-hot encoding
# +1 for extra class in labels dataset
y_train = tf.keras.utils.to_categorical(y_train, num_classes+1)
y_test = tf.keras.utils.to_categorical(y_test, num_classes+1)

# Construct network
input_shape = X_train.shape[1:]
model = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes+1, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
batch_size = 128
epochs = 2
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

# Test
print('\033[H\033[J')
plt.close('all')
indices = random.sample(range(0, len(X_test)), 9)
true = np.argmax(y_test[indices], axis=1)
predicted = np.argmax(model.predict(X_test[indices]), axis=-1)

# Convert numerical labels to letters
true = digits_to_letters(true)
predicted = digits_to_letters(predicted)

# Create strings for plot titles
strings = [f'true: {true[i]}, pred.: {predicted[i]}' for i in range(len(indices))]
plot_tensor(X_test[indices], 'Sample of CNN predictions', 0, 'gist_yarg', False, strings)
plt.show()
