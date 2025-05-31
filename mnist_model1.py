import keras
from keras import layers
from keras.datasets import mnist


(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 *28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 *28))
test_images = test_images.astype('float32') / 255


model = keras.Sequential([
    layers.Dense(units=256, activation='relu'),
    layers.Dense(units=10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=train_images, y=train_labels, epochs=16, batch_size=128)

model.save(filepath='mnist_reader.keras')
