import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train/255.0
x_test=x_test/255.0

y_test=to_categorical(y_test,10)
y_train=to_categorical(y_train,10)

model=Sequential()
model.add(Conv2D(6,kernel_size=(5,5),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=64,validation_split=0.2)
test_loss,test_accuracy=model.evaluate(x_test,y_test)
print(f'test_accuracy{test_accuracy*100:.2f}%')
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)  # Get the class with the highest probability
true_classes = np.argmax(y_test, axis=1)  # Get the true class labels

# Example: Compare predictions with true labels
for i in range(5):
    print(f"Predicted: {predicted_classes[i]}, True: {true_classes[i]}")
    plt.imshow(x_test[i])
    plt.show()