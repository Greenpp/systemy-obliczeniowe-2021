# %%
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import Model, Sequential
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

RANDOM_SEED = 1410
# %%
(X_train, y_train), (X_test, y_test)= load_data() 
# %%
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.ravel()
for i, img in enumerate(X_train[:4]):
    ax[i].imshow(img, cmap='gray')
plt.show()
# %%
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)).astype('float32') / 255.0
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)).astype('float32') / 255.0

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
# %%
IMG_HEIGHT, IMG_WIDTH = 28, 28
NUM_CLASSES = 10
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax'),
])

# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# %%
history = model.fit(X_train, y_train_cat, epochs=3, validation_data=(X_test, y_test_cat), batch_size=32)
# %%
tr_clf = DecisionTreeClassifier(random_state=RANDOM_SEED)
encoder = Model(model.inputs, model.layers[-2].output)

# %%
X_flat_train = X_train.reshape(-1, IMG_HEIGHT * IMG_WIDTH)
X_flat_test = X_test.reshape(-1, IMG_HEIGHT * IMG_WIDTH)

X_encoded_train = encoder.predict(X_train, batch_size=128)
X_encoded_test = encoder.predict(X_test, batch_size=128)

# %%
scores = {}

clf = clone(tr_clf)
clf.fit(X_flat_train, y_train)
res = clf.score(X_flat_test, y_test)
scores['No encoding'] = res


clf = clone(tr_clf)
clf.fit(X_encoded_train, y_train)
res = clf.score(X_encoded_test, y_test)
scores['Encoding'] = res

# %%
scores

# %%
