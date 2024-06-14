# Digit Recognizer
### 👉 [Click here to enter the competition](https://www.kaggle.com/competitions/digit-recognizer)

### Competition Description
MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. 
Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. 
As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. 
We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. 
We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

### Dataset
- `train.csv` - the training set.
- `test.csv` - the test set. The testing data contains the same information as the training set, but without the label column, it needs to be predicted.
- `sample_submission.csv` -Your submission file should be in the following format: For each of the 28000 images in the test set, output a single line containing the ImageId and the digit you predict. For example, if you predict that the first image is of a 3, the second image is of a 7, and the third image is of a 8, then your submission file would look like:
    ```
    ImageId,Label
    1,3
    2,7
    3,8 
    (27997 more lines)
    ```
    
### Evaluation
- Test set accuracy: 0.99289

### Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# 隨機種子確保可重現性
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# 設置環境變量確保使用單一CPU核心
import os
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 清除Keras的後端會話
tf.keras.backend.clear_session()

# 載入資料
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 訓練和驗證資料
X = train.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
y = to_categorical(train['label'].values)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=seed)

# 測試資料
X_test = test.values.reshape(-1, 28, 28, 1)

# 建立模型
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 訓練模型
model = create_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(X_train, y_train, epochs=200, batch_size=200, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, reduce_lr], verbose=2)

# 進行預測
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# 輸出檔案
submission = pd.DataFrame({'ImageId': np.arange(1, len(predicted_labels) + 1), 'Label': predicted_labels})
submission.to_csv('submission.csv', index=False)

print("Submission file has been created!")

