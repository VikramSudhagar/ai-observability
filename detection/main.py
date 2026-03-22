from importlib.resources import files
import sys
from xml.parsers.expat import model
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import os
import glob

WINDOW_SIZE = 48
NUM_CATEGORIES = 2

'''
Next steps, group multiple incidents into one. Take into account the duration of the event and the severity of the event

Current implementation cannot determine whether a spike occurs at a certain time everyday, it will only be able to determine abnormalities within it's short window
'''
def main():
    
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    values, labels = load_data(sys.argv[1])
    labels = tf.keras.utils.to_categorical(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(values), np.array(labels), test_size=0.2
    )
    y_integers = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    model = get_model()
    model.fit(x_train, y_train, epochs=30, class_weight=class_weight_dict)
    model.evaluate(x_test,  y_test, verbose=2)
    
    return 0

def load_data(data_dir, window_size=WINDOW_SIZE):
    # 1. Get a list of all CSV files in the directory
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    all_features = []
    all_labels = []

    for file in files:
        df = pd.read_csv(file)
        df['Value'] = df['Value'].ffill().bfill()
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        
        # Feature Engineering (Normalized)
        df['hour'] = df['TimeStamp'].dt.hour / 23.0
        df['day'] = df['TimeStamp'].dt.dayofweek / 6.0
        df['Value_Diff'] = df['Value'].diff().fillna(0)
        std = df['Value_Diff'].std()
        df['Value_Diff'] = (df['Value_Diff'] - df['Value_Diff'].mean()) / (std if std > 0 else 1.0)
        
        p95 = df['Value'].quantile(0.95)
        df['Value'] = df['Value'] / (p95 if p95 > 0 else 1.0)
        df['Value'] = df['Value'].clip(0, 1)
        
        features = df[['Value', 'hour', 'day', 'Value_Diff']].values
        labels = df['Label'].values

        for i in range(len(features) - window_size + 1):
            all_features.append(features[i : i + window_size])
            all_labels.append(labels[i + window_size - 1])
    
    # 3. Convert lists to final NumPy arrays
    # This will naturally be in the (Samples, Window_Size, Features) shape
    return np.array(all_features), np.array(all_labels)

def get_model():
    
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        tf.keras.layers.LSTM(64, input_shape=(WINDOW_SIZE, 4), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        
        # Second LSTM: return_sequences=False because we want one summary vector
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),

        # Dense layers for classification
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer (NUM_CATEGORIES should be 2 if it's just Normal vs Anomaly)
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        # Using CategoricalCrossentropy because you used to_categorical in main
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")]
    )
    return model
    
    

if __name__ == "__main__":
    main()
