import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, SimpleRNN
from tensorflow.keras.utils import to_categorical

# Usage
classifier = DiabetesDrugInteractionClassifier('/content/drive/My Drive/indir.csv')  # Update with the correct path
classifier.run_analysis()

class DiabetesDrugInteractionClassifier:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, encoding='utf-8')
        self.ml_models = {
            'Random Forest': RandomForestClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier()
        }

        self.dl_models = {
            'RNN': self._build_rnn_model,
            'CNN': self._build_cnn_model,
            'LSTM': self._build_lstm_model,
            'ANN': self._build_ann_model,
            'GRU': self._build_gru_model,
            'FNN': self._build_fnn_model
        }

    def preprocess_data(self):
        # Convert categorical columns to numeric
        le = LabelEncoder()
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = le.fit_transform(self.data[col])

        # Assuming last column is target, others are features
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y

    def _build_base_dl_model(self, input_shape):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_rnn_model(self, input_shape):
        model = Sequential([
            SimpleRNN(64, input_shape=(1, input_shape)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_cnn_model(self, input_shape):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=(1, input_shape)),
            MaxPooling1D(2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(64, input_shape=(1, input_shape)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_gru_model(self, input_shape):
        model = Sequential([
            GRU(64, input_shape=(1, input_shape)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_ann_model(self, input_shape):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_fnn_model(self, input_shape):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def cross_validate(self, X, y, model_type='ml'):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}

        if model_type == 'ml':
            for name, model in self.ml_models.items():
                fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

                for fold, (train_index, val_index) in enumerate(kf.split(X, y), 1):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                    accuracy = accuracy_score(y_val, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')

                    fold_results['accuracy'].append(accuracy)
                    fold_results['precision'].append(precision)
                    fold_results['recall'].append(recall)
                    fold_results['f1'].append(f1)

                results[name] = fold_results

        else:  # Deep Learning Models
            X = X.reshape(-1, 1, X.shape[1])
            input_shape = X.shape[2]

            for name, model_builder in self.dl_models.items():
                fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

                for fold, (train_index, val_index) in enumerate(kf.split(X, y), 1):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    model = model_builder(input_shape)
                    model.fit(X_train, y_train, epochs=50, verbose=0)

                    y_pred = (model.predict(X_val) > 0.5).astype(int)

                    accuracy = accuracy_score(y_val, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')

                    fold_results['accuracy'].append(accuracy)
                    fold_results['precision'].append(precision)
                    fold_results['recall'].append(recall)
                    fold_results['f1'].append(f1)

                results[name] = fold_results

        return results

    def run_analysis(self):
        X, y = self.preprocess_data()

        print("Machine Learning Models Cross-Validation Results:")
        ml_results = self.cross_validate(X, y, model_type='ml')
        for model, metrics in ml_results.items():
            print(f"\n{model} Results:")
            for metric, values in metrics.items():
                print(f"{metric.capitalize()}: {values}")

        print("\nDeep Learning Models Cross-Validation Results:")
        dl_results = self.cross_validate(X, y, model_type='dl')
        for model, metrics in dl_results.items():
            print(f"\n{model} Results:")
            for metric, values in metrics.items():
                print(f"{metric.capitalize()}: {values}")

# Usage
classifier = DiabetesDrugInteractionClassifier('diabetes_drug_inetractions_and_risks.csv')
classifier.run_analysis()
