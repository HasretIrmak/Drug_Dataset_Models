# Gerekli kütüphaneler
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, SimpleRNN, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Veri setini yükleme
from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name)

# Kategorik verileri kodlama
for column in data.columns:
    if data[column].dtype == 'object':  # Eğer sütun string içeriyorsa
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

# Özellik ve etiket ayrımı
X = data.iloc[:, :-1].values  # Özellikler
y = data.iloc[:, -1].values   # Etiketler

# Etiketleri kategorik hale getirme
y = to_categorical(y)

# Özelliklerin ölçeklendirilmesi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CNN, RNN, LSTM ve GRU için veriyi 3 boyutlu hale getirme
X_scaled_3d = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Model oluşturma fonksiyonları
def create_dnn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_fnn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(64, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(input_shape):
    model = Sequential([
        GRU(64, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Değerlendirme fonksiyonu
def evaluate_model(model_creator, X, y, input_shape=None, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores, f1_scores, recall_scores, precision_scores = [], [], [], []
    fold_accuracies = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Giriş şeklinin doğrulanması ve yeniden şekillendirme
        if input_shape:
            X_train = X_train.reshape(X_train.shape[0], *input_shape)
            X_test = X_test.reshape(X_test.shape[0], *input_shape)

        model = model_creator(X_train.shape[1:] if input_shape else X_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        acc = accuracy_score(y_test_classes, y_pred_classes)
        fold_accuracies.append((fold, acc))

        accuracy_scores.append(acc)
        f1_scores.append(f1_score(y_test_classes, y_pred_classes, average='weighted'))
        recall_scores.append(recall_score(y_test_classes, y_pred_classes, average='weighted'))
        precision_scores.append(precision_score(y_test_classes, y_pred_classes, average='weighted'))

    print(f"Fold Scores: {fold_accuracies}")
    return {
        'accuracy': np.mean(accuracy_scores),
        'f1_score': np.mean(f1_scores),
        'recall': np.mean(recall_scores),
        'precision': np.mean(precision_scores),
        'fold_accuracies': fold_accuracies
    }

# Modellerin değerlendirilmesi
results = {}
results['DNN'] = evaluate_model(create_dnn_model, X_scaled, y)
results['FNN'] = evaluate_model(create_fnn_model, X_scaled, y)
results['RNN'] = evaluate_model(create_rnn_model, X_scaled_3d, y, input_shape=(X_scaled.shape[1], 1))
results['LSTM'] = evaluate_model(create_lstm_model, X_scaled_3d, y, input_shape=(X_scaled.shape[1], 1))
results['GRU'] = evaluate_model(create_gru_model, X_scaled_3d, y, input_shape=(X_scaled.shape[1], 1))
results['CNN'] = evaluate_model(create_cnn_model, X_scaled_3d, y, input_shape=(X_scaled.shape[1], 1))

# Sayısal sonuçların yazdırılması
print("Model Performances (K-Fold Cross Validation):")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        if metric == 'fold_accuracies':
            print(f"  Fold Accuracies: {value}")
        else:
            print(f"  {metric.capitalize()}: {value:.4f}")

# Sonuçların görselleştirilmesi
metrics = ['accuracy', 'f1_score', 'recall', 'precision']
for metric in metrics:
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), [results[model][metric] for model in results], color=['blue', 'green', 'red', 'purple', 'orange', 'cyan'])
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.ylabel(metric.capitalize())
    plt.show()
