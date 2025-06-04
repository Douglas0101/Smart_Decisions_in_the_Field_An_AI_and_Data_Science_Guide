# src/models/train_lstm_model.py (Versão de Otimização Avançada - Final)

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# TensorFlow e KerasTuner para Deep Learning e Otimização
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt

# --- Importando Módulos do Projeto ---
import sys

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT_PATH))
from src.models.evaluate_model import calculate_forecasting_metrics, print_evaluation_report

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROCESSED_DATA_DIR = PROJECT_ROOT_PATH / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT_PATH / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT_PATH / "models"
TUNER_DIR = PROJECT_ROOT_PATH / "tuner"  # Diretório para os resultados do KerasTuner
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TUNER_DIR.mkdir(parents=True, exist_ok=True)


def create_tf_dataset(X, y, batch_size, shuffle=True):
    """Cria um pipeline de dados de alta performance com tf.data."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X), seed=42)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def scale_and_reshape_data(df, sequence_length=12):
    """Prepara os dados para LSTM: escala e cria sequências."""
    logging.info(f"Escalonando e remodelando dados para sequências de {sequence_length} passos...")
    target_col = 'quantidade_cabecas'
    features_df = df.drop(columns=['data', 'ano'])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features_df)

    X, y = [], []
    target_idx = features_df.columns.get_loc(target_col)
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i - sequence_length:i])
        y.append(scaled_features[i, target_idx])

    return np.array(X), np.array(y), scaler, features_df.columns


def run_lstm_tuning_and_training_pipeline(epochs=100, batch_size=64, sequence_length=12):
    logging.info("--- INICIANDO PIPELINE DE OTIMIZAÇÃO E TREINAMENTO DEEP LEARNING ---")

    # --- Preparação de Dados ---
    master_table_path = PROCESSED_DATA_DIR / 'master_analytical_table.parquet'
    df = pd.read_parquet(master_table_path)
    df['mes'] = df['data'].dt.month
    df['trimestre'] = df['data'].dt.quarter
    df = pd.get_dummies(df, columns=['uf', 'especie_normalizada'], prefix=['uf', 'especie'])
    df = df.sort_values('data').reset_index(drop=True)

    target_col = 'quantidade_cabecas'
    cols = [target_col] + [col for col in df.columns if col != target_col]
    df = df[cols]

    X, y, scaler, feature_names = scale_and_reshape_data(df, sequence_length)
    train_size = int(len(X) * 0.8)
    X_train, X_val, y_train, y_val = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # --- Otimização de Hiperparâmetros com KerasTuner ---
    logging.info("Iniciando a busca por melhores hiperparâmetros com KerasTuner...")

    def build_model(hp):
        """Função que constrói o modelo para o KerasTuner."""
        model = Sequential([
            Input(shape=(X.shape[1], X.shape[2])),
            LSTM(units=hp.Int('lstm_1_units', min_value=50, max_value=150, step=25), return_sequences=True),
            Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)),
            LSTM(units=hp.Int('lstm_2_units', min_value=30, max_value=100, step=20), return_sequences=False),
            Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)),
            Dense(units=hp.Int('dense_units', min_value=20, max_value=50, step=10), activation='relu'),
            Dense(1)
        ])

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mean_squared_error')
        return model

    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=30,
        factor=3,
        directory=TUNER_DIR,
        project_name='agro_lstm_tuning',
        overwrite=True
    )

    # --- Callbacks para Busca e Treinamento ---
    model_path = MODELS_DIR / "lstm_tuned_best_model.keras"
    # O ModelCheckpoint agora é usado para salvar o melhor modelo encontrado durante a busca
    callbacks_for_search = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=callbacks_for_search)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(f"Busca concluída. Melhores hiperparâmetros: {best_hps.values}")

    # --- Avaliação Final ---
    logging.info("Avaliando o melhor modelo salvo do disco...")

    # *** CORREÇÃO APLICADA: Carregamos o melhor modelo salvo pelo ModelCheckpoint ***
    best_model = load_model(model_path)

    y_pred_scaled = best_model.predict(X_val)

    # Inverte o escalonamento para obter os valores reais
    dummy_array_pred = np.zeros(shape=(len(y_pred_scaled), len(feature_names)))
    target_idx = list(feature_names).index(target_col)
    dummy_array_pred[:, target_idx] = y_pred_scaled.ravel()
    y_pred = scaler.inverse_transform(dummy_array_pred)[:, target_idx]

    dummy_array_test = np.zeros(shape=(len(y_val), len(feature_names)))
    dummy_array_test[:, target_idx] = y_val.ravel()
    y_test_original = scaler.inverse_transform(dummy_array_test)[:, target_idx]

    metrics = calculate_forecasting_metrics(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    metrics['r2'] = r2
    print_evaluation_report(metrics)
    logging.info(f"Coeficiente de Determinação (R²): {r2:.4f}")

    # --- Visualização ---
    test_dates = df['data'].iloc[train_size + sequence_length:]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(test_dates, y_test_original, label='Valores Reais', marker='.')
    ax.plot(test_dates, y_pred, label='Previsões (LSTM Otimizado)', marker='x', alpha=0.8)
    ax.set_title('Validação do Modelo LSTM Otimizado', fontsize=16)
    ax.legend()
    fig.savefig(REPORTS_DIR / "lstm_tuned_real_vs_previsao.png")
    plt.show()


if __name__ == '__main__':
    run_lstm_tuning_and_training_pipeline()
