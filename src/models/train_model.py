# src/models/train_model.py

import pandas as pd
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet  # <<< ADICIONADO
import warnings

# Ignora avisos comuns para uma saída mais limpa
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Caminhos e Constantes ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# --- Funções Existentes (sem alteração) ---
def split_time_series(ts, test_size=12):
    print(f"Dividindo a série temporal: {len(ts) - test_size} para treino, {test_size} para teste.")
    train_data = ts.iloc[:-test_size]
    test_data = ts.iloc[-test_size:]
    return train_data, test_data


def train_sarima_model(train_data):
    print("Treinando o modelo SARIMA... Isso pode levar alguns minutos.")
    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    print("Treinamento do modelo SARIMA concluído.")
    return results


def forecast_model(model_results, steps):
    print(f"Gerando previsão SARIMA para {steps} passos à frente...")
    forecast = model_results.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean
    return mean_forecast


# --- NOVA FUNÇÃO PARA O PROPHET ---
def train_prophet_model(train_data):
    """
    Treina um modelo Prophet.

    Args:
        train_data (pd.Series): A série temporal de treinamento.

    Returns:
        Prophet: O objeto do modelo treinado.
    """
    print("Treinando o modelo Prophet...")

    # O Prophet exige que o DataFrame tenha colunas com nomes específicos: 'ds' e 'y'
    prophet_df = train_data.reset_index()
    prophet_df.columns = ['ds', 'y']

    # Instancia e treina o modelo
    # Adicionamos a sazonalidade anual, que é forte em nossos dados.
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    model.fit(prophet_df)

    print("Treinamento do modelo Prophet concluído.")
    return model


def forecast_prophet_model(model, steps):
    """
    Gera previsões a partir de um modelo Prophet treinado.
    """
    print(f"Gerando previsão Prophet para {steps} passos à frente...")

    # Cria um DataFrame futuro para fazer as previsões
    future = model.make_future_dataframe(periods=steps, freq='MS')
    forecast = model.predict(future)

    # A previsão está na coluna 'yhat', e queremos apenas os valores do período de previsão
    mean_forecast = forecast['yhat'].iloc[-steps:]
    # Ajusta o índice para corresponder ao formato da série original
    mean_forecast.index = future['ds'].iloc[-steps:]

    return mean_forecast


# --- Bloco de Execução Principal (Atualizado) ---
if __name__ == '__main__':
    from src.features.build_features import create_time_series_for_modeling
    from src.models.evaluate_model import calculate_forecasting_metrics, print_evaluation_report

    CLEAN_ABATE_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed',
                                    'abate_animal_limpo.csv')

    print("\n--- Testando o Pipeline de Treinamento de Modelo (SARIMA vs. Prophet) ---")
    ts_aves = create_time_series_for_modeling(CLEAN_ABATE_FILE, 'AVES')

    if ts_aves is not None:
        train_set, test_set = split_time_series(ts_aves, test_size=12)

        # --- Bloco SARIMA ---
        print("\n" + "=" * 20 + " MODELO SARIMA " + "=" * 20)
        modelo_sarima = train_sarima_model(train_set)
        previsao_sarima = forecast_model(modelo_sarima, steps=len(test_set))
        metricas_sarima = calculate_forecasting_metrics(test_set, previsao_sarima)
        print_evaluation_report(metricas_sarima)

        # --- Bloco Prophet ---
        print("\n" + "=" * 20 + " MODELO PROPHET " + "=" * 20)
        modelo_prophet = train_prophet_model(train_set)
        previsao_prophet = forecast_prophet_model(modelo_prophet, steps=len(test_set))
        metricas_prophet = calculate_forecasting_metrics(test_set, previsao_prophet)
        print_evaluation_report(metricas_prophet)