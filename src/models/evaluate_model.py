# src/models/evaluate_model.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_forecasting_metrics(y_true, y_pred):
    """
    Calcula um conjunto de métricas de erro para problemas de previsão de séries temporais.

    Args:
        y_true (array-like): Os valores reais (verdadeiros).
        y_pred (array-like): Os valores previstos pelo modelo.

    Returns:
        dict: Um dicionário contendo as métricas calculadas.
    """
    # Mean Absolute Error (MAE) - Erro médio absoluto. Fácil de interpretar.
    mae = mean_absolute_error(y_true, y_pred)

    # Mean Squared Error (MSE) - Erro quadrático médio. Penaliza mais os erros grandes.
    mse = mean_squared_error(y_true, y_pred)

    # Root Mean Squared Error (RMSE) - Raiz do erro quadrático médio.
    # Está na mesma unidade da variável original, o que facilita a interpretação.
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error (MAPE) - Erro percentual médio absoluto.
    # Cuidado: pode ser problemático se y_true tiver zeros.
    # Adicionamos um pequeno epsilon para evitar divisão por zero.
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }

    return metrics


def print_evaluation_report(metrics):
    """
    Imprime um relatório formatado com as métricas de avaliação.
    """
    print("\n--- Relatório de Avaliação do Modelo ---")
    print(f"  MAE (Erro Absoluto Médio):      {metrics['mae']:,.0f}")
    print(f"  RMSE (Raiz do Erro Quadrático): {metrics['rmse']:,.0f}")
    print(f"  MAPE (Erro Percentual Médio):   {metrics['mape']:.2f}%")
    print("-----------------------------------------")
    print(f"  Interpretação: Em média, as previsões do modelo erram em {metrics['mae']:,.0f} unidades.")


# --- Bloco de Execução Principal ---
if __name__ == '__main__':
    print("\n--- Testando o Módulo de Avaliação ---")

    # Criando dados de exemplo
    valores_reais = np.array([100, 110, 120, 115, 125, 130])
    valores_previstos = np.array([102, 108, 123, 118, 123, 135])

    print("\nDados de Exemplo:")
    print(f"Reais:    {valores_reais}")
    print(f"Previstos: {valores_previstos}")

    # Calculando e imprimindo as métricas
    metricas_exemplo = calculate_forecasting_metrics(valores_reais, valores_previstos)
    print_evaluation_report(metricas_exemplo)