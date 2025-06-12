# src/models/evaluate_model.py

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuração do logging para o módulo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


class ModelEvaluator:
    """
    Encapsula a lógica de avaliação para modelos de previsão de séries temporais.

    Esta classe fornece métodos para dividir dados, avaliar a performance de
    um modelo em um cenário específico e calcular métricas de erro.
    """

    def __init__(self, validation_start_date: str):
        """
        Inicializa o avaliador com a data de corte para validação.

        Args:
            validation_start_date (str): A data (formato 'YYYY-MM-DD') que marca o início
                                         do conjunto de validação.
        """
        try:
            self.validation_start_date = pd.to_datetime(validation_start_date)
            logging.info(
                f"ModelEvaluator inicializado com data de validação a partir de: {self.validation_start_date.date()}")
        except ValueError:
            logging.error(f"Formato de data inválido para 'validation_start_date'. Use 'YYYY-MM-DD'.")
            raise

    def split_data(self, df: pd.DataFrame, date_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide um DataFrame em conjuntos de treino e validação com base em uma data.

        Args:
            df (pd.DataFrame): O DataFrame completo contendo os dados históricos.
            date_column (str): O nome da coluna que contém as datas.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Uma tupla contendo o DataFrame de treino
                                               e o DataFrame de validação.
        """
        df[date_column] = pd.to_datetime(df[date_column])

        df_train = df[df[date_column] < self.validation_start_date].copy()
        df_validation = df[df[date_column] >= self.validation_start_date].copy()

        logging.info(f"Dados divididos: {len(df_train)} registros para treino, {len(df_validation)} para validação.")

        return df_train, df_validation

    def evaluate_scenario(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula as métricas de erro para um conjunto de previsões.

        Args:
            y_true (pd.Series): Os valores reais (ground truth).
            y_pred (np.ndarray): Os valores previstos pelo modelo (array NumPy).

        Returns:
            Dict[str, float]: Um dicionário contendo as métricas MAE, RMSE e MAPE.
        """
        # [CORREÇÃO] Altera a verificação de y_pred.empty para y_pred.size == 0,
        # que é a forma correta de verificar se um array NumPy está vazio.
        if y_true.empty or y_pred.size == 0:
            logging.warning("Conjunto de dados de avaliação vazio. Pulando cálculo de métricas.")
            return {}

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        y_true_no_zeros = y_true.replace(0, np.nan).dropna()
        if len(y_true_no_zeros) == 0:
            mape = np.inf
        else:
            # Garante que y_pred seja alinhado com os índices de y_true_no_zeros
            y_pred_mape = pd.Series(y_pred, index=y_true.index)[y_true_no_zeros.index]
            mape = (np.abs((y_true_no_zeros - y_pred_mape) / y_true_no_zeros)).mean() * 100

        metrics = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE (%)": round(mape, 4)
        }

        logging.info(f"Métricas calculadas: {metrics}")
        return metrics
