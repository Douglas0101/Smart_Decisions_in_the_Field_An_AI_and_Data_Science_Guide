# src/models/tune_xgboost_model.py

import pandas as pd
import xgboost as xgb
import optuna
import logging
from typing import Dict, Any
from sklearn.metrics import mean_squared_error
import numpy as np

# Desativa o logging detalhado do Optuna para manter a saída limpa
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


class HyperparameterTuner:
    """
    Encapsula a lógica de otimização de hiperparâmetros, utilizando a API nativa do XGBoost
    para máxima compatibilidade e robustez.
    """

    def __init__(self, df_train: pd.DataFrame, df_val: pd.DataFrame, target_col: str):
        """
        Inicializa o otimizador com os dados de treino e validação.
        """
        self.target_col = target_col
        self.feature_cols = [col for col in df_train.columns if col not in [target_col, 'data']]

        # [SOLUÇÃO AVANÇADA] Converte os DataFrames para DMatrix, o formato de dados nativo do XGBoost.
        # Isto é feito uma vez para otimizar a performance durante os trials.
        self.dtrain = xgb.DMatrix(df_train[self.feature_cols], label=df_train[self.target_col])
        self.dval = xgb.DMatrix(df_val[self.feature_cols], label=df_val[self.target_col])
        self.y_val = df_val[self.target_col]  # Mantemos y_val para o cálculo final do erro.

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Função objetivo que o Optuna tentará minimizar.
        """
        # O espaço de busca de parâmetros permanece o mesmo
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'n_jobs': -1
        }

        # [SOLUÇÃO AVANÇADA] Utiliza xgb.train (API nativa) em vez de model.fit()
        # A API nativa aceita 'early_stopping_rounds' de forma direta e fiável.
        bst = xgb.train(
            params=params,
            dtrain=self.dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(self.dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Realiza previsões com o modelo treinado (Booster)
        preds = bst.predict(self.dval)
        rmse = np.sqrt(mean_squared_error(self.y_val, preds))

        return rmse

    def tune(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Executa o processo de otimização de hiperparâmetros.
        """
        logging.info(f"Iniciando a otimização de hiperparâmetros com {n_trials} trials (usando API nativa)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective, n_trials=n_trials)

        best_trial = study.best_trial
        result = {
            "best_score_rmse": round(best_trial.value, 4),
            "best_params": best_trial.params
        }

        logging.info(f"Otimização concluída. Melhor RMSE: {result['best_score_rmse']}")
        logging.info(f"Melhores parâmetros encontrados: {result['best_params']}")

        return result
