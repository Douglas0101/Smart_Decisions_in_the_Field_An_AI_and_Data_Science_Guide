# src/models/tune_xgboost_model.py

import pandas as pd
import xgboost as xgb
import logging
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT_PATH / "data" / "processed"


def tune_hyperparameters():
    """
    Carrega os dados, define um grid de hiperparâmetros e usa GridSearchCV
    para encontrar a melhor combinação para o modelo XGBoost.
    """
    logging.info("--- INICIANDO PIPELINE DE OTIMIZAÇÃO DE HIPERPARÂMETROS ---")

    # --- Carregando a Tabela Analítica Mestre ---
    master_table_path = PROCESSED_DATA_DIR / 'master_analytical_table.parquet'
    if not master_table_path.exists():
        logging.error(f"Tabela analítica mestre não encontrada. Execute o pré-processamento.")
        return

    df = pd.read_parquet(master_table_path)
    logging.info("Tabela analítica carregada.")

    # --- Engenharia de Features (a mesma do script de treino) ---
    df['mes'] = df['data'].dt.month
    df['trimestre'] = df['data'].dt.quarter
    df = pd.get_dummies(df, columns=['uf', 'especie_normalizada'], prefix=['uf', 'especie'])
    df = df.sort_values('data').reset_index(drop=True)

    # --- Divisão Temporal ---
    # Para o GridSearch, é importante usar uma validação cruzada que respeite a ordem temporal.
    # No entanto, para um exemplo mais simples, vamos usar a mesma divisão de treino/teste.
    # Em um cenário avançado, usaríamos TimeSeriesSplit do scikit-learn.
    test_size = int(len(df) * 0.2)
    split_index = len(df) - test_size
    train_df = df.iloc[:split_index]

    X_train = train_df.drop(columns=['quantidade_cabecas', 'data', 'ano'])
    y_train = train_df['quantidade_cabecas']

    logging.info(f"Dados preparados para otimização com {len(X_train)} amostras.")

    # --- Definição do Grid de Parâmetros ---
    # NOTA: Este é um grid pequeno para fins de demonstração. Em um projeto real,
    # você exploraria uma gama maior de valores.
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [500, 1000],
        'colsample_bytree': [0.7, 0.8]
    }

    # --- Configuração do Modelo e do GridSearchCV ---
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    # Usamos RMSE como a métrica para otimizar. `neg_mean_squared_error` é o padrão,
    # e o GridSearchCV tentará maximizá-lo (por isso o negativo).
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                              greater_is_better=False)

    grid_search = GridSearchCV(
        estimator=xgbr,
        param_grid=param_grid,
        scoring=rmse_scorer,
        cv=3,  # Usando validação cruzada de 3 folds. Para séries temporais, TimeSeriesSplit é mais indicado.
        verbose=2  # Mostra o progresso
    )

    logging.info("Iniciando a busca em grade (Grid Search)... Isso pode demorar.")
    grid_search.fit(X_train, y_train)

    # --- Exibição dos Melhores Resultados ---
    logging.info("Busca em grade concluída.")
    print("\n=========================================================")
    print("      MELHORES HIPERPARÂMETROS ENCONTRADOS")
    print("=========================================================")
    print(f"Melhor pontuação (RMSE): {-grid_search.best_score_:,.0f}")
    print("Melhores parâmetros:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    print("=========================================================")


if __name__ == "__main__":
    tune_hyperparameters()

