# src/models/train_multivariate_model.py (Versão Final Corrigida)

import pandas as pd
import xgboost as xgb
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# --- Importando Módulos do Projeto ---
import sys

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT_PATH))
from src.models.evaluate_model import calculate_forecasting_metrics, print_evaluation_report

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROCESSED_DATA_DIR = PROJECT_ROOT_PATH / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT_PATH / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT_PATH / "models"  # Pasta para salvar o modelo treinado
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_time_series_features(df, target_col):
    """
    IMPLEMENTAÇÃO AVANÇADA: Cria features baseadas no tempo, como lags e janelas móveis.
    Essas features dão ao modelo uma "memória" do que aconteceu no passado recente.
    """
    logging.info("Criando features temporais avançadas (lags e janelas móveis)...")

    # É importante criar lags para cada grupo (UF e Espécie) independentemente
    df = df.sort_values(by=['uf', 'especie_normalizada', 'data'])

    # Lags: O valor do alvo em períodos anteriores
    df[f'{target_col}_lag_1'] = df.groupby(['uf', 'especie_normalizada'])[target_col].shift(1)
    df[f'{target_col}_lag_3'] = df.groupby(['uf', 'especie_normalizada'])[target_col].shift(3)
    df[f'{target_col}_lag_12'] = df.groupby(['uf', 'especie_normalizada'])[target_col].shift(12)  # Sazonalidade anual

    # Janelas Móveis: A tendência recente
    df[f'{target_col}_rolling_mean_3'] = df.groupby(['uf', 'especie_normalizada'])[target_col].transform(
        lambda x: x.shift(1).rolling(window=3).mean()
    )
    df[f'{target_col}_rolling_std_3'] = df.groupby(['uf', 'especie_normalizada'])[target_col].transform(
        lambda x: x.shift(1).rolling(window=3).std()
    )

    # Remove os NaNs gerados pelos lags e rolling windows iniciais
    df.dropna(inplace=True)
    return df


def run_optimized_modeling_pipeline():
    """
    Executa o pipeline de modelagem usando os melhores hiperparâmetros
    e features temporais avançadas.
    """
    logging.info("--- INICIANDO PIPELINE DE MODELAGEM COM HIPERPARÂMETROS OTIMIZADOS E FEATURES AVANÇADAS ---")

    # --- Carregando a Tabela Analítica Mestre ---
    master_table_path = PROCESSED_DATA_DIR / 'master_analytical_table.parquet'
    if not master_table_path.exists():
        logging.error(f"Tabela analítica mestre não encontrada. Execute o pré-processamento.")
        return

    df = pd.read_parquet(master_table_path)
    logging.info("Tabela analítica carregada.")

    target_col = 'quantidade_cabecas'  # Definição da variável alvo

    # --- Engenharia de Features ---
    df['mes'] = df['data'].dt.month
    df['trimestre'] = df['data'].dt.quarter
    df = create_time_series_features(df, target_col)
    df = pd.get_dummies(df, columns=['uf', 'especie_normalizada'], prefix=['uf', 'especie'])
    df = df.sort_values('data').reset_index(drop=True)
    logging.info(f"Engenharia de features concluída. Shape final dos dados: {df.shape}")

    # --- Divisão Temporal ---
    test_size = int(len(df) * 0.2)
    split_index = len(df) - test_size
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    features_to_drop = [target_col, 'data', 'ano']
    X_train = train_df.drop(columns=features_to_drop)
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=features_to_drop)
    y_test = test_df[target_col]

    logging.info(f"Dados preparados. Treino: {len(X_train)} linhas, Teste: {len(X_test)} linhas.")

    # ==========================================================================
    # ESTIMAÇÃO DO MODELO OTIMIZADO
    # ==========================================================================
    logging.info("Iniciando o treinamento do modelo XGBoost OTIMIZADO...")

    best_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5,
        'colsample_bytree': 0.7, 'subsample': 0.8, 'eval_metric': 'rmse',
        'n_jobs': -1, 'random_state': 42
    }

    xgbr_optimized = xgb.XGBRegressor(**best_params)
    xgbr_optimized.fit(X_train, y_train, verbose=False)
    logging.info("Treinamento do modelo otimizado concluído.")

    # --- Salvando o Modelo Treinado ---
    model_path = MODELS_DIR / "xgboost_optimized_model.json"
    xgbr_optimized.save_model(model_path)
    logging.info(f"Modelo treinado salvo com sucesso em: {model_path}")

    # ==========================================================================
    # VALIDAÇÃO DO MODELO OTIMIZADO
    # ==========================================================================
    logging.info("Avaliando o modelo OTIMIZADO no conjunto de teste...")
    y_pred = xgbr_optimized.predict(X_test)
    metrics = calculate_forecasting_metrics(y_test, y_pred)
    print_evaluation_report(metrics)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Coeficiente de Determinação (R²): {r2:.3f}")

    # --- Geração de Gráficos ---
    # Gráfico de Validação
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(test_df['data'], y_test.values, label='Valores Reais', marker='.', linestyle='-')
    ax.plot(test_df['data'], y_pred, label='Previsões (XGBoost Otimizado)', marker='x', linestyle='--')
    ax.set_title('Validação Final: Modelo Otimizado vs. Valores Reais', fontsize=16)
    ax.legend()
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "xgboost_optimized_real_vs_previsao.png")
    plt.show()

    # Gráfico de Importância de Features
    logging.info("Gerando a importância das features do modelo OTIMIZADO...")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgbr_optimized.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance, hue='feature', palette='plasma', ax=ax,
                dodge=False)

    # === CORREÇÃO APLICADA AQUI ===
    # Verifica se a legenda existe antes de tentar removê-la
    legend = ax.get_legend()
    if legend:
        legend.remove()

    ax.set_title('Importância das Features (Modelo Otimizado com Features Temporais)', fontsize=16)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "xgboost_optimized_feature_importance.png")
    plt.show()

    logging.info("--- PIPELINE DE MODELAGEM OTIMIZADA CONCLUÍDO ---")


if __name__ == "__main__":
    run_optimized_modeling_pipeline()
