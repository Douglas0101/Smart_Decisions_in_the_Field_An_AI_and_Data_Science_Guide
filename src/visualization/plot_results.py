# src/visualization/plot_results.py

import pandas as pd
import plotly.graph_objects as go
import os

# --- Caminhos e Constantes ---
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)  # Cria a pasta de figuras se não existir


def plot_forecast_vs_actual(train_data, test_data, forecast_data, model_name="Modelo"):
    """
    Cria e exibe um gráfico interativo comparando os dados de treino,
    dados reais de teste e a previsão do modelo.

    Args:
        train_data (pd.Series): Série temporal de treinamento.
        test_data (pd.Series): Série temporal de teste (valores reais).
        forecast_data (pd.Series): Série temporal com os valores previstos.
        model_name (str): Nome do modelo para usar no título e na legenda.
    """
    print(f"Gerando gráfico de resultados para o {model_name}...")

    # Cria a figura
    fig = go.Figure()

    # Adiciona a linha de dados de TREINO
    fig.add_trace(go.Scatter(
        x=train_data.index,
        y=train_data,
        mode='lines',
        name='Dados de Treino',
        line=dict(color='royalblue')
    ))

    # Adiciona a linha de dados REAIS de TESTE
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=test_data,
        mode='lines+markers',
        name='Valores Reais (Teste)',
        line=dict(color='darkorange')
    ))

    # Adiciona a linha de PREVISÃO
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data,
        mode='lines+markers',
        name='Previsão do Modelo',
        line=dict(color='mediumseagreen', dash='dot')
    ))

    # Configurações do layout do gráfico
    fig.update_layout(
        title=f"Comparação de Previsão: {model_name} vs. Dados Reais",
        xaxis_title="Data",
        yaxis_title="Quantidade de Cabeças",
        legend_title="Séries",
        template="plotly_white",
        font=dict(family="Arial", size=12)
    )

    # Exibe o gráfico interativo
    fig.show()

    # Opcional: Salvar a figura como um arquivo HTML interativo
    file_name = f"forecast_{model_name.replace(' ', '_').lower()}.html"
    file_path = os.path.join(FIGURES_DIR, file_name)
    fig.write_html(file_path)
    print(f"Gráfico interativo salvo em: {file_path}")


# --- Bloco de Execução Principal ---
if __name__ == '__main__':
    # Para executar este script de forma independente, precisamos chamar as funções dos módulos anteriores
    from src.features.build_features import create_time_series_for_modeling
    from src.models.train_model import split_time_series, train_sarima_model, forecast_model

    # Caminho para o arquivo de dados limpos
    CLEAN_ABATE_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed',
                                    'abate_animal_limpo.csv')

    print("\n--- Testando o Módulo de Visualização ---")

    # 1. Preparar os dados
    ts_aves = create_time_series_for_modeling(CLEAN_ABATE_FILE, 'AVES')
    if ts_aves is not None:
        train, test = split_time_series(ts_aves, test_size=12)

        # 2. Treinar e prever
        modelo = train_sarima_model(train)
        previsao = forecast_model(modelo, steps=len(test))

        # 3. Plotar os resultados
        plot_forecast_vs_actual(train, test, previsao, model_name="SARIMA Baseline")