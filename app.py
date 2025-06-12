# app.py

import streamlit as st
# [CORREÇÃO] A importação de 'pandas' foi removida pois não era chamada diretamente.
import json
from pathlib import Path
import plotly.graph_objects as go

# Importa a classe do nosso novo módulo de pipeline
from src.pipeline import PredictionPipeline

# --- Configuração da Página ---
st.set_page_config(
    page_title="Previsão do Agronegócio",
    page_icon="🐄",
    layout="wide"
)


# --- Funções de Cache ---
# O cache do Streamlit evita que os dados e modelos sejam recarregados a cada interação.
@st.cache_data
def load_data(config):
    """Carrega e pré-processa os dados uma única vez."""
    pipeline = PredictionPipeline(config)
    return pipeline.load_and_preprocess_data()


@st.cache_data
def load_tuning_results():
    """Carrega os resultados da otimização de hiperparâmetros."""
    path = Path("reports/hyperparameter_tuning_results.json")
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}


@st.cache_data
def run_forecast_for_scenario(_config, _df_full, _scenario, _params):
    """Executa o pipeline de previsão para um cenário específico."""
    # O sublinhado nos argumentos indica ao Streamlit para não hashear o seu conteúdo,
    # uma vez que já estamos a lidar com o cache nos dados e parâmetros.
    pipeline = PredictionPipeline(_config)
    query = " & ".join([f"`{col}` == '{val}'" for col, val in _scenario.items()])
    df_history = _df_full.query(query)

    if df_history.empty:
        return None, None

    forecast = pipeline.train_and_forecast(df_history, _params, horizon_months=36)
    return df_history, forecast


# --- Interface do Utilizador ---
st.title("📈 Dashboard de Previsão para o Agronegócio")
st.markdown("Selecione um cenário para visualizar as previsões de abate de animais para os próximos 36 meses.")

# Carrega os dados e resultados uma única vez
abate_config = {
    'name': 'abate_animais', 'filename': 'sigsifrelatorioabatesporanouf.csv',
    'target_column': 'quantidade_cabecas', 'date_column': 'data', 'date_format': '%m/%Y',
    'group_by_columns': ['uf', 'especie_normalizada'],
    'column_mapping': {'data': ['MÊS/ANO', 'MES_ANO'], 'uf': ['UF', 'UF_PROCEDENCIA'],
                       'especie_original': ['ANIMAL', 'CATEGORIA'], 'quantidade_cabecas': ['QUANTIDADE', 'QTD']},
    'processing_func': 'process_abate_data'
}
df_abate = load_data(abate_config)
tuning_results = load_tuning_results().get('abate_animais', {})

if df_abate is not None:
    # Seletores para o cenário
    col1, col2 = st.columns(2)
    with col1:
        ufs_disponiveis = sorted(df_abate['uf'].unique())
        uf_selecionada = st.selectbox("Selecione a UF:", ufs_disponiveis,
                                      index=ufs_disponiveis.index('MT') if 'MT' in ufs_disponiveis else 0)

    with col2:
        especies_disponiveis = sorted(df_abate['especie_normalizada'].unique())
        especie_selecionada = st.selectbox("Selecione a Espécie:", especies_disponiveis,
                                           index=especies_disponiveis.index(
                                               'BOVINOS') if 'BOVINOS' in especies_disponiveis else 0)

    # --- Lógica de Backend e Visualização ---
    scenario_selecionado = {'uf': uf_selecionada, 'especie_normalizada': especie_selecionada}
    scenario_key = f"{uf_selecionada}_{especie_selecionada}"

    # Obtém os parâmetros otimizados para o cenário, ou usa os padrão
    params = tuning_results.get(scenario_key, {}).get('best_params', {})
    if not params:
        st.warning("Não foram encontrados parâmetros otimizados para este cenário. A usar parâmetros padrão.")
        params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 'colsample_bytree': 0.7,
                  'subsample': 0.8, 'n_jobs': -1, 'random_state': 42}

    # Executa a previsão
    df_historico, df_previsao = run_forecast_for_scenario(abate_config, df_abate, scenario_selecionado, params)

    if df_historico is None:
        st.error("Não há dados históricos disponíveis para o cenário selecionado.")
    else:
        # Mostra a métrica de confiança (RMSE da otimização)
        rmse = tuning_results.get(scenario_key, {}).get('best_score_rmse', 'N/A')
        st.metric(label="Confiança do Modelo (RMSE da Validação)",
                  value=f"{rmse:,.2f}" if isinstance(rmse, float) else rmse)

        # Cria o gráfico interativo com Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_historico['data'], y=df_historico['quantidade_cabecas'],
            mode='lines+markers', name='Dados Históricos', line=dict(color='royalblue')
        ))

        if df_previsao is not None:
            fig.add_trace(go.Scatter(
                x=df_previsao.index, y=df_previsao[0.5],
                mode='lines', name='Previsão (Mediana)', line=dict(color='darkorange', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=df_previsao.index, y=df_previsao[0.95],
                mode='lines', name='Limite Superior', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df_previsao.index, y=df_previsao[0.05],
                mode='lines', name='Intervalo de Confiança (90%)', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)'
            ))

        fig.update_layout(
            title=f"Previsão de Abate de {especie_selecionada} em {uf_selecionada}",
            xaxis_title="Data", yaxis_title="Quantidade de Cabeças",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Não foi possível carregar os dados de abate. Verifique a configuração e o ficheiro de origem.")