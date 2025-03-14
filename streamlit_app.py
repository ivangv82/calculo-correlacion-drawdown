import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --------------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------------------------------------

def get_data(tickers, start_date, end_date):
    """
    Descarga de datos desde yfinance con precios ajustados (auto_adjust=True).
    Se devuelve un DataFrame con las columnas en mayúsculas.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    # Si el DataFrame tiene columnas MultiIndex, extrae la parte de "Close".
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs('Close', axis=1, level=0)
    else:
        # Si existe la columna 'Close' la seleccionamos (caso de un único ticker).
        if 'Close' in data.columns:
            data = data[['Close']]
            # Renombrar la columna si es un único ticker
            if len(tickers) == 1:
                data.columns = tickers

    # Convertimos todas las columnas a mayúsculas para evitar problemas de case
    data.columns = [col.upper() for col in data.columns]
    return data

def compute_log_returns(data):
    """Calcula los retornos logarítmicos y elimina el primer valor NA."""
    returns = np.log(data / data.shift(1)).dropna()
    return returns

def compute_drawdowns(data):
    """Calcula el drawdown: (Precio actual / Máximo histórico hasta el momento) - 1."""
    dd = data.apply(lambda x: x / x.cummax() - 1)
    return dd

def compute_stats(returns, dd):
    """
    Crea una tabla con:
      - Mean Return: media de los retornos
      - Std Return: volatilidad de los retornos
      - Max Drawdown: valor mínimo (peor) en la serie de drawdowns
    """
    stats = pd.DataFrame({
        'Mean Return': returns.mean(),
        'Std Return': returns.std(),
        'Max Drawdown': dd.min()
    })
    return stats

def plot_corr_heatmap(corr_matrix, title):
    """
    Genera y retorna una figura de un mapa de calor (heatmap)
    con la mitad superior enmascarada para facilitar la visualización.
    """
    fig, ax = plt.subplots(figsize=(8,6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def segment_analysis(returns, dd, segments):
    """
    Calcula la correlación de retornos y drawdowns por tramos,
    y genera las figuras correspondientes.
    
    Devuelve una lista de figuras:
      - 2 figuras por cada tramo (correlación de retornos y de drawdowns)
      - 1 figura final con el peor drawdown por tramo.
    """
    figs = []
    n = len(returns)
    seg_size = n // segments
    
    # Diccionarios para almacenar resultados
    returns_corr_segments = {}
    drawdowns_corr_segments = {}
    max_dd_segments = {}

    for i in range(segments):
        start_idx = i * seg_size
        end_idx = n if i == segments - 1 else (i+1) * seg_size
        
        # Subconjunto de datos para el tramo
        seg_returns = returns.iloc[start_idx:end_idx]
        seg_dd = dd.iloc[start_idx:end_idx]
        
        # Etiqueta de tramo y rango de fechas
        seg_label = f"Segmento {i+1}"
        seg_start_date = seg_returns.index[0].strftime('%Y-%m-%d')
        seg_end_date = seg_returns.index[-1].strftime('%Y-%m-%d')
        
        # Correlaciones de retornos y drawdowns
        returns_corr_segments[seg_label] = seg_returns.corr(method='spearman')
        drawdowns_corr_segments[seg_label] = seg_dd.corr(method='spearman')
        
        # Máximo drawdown (mínimo) para cada activo en este tramo
        max_dd_segments[seg_label] = seg_dd.min()
        
        # Figura: correlación de retornos
        fig_ret = plot_corr_heatmap(
            returns_corr_segments[seg_label],
            f"Correlación de Retornos ({seg_label}) [{seg_start_date} - {seg_end_date}]"
        )
        figs.append(fig_ret)
        
        # Figura: correlación de drawdowns
        fig_dd = plot_corr_heatmap(
            drawdowns_corr_segments[seg_label],
            f"Correlación de Drawdowns ({seg_label}) [{seg_start_date} - {seg_end_date}]"
        )
        figs.append(fig_dd)
    
    # Heatmap con el peor drawdown por tramo
    max_dd_df = pd.DataFrame(max_dd_segments).T
    fig_max_dd, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(max_dd_df, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Peor (Mínimo) Drawdown por Segmento (NO es correlación)")
    plt.xlabel("Activo")
    plt.ylabel("Segmento")
    plt.tight_layout()
    figs.append(fig_max_dd)
    
    return figs

def compare_assets(data, dd, returns, ticker1, ticker2):
    """
    Genera y retorna una lista de figuras comparando:
      - Evolución de precios
      - Evolución de drawdowns
      - Correlación rolling de retornos (30 días)
      - Correlación rolling de drawdowns (30 días)
    """
    figs = []
    t1 = ticker1.upper()
    t2 = ticker2.upper()
    
    if t1 not in data.columns or t2 not in data.columns:
        # No generamos figuras si no existen las columnas
        return figs
    
    # Figura 1: evolución de la cotización y el drawdown de los dos activos
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex=True)
    ax1.plot(data.index, data[t1], label=t1)
    ax1.plot(data.index, data[t2], label=t2)
    ax1.set_title("Cotización de los Activos")
    ax1.legend()
    
    ax2.plot(dd.index, dd[t1], label=t1)
    ax2.plot(dd.index, dd[t2], label=t2)
    ax2.set_title("Evolución del Drawdown")
    ax2.legend()
    plt.tight_layout()
    figs.append(fig1)

    # Figura 2: correlación rolling de retornos (30 días)
    window = 30
    rolling_corr_returns = returns[t1].rolling(window).corr(returns[t2])
    fig2, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(rolling_corr_returns.index, rolling_corr_returns, label="Correlación Rolling Retornos")
    ax3.set_title("Correlación Rolling de Retornos (30 días)")
    ax3.legend()
    plt.tight_layout()
    figs.append(fig2)

    # Figura 3: correlación rolling de drawdowns (30 días)
    rolling_corr_dd = dd[t1].rolling(window).corr(dd[t2])
    fig3, ax4 = plt.subplots(figsize=(10,4))
    ax4.plot(rolling_corr_dd.index, rolling_corr_dd, label="Correlación Rolling Drawdowns", color='orange')
    ax4.set_title("Correlación Rolling de Drawdowns (30 días)")
    ax4.legend()
    plt.tight_layout()
    figs.append(fig3)

    return figs

# --------------------------------------------------------------------------------
# APLICACIÓN STREAMLIT
# --------------------------------------------------------------------------------

def main():
    st.title("Análisis de Correlación de Activos con Streamlit")

    # --- Entrada de parámetros ---
    st.subheader("Parámetros de entrada")
    ticker_input = st.text_input("Introduce los tickers separados por comas", "GLD, TLT, SPY, QQQ")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha de inicio", datetime(2006,1,1))
    with col2:
        end_date = st.date_input("Fecha final", datetime.today())

    # Botón para ejecutar la descarga y el análisis
    if st.button("Descargar y Analizar"):
        tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # --- Descarga de datos ---
        st.write("### Descargando datos...")
        data = get_data(tickers, start_str, end_str)

        if data.empty:
            st.warning("No se obtuvieron datos. Verifica los tickers o las fechas.")
            return

        # --- Cálculo de retornos y drawdowns ---
        returns = compute_log_returns(data)
        dd = compute_drawdowns(data)

        # --- Estadísticas ---
        stats = compute_stats(returns, dd)
        st.write("### Estadísticas de los activos")
        st.dataframe(stats)

        # --- Correlaciones globales ---
        corr_returns = returns.corr(method='spearman')
        corr_dd = dd.corr(method='spearman')

        # Mapa de calor de la correlación de retornos
        fig_ret = plot_corr_heatmap(
            corr_returns,
            f"Matriz de Correlación de Retornos (Spearman) [{start_str} - {end_str}]"
        )
        st.pyplot(fig_ret)

        # Mapa de calor de la correlación de drawdowns
        fig_dd = plot_corr_heatmap(
            corr_dd,
            f"Matriz de Correlación de Drawdowns (Spearman) [{start_str} - {end_str}]"
        )
        st.pyplot(fig_dd)

        # --- Análisis por tramos ---
        seg_option = st.checkbox("¿Desea calcular la correlación en varios tramos?")
        if seg_option:
            segments = st.number_input(
                "¿En cuántos tramos desea dividir los cálculos?",
                min_value=2, max_value=20, value=3
            )
            figs_segment = segment_analysis(returns, dd, segments)
            for fig in figs_segment:
                st.pyplot(fig)

        # --- Comparación de dos activos ---
        compare_option = st.checkbox("¿Desea comparar 2 activos en concreto?")
        if compare_option:
            st.write("Activos disponibles:", list(data.columns))
            col3, col4 = st.columns(2)
            with col3:
                ticker1 = st.selectbox("Selecciona el primer ticker a comparar", data.columns)
            with col4:
                ticker2 = st.selectbox("Selecciona el segundo ticker a comparar", data.columns)

            if ticker1 and ticker2 and ticker1 != ticker2:
                figs_compare = compare_assets(data, dd, returns, ticker1, ticker2)
                if figs_compare:
                    for fig in figs_compare:
                        st.pyplot(fig)
                else:
                    st.warning("No se pudo comparar. Revisa que ambos activos existan en los datos.")
            else:
                st.warning("Debes elegir dos activos distintos para comparar.")


# Llamamos a la función principal de Streamlit
if __name__ == "__main__":
    main()
