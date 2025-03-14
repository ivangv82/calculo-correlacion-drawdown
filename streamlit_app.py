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
    """Descarga de datos desde yfinance con precios ajustados (auto_adjust=True)."""
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

    # Convertimos todas las columnas a mayúsculas para evitar problemas de mayúsc/minúsc
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
      - Annual Return (%): anualización del retorno medio diario (log returns)
      - Annual Std (%): anualización de la volatilidad diaria
      - Max Drawdown (%): peor drawdown histórico
    Todo en porcentaje y redondeado a 2 decimales.
    """
    daily_mean = returns.mean()
    daily_std = returns.std()
    
    # Asumiendo ~252 días de mercado al año
    annual_return = (np.exp(daily_mean * 252) - 1) * 100
    annual_std = daily_std * np.sqrt(252) * 100
    max_dd = dd.min() * 100  # Convertimos a %

    stats = pd.DataFrame({
        'Annual Return (%)': annual_return,
        'Annual Std (%)': annual_std,
        'Max Drawdown (%)': max_dd
    })
    return stats.round(2)

def plot_corr_heatmap(corr_matrix, title):
    """Devuelve un mapa de calor (heatmap) de correlaciones con la mitad superior enmascarada."""
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
    
    returns_corr_segments = {}
    drawdowns_corr_segments = {}
    max_dd_segments = {}

    for i in range(segments):
        start_idx = i * seg_size
        end_idx = n if i == segments - 1 else (i+1) * seg_size
        
        seg_returns = returns.iloc[start_idx:end_idx]
        seg_dd = dd.iloc[start_idx:end_idx]
        
        seg_label = f"Segmento {i+1}"
        seg_start_date = seg_returns.index[0].strftime('%Y-%m-%d')
        seg_end_date = seg_returns.index[-1].strftime('%Y-%m-%d')
        
        returns_corr_segments[seg_label] = seg_returns.corr(method='spearman')
        drawdowns_corr_segments[seg_label] = seg_dd.corr(method='spearman')
        
        max_dd_segments[seg_label] = seg_dd.min()
        
        # Correlación de Retornos (por tramo)
        fig_ret = plot_corr_heatmap(
            returns_corr_segments[seg_label],
            f"Correlación de Retornos ({seg_label}) [{seg_start_date} - {seg_end_date}]"
        )
        figs.append(fig_ret)
        
        # Correlación de Drawdowns (por tramo)
        fig_dd = plot_corr_heatmap(
            drawdowns_corr_segments[seg_label],
            f"Correlación de Drawdowns ({seg_label}) [{seg_start_date} - {seg_end_date}]"
        )
        figs.append(fig_dd)
    
    # Heatmap con el peor drawdown (mínimo) por tramo
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
      - (Figura 1) Precios en dos escalas (izq y der) + Drawdown en el subplot inferior
      - (Figura 2) Correlación rolling de retornos (30 días)
      - (Figura 3) Correlación rolling de drawdowns (30 días)
    """
    figs = []
    t1 = ticker1.upper()
    t2 = ticker2.upper()
    
    if t1 not in data.columns or t2 not in data.columns:
        return figs  # Vacío si no existen

    # 1) Gráfico de precios con dos escalas y debajo el drawdown
    fig = plt.figure(figsize=(10,8))

    # Subplot superior (precios) con doble eje Y
    ax1 = fig.add_subplot(2,1,1)
    ax2 = ax1.twinx()
    
    ax1.plot(data.index, data[t1], label=t1, color='blue')
    ax2.plot(data.index, data[t2], label=t2, color='red')
    
    ax1.set_title("Cotización de los Activos (2 Escalas)")
    ax1.set_ylabel(f"Precio {t1}")
    ax2.set_ylabel(f"Precio {t2}")
    
    # Unimos leyendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Subplot inferior (drawdown) con un solo eje
    ax3 = fig.add_subplot(2,1,2, sharex=ax1)
    ax3.plot(dd.index, dd[t1], label=t1, color='blue')
    ax3.plot(dd.index, dd[t2], label=t2, color='red')
    ax3.set_title("Evolución del Drawdown")
    ax3.legend()

    plt.tight_layout()
    figs.append(fig)

    # 2) Correlación rolling de retornos (30 días)
    window = 30
    rolling_corr_returns = returns[t1].rolling(window).corr(returns[t2])
    fig2, ax4 = plt.subplots(figsize=(10,4))
    ax4.plot(rolling_corr_returns.index, rolling_corr_returns, label="Correlación Rolling Retornos")
    ax4.set_title("Correlación Rolling de Retornos (30 días)")
    ax4.legend()
    plt.tight_layout()
    figs.append(fig2)

    # 3) Correlación rolling de drawdowns (30 días)
    rolling_corr_dd = dd[t1].rolling(window).corr(dd[t2])
    fig3, ax5 = plt.subplots(figsize=(10,4))
    ax5.plot(rolling_corr_dd.index, rolling_corr_dd, label="Correlación Rolling Drawdowns", color='orange')
    ax5.set_title("Correlación Rolling de Drawdowns (30 días)")
    ax5.legend()
    plt.tight_layout()
    figs.append(fig3)

    return figs

# --------------------------------------------------------------------------------
# APLICACIÓN STREAMLIT
# --------------------------------------------------------------------------------

def main():
    # --------------------------------------------------------------------------------
    # LOGO Y CABECERA
    # --------------------------------------------------------------------------------
    # Mostramos el logo con link a tu web. Ajusta el 'height' a tu gusto.
 st.markdown(
    """
    <p style="text-align:center">
        <a href="https://formacionenbolsa.com" target="_blank">
            <img src="https://raw.githubusercontent.com/ivangv82/calculo-correlacion-drawdown/main/logoFB.png" alt="FormacionenBolsa" style="height:80px;">
        </a>
    </p>
    """,
    unsafe_allow_html=True
)

    st.title("Análisis de Correlación de Activos 🚀")
    st.markdown("¡Bienvenid@ a la aplicación de análisis de activos de [formacionenbolsa.com](https://formacionenbolsa.com)!")

    # Inicializamos las variables de sesión, si no existen
    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False
    if "data" not in st.session_state:
        st.session_state.data = None

    # --- Entrada de parámetros ---
    st.subheader("Parámetros de entrada ⚙️")
    ticker_input = st.text_input("Introduce los tickers separados por comas", "GLD, TLT, SPY, QQQ")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha de inicio", datetime(2006,1,1))
    with col2:
        end_date = st.date_input("Fecha final", datetime.today())

    # Botón para ejecutar la descarga y el análisis
    if st.button("Descargar y Analizar"):
        st.session_state.run_analysis = True
        
        # Guardamos los tickers y fechas en session_state
        st.session_state.tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        st.session_state.start_str = start_date.strftime('%Y-%m-%d')
        st.session_state.end_str = end_date.strftime('%Y-%m-%d')

        # --- Descarga de datos ---
        data = get_data(st.session_state.tickers, st.session_state.start_str, st.session_state.end_str)
        st.session_state.data = data
        
        # Verificación de datos
        if data.empty:
            st.warning("No se obtuvieron datos. Verifica los tickers o las fechas.")
            return
        
        # Cálculo de retornos y drawdowns
        st.session_state.returns = compute_log_returns(data)
        st.session_state.dd = compute_drawdowns(data)
        
        # Estadísticas anuales en %
        st.session_state.stats = compute_stats(st.session_state.returns, st.session_state.dd)
        
        # Correlaciones globales
        st.session_state.corr_returns = st.session_state.returns.corr(method='spearman')
        st.session_state.corr_dd = st.session_state.dd.corr(method='spearman')

    # Solo mostramos los resultados si run_analysis es True y hay datos
    if st.session_state.run_analysis and st.session_state.data is not None and not st.session_state.data.empty:
        
        # --- Mostrar resultados de análisis global ---
        st.write("### Estadísticas de los activos (en %) 📊")
        st.dataframe(st.session_state.stats)

        # Correlación de retornos
        fig_ret = plot_corr_heatmap(
            st.session_state.corr_returns,
            f"Matriz de Correlación de Retornos (Spearman) [{st.session_state.start_str} - {st.session_state.end_str}]"
        )
        st.pyplot(fig_ret)

        # Correlación de drawdowns
        fig_dd = plot_corr_heatmap(
            st.session_state.corr_dd,
            f"Matriz de Correlación de Drawdowns (Spearman) [{st.session_state.start_str} - {st.session_state.end_str}]"
        )
        st.pyplot(fig_dd)

        # --- Análisis por tramos ---
        seg_option = st.checkbox("¿Desea calcular la correlación en varios tramos? 🔀")
        if seg_option:
            segments = st.number_input("¿En cuántos tramos desea dividir los cálculos?", min_value=2, max_value=20, value=3)
            figs_segment = segment_analysis(st.session_state.returns, st.session_state.dd, segments)
            for fig in figs_segment:
                st.pyplot(fig)

        # --- Comparación de dos activos ---
        compare_option = st.checkbox("¿Desea comparar 2 activos en concreto? 🤝")
        if compare_option:
            st.write("Activos disponibles:", list(st.session_state.data.columns))
            col3, col4 = st.columns(2)
            with col3:
                ticker1 = st.selectbox("Selecciona el primer ticker a comparar", st.session_state.data.columns)
            with col4:
                ticker2 = st.selectbox("Selecciona el segundo ticker a comparar", st.session_state.data.columns)

            if ticker1 and ticker2 and ticker1 != ticker2:
                figs_compare = compare_assets(
                    st.session_state.data, 
                    st.session_state.dd, 
                    st.session_state.returns, 
                    ticker1, 
                    ticker2
                )
                if figs_compare:
                    for fig in figs_compare:
                        st.pyplot(fig)
                else:
                    st.warning("No se pudo comparar. Revisa que ambos activos existan en los datos.")
            else:
                st.warning("Debes elegir dos activos distintos para comparar.")

    # --------------------------------------------------------------------------------
    # ENLACE FINAL
    # --------------------------------------------------------------------------------
    st.markdown(
        """
        ---
        ### [Visita formacionenbolsa.com para más recursos 🚀](https://formacionenbolsa.com)
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
