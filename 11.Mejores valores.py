import polars as pl
import plotly.express as px

# 1. Crear el DataFrame de Polars con variables en español
# Los datos reflejan tus modelos y subconjuntos (1% y 5%)
datos_tesis = {
'Metrica':['Diversidad','Diversidad','Diversidad','Coherencia Umass','Coherencia Umass','Coherencia Umass','Diversidad','Diversidad','Diversidad','Coherencia Umass','Coherencia Umass','Coherencia Umass'],
'Modelo':['LDA','NMF','BERTopic','LDA','NMF','BERTopic','LDA','NMF','BERTopic','LDA','NMF','BERTopic'],
'Valor'	:[0.93,1,0.47,-0.57,-0.17,-0.13,0.93,0.8,0.5,-2.24,-5.32,-9.14],
'Datos' :['Original','Original','Original','Original','Original','Original','LLM','LLM','LLM','LLM','LLM','LLM']	,
'Numero_de_topicos':[3,3,5,8,3,6,3,8,4,3,4,2]	

}



df = pl.DataFrame(datos_tesis)

# 2. Crear la columna combinada para la leyenda: "Modelo (Datos)"
df = df.with_columns(
    pl.format("{} ({})", pl.col("Modelo"), pl.col("Datos")).alias("Modelo_y_Datos")
)

# 3. Función para generar los gráficos dispersión (Scatter)
def graficar_metrica(df_input, nombre_metrica, titulo):
    df_filtrado = df_input.filter(pl.col("Metrica") == nombre_metrica)
    
    fig = px.scatter(
        df_filtrado,
        x="Numero_de_topicos",
        y="Valor",
        color="Datos",  # Aquí se combinan Modelo y Datos en la leyenda
        symbol="Modelo",         # Opcional: diferentes formas para cada modelo
        title=titulo,
        labels={
            "Numero_de_topicos": "Número de tópicos",
            "Valor": f"Valor de {nombre_metrica}",
            "Modelo_y_Datos": "Modelo y Proporción"
        },
        template="plotly_white"
    )

    fig.update_traces(marker=dict(size=12))
    fig.update_layout(
        title_x=0.5,
        legend_title_text="Datos, Modelo",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# 4. Generar y mostrar los gráficos
fig_div = graficar_metrica(df, "Diversidad", 'Mejores valores de Diversidad')
fig_coh = graficar_metrica(df, "Coherencia Umass", 'Mejores valores de Coherencia')

fig_div.show()
fig_coh.show()