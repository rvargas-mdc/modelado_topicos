import polars as pl
from modulos import *
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import kaleido
import seaborn as sns

RUTA_BASE = str(Path(__file__).resolve().parent)




def plot_pareto_frontier(df, nombre_modelo, titulo_grafico, subtitulo_grafico, prom_coherencia, prom_diversidad, nombre_imagen, titulo_leyenda, posicion_coherencia, posicion_diversidad, sufijo_modelo=None):
    # Convert to pandas for Plotly compatibility if necessary
    print('.'*20, subtitulo_grafico)
    if 'LLM' in titulo_grafico:
        num_subconjuntos = 1
        df = (df.with_columns(pl.col('subconjunto')
                               .str.replace('gpt','GPT')
                               .str.replace('gemini','Gemini')
                               .str.replace('deepseek','DeepSeek')
                               .str.replace('claude','Claude')
                               )
             )
        if ('BERT' in nombre_modelo) or ('mejores' in subtitulo_grafico):
            df = df.with_columns(pl.format('{} {}%',
                                        pl.col('subconjunto'),
                                        (pl.col('hiperparametros')*100).cast(int)
                                         )
                                    .alias('subconjunto')
                                )
    elif 'mejores' in subtitulo_grafico:
        num_subconjuntos = 1
    else:
        num_subconjuntos = 5

    print('.'*20, subtitulo_grafico, num_subconjuntos)
    dimensiones = {
        1:{'alto': 500, 'ancho': 500, 'columnas': 1, 'margen_leyenda':-0.2},
        5:{'alto': 1000, 'ancho': 900, 'columnas': 3, 'margen_leyenda':-0.2}
    }

    
    pdf = df.to_pandas()
    max_coherencia = float(pdf['coherencia_umass'].max())
    min_coherencia = float(pdf['coherencia_umass'].min())
    min_diversidad = float(pdf['diversidad'].min())
    rango_coherencia = max_coherencia - min_coherencia
    lim_sup = max_coherencia - rango_coherencia * .2
    lim_inf = min_coherencia + rango_coherencia * .2
    print()
    if (prom_coherencia > lim_sup):
        posicion_diversidad = 'bottom right'
    elif (prom_coherencia < lim_inf):
        posicion_diversidad = 'top right'

    dimensiones_fig = dimensiones[num_subconjuntos]
    if ('LLM' in titulo_grafico):
        fig = px.scatter(
            pdf, 
            x="diversidad", 
            y="coherencia_umass",
            color="numero_topicos",
            symbol="subconjunto",
            hover_name="model_id",
            size_max=15,
            labels={
                "diversidad": "Diversidad",
                "coherencia_umass": "Coherencia UMass",
                "numero_topicos": "# Tópicos"
            },
            template="plotly_white",
            color_continuous_scale="Viridis"
        ) 
    elif ('mejores' in subtitulo_grafico):
        pdf["Subconjunto-Columna"] = pdf['subconjunto'] + '-' + pdf['columna']
        fig = px.scatter(
            pdf, 
            x="diversidad", 
            y="coherencia_umass",
            color="numero_topicos",
            symbol="Subconjunto-Columna",
            hover_name="model_id",
            size_max=15,
            labels={
                "diversidad": "Diversidad",
                "coherencia_umass": "Coherencia UMass",
                "numero_topicos": "# Tópicos"
            },
            template="plotly_white",
            color_continuous_scale="Viridis"
        ) 
    else:
        fig = px.scatter(
            pdf, 
            x="diversidad", 
            y="coherencia_umass",
            color="numero_topicos",
            symbol="subconjunto",
            hover_name="model_id",
            facet_col='columna',
            facet_col_wrap=dimensiones_fig['columnas'],
            size_max=15,
            labels={
                "diversidad": "Diversidad",
                "coherencia_umass": "Coherencia UMass",
                "numero_topicos": "# Tópicos"
            },
            template="plotly_white",
            color_continuous_scale="Viridis"
        )


    fig.update_layout(
            height=dimensiones_fig['alto'],        
            width=dimensiones_fig['ancho'],       
            title=dict(
                text=f'<b>{titulo_grafico}</b>',
                subtitle=dict(text=f'<b>{subtitulo_grafico}</b>', 
                              font=dict(size=14))
            ),
            title_x=0.5,
            margin=dict(l=50, r=150, t=120, b=150), 
            font=dict(size=14),

            showlegend=True,
            legend=dict(
                orientation="h",     
                yanchor="top",       
                y=dimensiones_fig['margen_leyenda'],             
                xanchor="center",    
                x=0.5,
                bordercolor="Black",
                borderwidth=1,
                title_text="<b>Model Parameters</b>"
            ),

        )
    if 'BERT' in nombre_modelo:
        fig.update_layout(coloraxis_showscale=True)
    # Add quadrant lines based on averages
    fig.add_hline(
        y=prom_coherencia, 
        line_dash="dot", 
        annotation_text="CPG",
        annotation_position=posicion_coherencia  # This moves it UNDER the line
    )

    fig.add_vline(
        x=prom_diversidad, 
        line_dash="dot", 
        annotation_text="DPG",
        annotation_position=posicion_diversidad    # Standard position for vertical lines
    )


    fig.update_layout(
        title_x=0.5,
        legend_title=titulo_leyenda,
        font=dict(family="Arial", size=12)
    )

    fig.add_annotation(
        text="CPG y DPG: Promedio global Coherencia y Diversidad",
        xref="paper",
        yref="paper",
        x=0.30,
        y=-0.09,
        xanchor="center",
        yanchor="top",
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    fig.show()
    if sufijo_modelo:
        sufijo_modelo = sufijo_modelo +'_'
    else:
        sufijo_modelo = ''

    if ('LLM' in titulo_grafico):
        carpeta = 'llm'
    else:
        carpeta = 'no llm'
        
    fig.write_image(f'{RUTA_BASE}/graficos/{carpeta}/{nombre_imagen}.png')

def plot_parameter_flow(df):
    # Mapping categorical columns to integers for the parallel plot
    pdf = df.to_pandas()
    for col in ["subconjunto", "columna", "numero_topicos"]:
        pdf[f"{col}_idx"] = pdf[col].astype("category").cat.codes

    fig = px.parallel_coordinates(
        pdf,
        dimensions=[
            "subconjunto_idx", "columna_idx",  
            "diversidad", "coherencia_umass"
        ],
        color="coherencia_umass",
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="<b>Hyperparameter Influence on Model Metrics</b>"
    )
    fig.show()

def obtener_del_dominio(df_modelo, df_metricas, n_top, columna):
    df_del_dominio = (df_modelo
                        .filter(pl.col('palabra').is_in(NO_STOPWORDS))
                        .group_by('subconjunto','columna','numero_topicos','hiperparametros')
                        .agg(pl.col('palabra').n_unique().alias('del_dominio'))
                        )
    df_palabras = (df_modelo
                    .group_by('subconjunto','columna','numero_topicos','hiperparametros')
                    .agg(pl.col('palabra').n_unique().alias('palabras_unicas'))
                  )
    df_mejores = (df_del_dominio
                     .join(df_palabras,
                            on = ['subconjunto','columna','numero_topicos','hiperparametros']  
                          )
                      .with_columns(
                            (pl.col('del_dominio') / 
                             pl.col('palabras_unicas'))
                            .alias('porc_del_dominio')
                                   )
                   )
    if columna not in ['coherencia_umass','diversidad']:
        df_mejores = (df_mejores
                      .sort(columna,descending=True)
                      .head(n_top))

    df_nueva_metrica = (df_mejores
                      .join(df_metricas,
                            on=['subconjunto','columna','numero_topicos','hiperparametros']
                            )
                      .select(df_metricas.columns +
                              [c for c in df_mejores.columns 
                                 if c not in df_metricas.columns])
                      .sort(columna,descending=True)
                    )
    if columna in ['coherencia_umass','diversidad']:
        df_nueva_metrica = (df_nueva_metrica
                      .sort(columna,descending=True)
                      .head(n_top))    
    return df_nueva_metrica 


def generate_wordcloud(df, column_name, nombre_imagen):
    # 1. Join all words into a single string
    # In your case, 'palabra' is likely the column
    text = " ".join(df[column_name].to_list())

    # 2. Generate the cloud
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color="white",
        colormap="viridis",
        max_words=100
    ).generate(text)

    # 3. Display using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"{RUTA_BASE}/graficos/{nombre_imagen}.png", dpi=300, bbox_inches='tight')
    plt.show()

def crea_df_para_nube(df_modelo, df_metricas, id_modelo, subconjunto, columna, numero_topcios):
    df_nube = df_modelo.filter(
          pl.col("model_id").is_in([id_modelo]) &
          pl.col("subconjunto").is_in([subconjunto]) &
          pl.col("columna").is_in([columna]) &
          pl.col("numero_topicos").is_in([numero_topcios]))
    df_nube_met = (df_nube.join(df_metricas, 
                               on='model_id')
                          .select('coherencia_umass',
                                  'diversidad')
                          .unique()
                  )
    diversidad = df_nube_met['diversidad'][0]
    coherencia = df_nube_met['coherencia_umass'][0]

    return df_nube, coherencia, diversidad

def imprime_palabras(df_palabras):
    for row in (df_palabras.group_by(["num_topico"])
    .agg(
        pl.col("palabra").str.join(", ").alias("all_words")
    )
    ).sort("num_topico").iter_rows():
        print(row[0], row[1])


#df, nombre_modelo, columnas_diversidad = (df_lda, 'LDA', columnas_diversidad)
def calcular_metricas(df, nombre_modelo, columnas_diversidad):
    columnas_diversidad1 = columnas_diversidad[1]
    columnas_diversidad2 = columnas_diversidad[2]
    columnas_diversidad3 = columnas_diversidad[3]
    columnas_diversidad4 = columnas_diversidad[4]
    if nombre_modelo == 'LDA':
        df_modelo = (df 
                .with_columns(pl.col('tipo_df')
                                .str.split('-')
                                .list.get(0).alias('subconjunto'),
                            pl.col('columna')
                            .str.replace('_anonimizado_limpio','')
                            .str.replace('_anonimizada_limpio','')
                            .str.replace('__','_'),
                            pl.lit('-').alias('hiperparametros')
                                )
                .with_columns(pl.format("{}_{}_K{}_{}", 
                                        pl.col("subconjunto"), 
                                        pl.col("columna"), 
                                        pl.col("numero_topicos"), 
                                        pl.col("hiperparametros")
                                ).alias("model_id")
                             )
            )

        if 'pals_en_topicos' in df_modelo.columns:
            df_modelo = df_modelo.with_columns(pl.format('{} {}',
                                                         pl.col('subconjunto'),
                                                         pl.col('pals_en_topicos'))
                                                  .alias('subconjunto')
                                              )
                
        df_modelo = df_modelo.select(columnas_diversidad1)
        dict_hiperparametros = {}
    elif nombre_modelo =='NMF':
        df_modelo = (df 
                .with_columns(pl.col('tipo_df')
                                .str.split('-')
                                .list.get(0).alias('subconjunto'),
                            pl.col('hiperparametros')
                                .str.replace_all(r"'n_c': \d{1,2}, ", "")
                                .alias('txt_hiperparametros'),
                            pl.col('hiperparametros')
                                .str.extract(r"(\d), 'n_c")
                                .alias('hiperparametros'),
                            pl.col('columna')
                            .str.replace('_anonimizado_limpio','')
                            .str.replace('_anonimizada_limpio','')
                            .str.replace('__','_')
                                )
                .with_columns(pl.format("{}_{}_K{}_{}", 
                                        pl.col("subconjunto"), 
                                        pl.col("columna"), 
                                        pl.col("numero_topicos"), 
                                        pl.col("hiperparametros")
                                ).alias("model_id")
                             )
            )
        dict_hiperparametros = {a:b for (a,b) 
                in df_modelo.select('hiperparametros','txt_hiperparametros')
                            .unique()
                            .iter_rows()}
        if 'pals_en_topicos' in df_modelo.columns:
            df_modelo = df_modelo.with_columns(pl.format('{} {}',
                                                         pl.col('subconjunto'),
                                                         pl.col('pals_en_topicos'))
                                                  .alias('subconjunto')
                                              )
        df_modelo = df_modelo.select(columnas_diversidad1)
        dict_hiperparametros = {}
    elif nombre_modelo =='BERT':
        df_modelo_comprimido = (df.with_columns(pl.col('tipo_df')
                                .str.split('-')
                                .list.get(0).alias('subconjunto'),
                            pl.col('columna')
                            .str.replace('_anonimizado','')
                            .str.replace('_anonimizada','')
                            .str.replace('__','_')
                                )
                .with_columns(pl.format("{}_{}_{}", 
                                        pl.col("subconjunto"), 
                                        pl.col("columna"),
                                        pl.col("porc_min_tamano_topico")
                                ).alias("model_id"))
                .with_columns(pl.col('Topic').n_unique().over('model_id')
                                .alias('numero_topicos'))
                .with_columns(pl.format("{}_{}_K{}_{}", 
                                        pl.col("subconjunto"), 
                                        pl.col("columna"), 
                                        pl.col("numero_topicos"),
                                        pl.col("porc_min_tamano_topico")
                                ).alias("model_id")
                             ))
        df_modelo = (df_modelo_comprimido.explode("Representation")
                        .with_columns(pl.format('porc_docs_min:{}%',
                                                pl.col('porc_min_tamano_topico')*100
                                               )
                                        .alias('txt_hiperparametros')
                                    )
                        .rename({"Representation": "palabra",
                                 'porc_min_tamano_topico': 'hiperparametros',
                                 'Topic':'num_topico'})
                    )
        dict_hiperparametros = {a:b for (a,b) 
                in df_modelo.select('hiperparametros','txt_hiperparametros')
                            .unique()
                            .iter_rows()}
        df_modelo = df_modelo.select(columnas_diversidad1)

    df_palabras_por_modelo = (
        df_modelo.group_by(columnas_diversidad2)
        .agg(pl.count("num_topico").alias("ocurrencia_en_modelo"))
    )

    df_diversidad = (
        df_palabras_por_modelo.group_by(columnas_diversidad3)
        .agg([
            pl.col("palabra")
            .filter(pl.col("ocurrencia_en_modelo") == 1)
            .count()
            .alias("palabras_unicas"),
            
            pl.col("ocurrencia_en_modelo")
            .sum()
            .alias("palabras_en_modelo")
        ])
        .with_columns(
            (pl.col("palabras_unicas") / pl.col("palabras_en_modelo")).alias("diversidad")
        )
        .sort("diversidad", descending=True)
    )

    df_metricas = (df_modelo.join(df_diversidad, on=columnas_diversidad3)
    .select(columnas_diversidad4)
    .unique()
    .sort('diversidad',descending=True))

    prom_coherencia = df_metricas['coherencia_umass'].mean()
    prom_diversidad = df_metricas['diversidad'].mean()

    return {'df_modelo':df_modelo, 
            'df_palabras_por_modelo':df_palabras_por_modelo, 
            'df_metricas':df_metricas, 
            'prom_coherencia':prom_coherencia, 
            'prom_diversidad':prom_diversidad,
            'dict_hiperparametros':dict_hiperparametros}

def plot_box_plots(df_, titulo,columna_x, columna_y, titulo_x, titulo_y):
    df_sorted = df_.sort(pl.col(columna_x).cast(pl.Int32))

    # 2. Create the box plot
    fig = px.box(
        df_sorted.to_pandas(), 
        x="hiperparametros", 
        y=columna_y,
        color="hiperparametros", # Gives each box a unique color
        points="outliers",       # Options: 'all' (show every model), 'outliers', or False
        title=f'<b>{titulo}</b>',
        labels={
            columna_x: titulo_x,
            columna_y: titulo_y
        },
        template="plotly_white",
        category_orders={"hiperparametros": [str(i) for i in range(1, 9)]} # Force order 1-8
    )

    # 3. High-impact styling
    fig.update_layout(
        height=600,
        width=1000,
        showlegend=False,  # Legend is redundant since labels are on X-axis
        title_x=0.5,       # Center title
        font=dict(size=14),
        xaxis=dict(title=columna_x),
        yaxis=dict(title=columna_y)
    )

    fig.show()    