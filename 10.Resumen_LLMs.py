import polars as pl
from modulos import *
from pathlib import Path
from modulos_resumen import *

RUTA_BASE = str(Path(__file__).resolve().parent)



try:
    df_lda = pl.read_parquet( f'{RUTA_BASE}/ejecucion/lda_llm.parquet')
    df_nmf = pl.read_parquet( f'{RUTA_BASE}/ejecucion/nmf_llm.parquet')
except:
    df_lda1 =(pl.read_parquet( f'{RUTA_BASE}/ejecucion/lda_llm_4.parquet')
                .with_columns(pals_en_topicos=pl.lit(4),
                             columna=pl.lit('denuncia')))
    df_lda2 =(pl.read_parquet( f'{RUTA_BASE}/ejecucion/lda_llm_10.parquet')
                .with_columns(pals_en_topicos=pl.lit(10),
                            columna=pl.lit('denuncia')))
    df_lda = pl.concat([df_lda1,df_lda2])
    df_lda.write_parquet( f'{RUTA_BASE}/ejecucion/lda_llm.parquet')

    df_nmf1 =(pl.read_parquet( f'{RUTA_BASE}/ejecucion/nmf_llm_4.parquet')
               .with_columns(pals_en_topicos=pl.lit(4),
                              columna=pl.lit('denuncia')))
    df_nmf2 =(pl.read_parquet( f'{RUTA_BASE}/ejecucion/nmf_llm_10.parquet')
                .with_columns(pals_en_topicos=pl.lit(10),
                              columna=pl.lit('denuncia')))
    df_nmf = pl.concat([df_nmf1,df_nmf2])
    df_nmf.write_parquet( f'{RUTA_BASE}/ejecucion/nmf_llm.parquet')

df_bert = (pl.read_parquet( f'{RUTA_BASE}/ejecucion/crudo_bert_llm.parquet')
                 .with_columns(pals_en_topicos=pl.lit(0),
                              columna=pl.lit('denuncia')))


# valores BERT
import plotly.express as px

def graficar_bert(df):
    df = (df.filter(pl.col('Topic')==-1)
                .select('Count','tipo_df','porc_min_tamano_topico')
                .unique()
                .sort('Count'))
    df = (df.with_columns(pl.col('tipo_df')
                            .str.replace('gpt-5-mini','GPT')
                            .str.replace('gemini-2.5-flash-lite','Gemini')
                            .str.replace('deepseek-v3.2-speciale','DeepSeek')
                            .str.replace('claude-3-haiku','Claude')
                            )
            .with_columns(pl.format('{} ({}%)',
                                    pl.col('tipo_df'),
                                    (pl.col('porc_min_tamano_topico')*100).cast(int)
            ))
         )
    fig = px.bar(
        df, 
        x="Count", 
        y="tipo_df", 
        orientation='h',
        text="Count",
        labels={"Count": "Denuncias", "tipo_df": "Modelo"},
        template="plotly_white",
        title="Denuncias sin agrupar por Modelo"
    )

    # 4. Applying the unique Sea Blue color and fixing labels
    fig.update_traces(
        marker_color='#006994',  # Sea Blue Hex code
        textposition='outside', 
        cliponaxis=False
    )
    fig.update_layout(
        margin=dict(r=100), 
    )

    fig.update_traces(cliponaxis=False)
    fig.update_layout(
        xaxis_title="Denuncias",
        yaxis_title="Modelo",
        showlegend=False,
        # Adding extra margin on the right so the text isn't cut off
        margin=dict(r=50) 
    )

    fig.show()

graficar_bert(df_bert)

columnas_diversidad =  {
    1 : ['subconjunto','columna','numero_topicos','num_topico',
         'hiperparametros','model_id','palabra','coherencia_umass'],
    2 : ['subconjunto','columna','numero_topicos','hiperparametros',
         'model_id','palabra' ],
    3 : ['subconjunto','columna','numero_topicos','hiperparametros',
         'model_id' ],
    4 : ['subconjunto','columna','numero_topicos','hiperparametros',
         'model_id' ,'diversidad','coherencia_umass']
 }

def calcular_y_graficar(df_modelo, nombre_modelo, sufijo_modelo):
    metricas = calcular_metricas(df_modelo, nombre_modelo, columnas_diversidad)
    df_modelo = metricas['df_modelo'] 
    df_metricas = metricas['df_metricas']
    dict_hiperparametros = metricas['dict_hiperparametros']

    prom_coherencia = metricas['prom_coherencia']            
    prom_diversidad = metricas['prom_diversidad']

    if nombre_modelo == 'BERT':
        titulo_leyenda = 'Cantidad de tópicos <br>y Modelo'
    else:
        titulo_leyenda = 'Modelo y Cantidad de <br>palabras en tópico'
    plot_pareto_frontier(df_metricas, 
                        nombre_modelo, 
                        f'Modelos {nombre_modelo} ({sufijo_modelo}): Coherencia (UMass) vs Diversidad',
                        '',
                        prom_coherencia, 
                        prom_diversidad, 
                        f'{nombre_modelo}({sufijo_modelo})_sp_todos', 
                        titulo_leyenda,
                        posicion_coherencia='bottom right',
                        posicion_diversidad='bottom right')

    dict_criterios = {
        1:'porc_del_dominio',
        2:'del_dominio',
        3:'diversidad',
        4:'coherencia_umass'
    }
    posicion_anotaciones ={
        1:['bottom right','bottom right'],
        2:['bottom right','bottom right'],
        3:['bottom right','bottom right'],
        4:['bottom right','top right']
    }


    if nombre_modelo=='BERT':
        n_top = 3
    else:
        n_top = 10
    subtitulos = {
        1:f'{n_top} mejores resultados según porcentaje de palabras en dominio',
        2:f'{n_top} mejores resultados según coincidencias en dominio',
        3:f'{n_top} mejores resultados según diversidad',
        4:f'{n_top} mejores resultados según Coherencia Umass'
    }

    dict_mejores = {}
    dict_mejores[1] = obtener_del_dominio(df_modelo, df_metricas, n_top, dict_criterios[1])
    dict_mejores[1].write_excel(f'{RUTA_BASE}/graficos/{nombre_modelo}{sufijo_modelo}_mejores_porc.xlsx')
    dict_mejores[2] = obtener_del_dominio(df_modelo, df_metricas, n_top, dict_criterios[2])
    dict_mejores[2].write_excel(f'{RUTA_BASE}/graficos/{nombre_modelo}{sufijo_modelo}_mejores_dominio.xlsx')
    dict_mejores[3] = obtener_del_dominio(df_modelo, df_metricas, n_top, dict_criterios[3])
    dict_mejores[3].write_excel(f'{RUTA_BASE}/graficos/{nombre_modelo}{sufijo_modelo}_mejores_diversidad.xlsx')
    dict_mejores[4] = obtener_del_dominio(df_modelo, df_metricas, n_top, dict_criterios[4])
    dict_mejores[4].write_excel(f'{RUTA_BASE}/graficos/{nombre_modelo}{sufijo_modelo}_mejores_umass.xlsx')

    print('='*50)
    print('='*50)

    for i in dict_criterios.keys():

        print(dict_criterios[i].upper())
        print('.'*50)
        df_ = dict_mejores[i].head(1)
        valor = {c:df_[c][0] for c in dict_mejores[i].columns}
        for v in valor.items():
            print(v)
        plot_pareto_frontier(dict_mejores[i], 
                            nombre_modelo, 
                            f'{nombre_modelo} (LLM): Coherencia UMass vs Diversidad',
                            subtitulos[i],
                            prom_coherencia, 
                            prom_diversidad, 
                            f'{nombre_modelo}_(LLM)_mejores_{dict_criterios[i]}',
                            titulo_leyenda,
                            posicion_coherencia=posicion_anotaciones[i][0], 
                            posicion_diversidad=posicion_anotaciones[i][1])

        df_nube, coherencia, diversidad = crea_df_para_nube(df_modelo, 
                                                            df_metricas, 
                                                            valor['model_id'],
                                                            valor['subconjunto'],
                                                            valor['columna'],
                                                            valor['numero_topicos'],
                                                            )
        generate_wordcloud(df_nube, "palabra", f'{nombre_modelo}_(LLM)_wc_3_denuncia_todos')
        print('Coherencia',coherencia)
        print('Diversidad', diversidad)

        imprime_palabras(df_nube)
        print('='*50)
        print('='*50)
        print(' ')
        print('='*50)
        print('='*50)


    return df_modelo, df_metricas, dict_hiperparametros

# LDA ----------------------------------------------
nombre_modelo = 'LDA'
#df_modelo, df_metricas, _ = calcular_y_graficar(df_lda, nombre_modelo, 'LLM')

# NMF ----------------------------------------------
nombre_modelo = 'NMF'
#df_modelo, df_metricas, dict_hiperparametros = calcular_y_graficar(df_nmf, nombre_modelo, 'LLM')
#print(dict_hiperparametros)

# BERT ----------------------------------------------
nombre_modelo = 'BERT'
df_modelo, df_metricas, dict_hiperparametros = calcular_y_graficar(df_bert, nombre_modelo, 'LLM')
print(dict_hiperparametros)
