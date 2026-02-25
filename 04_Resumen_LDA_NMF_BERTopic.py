import polars as pl
from modulos import *
from pathlib import Path
from modulos_resumen import *

RUTA_BASE = str(Path(__file__).resolve().parent)


df_lda = pl.read_parquet( f'{RUTA_BASE}/ejecucion/lda.parquet')
df_nmf = pl.read_parquet( f'{RUTA_BASE}/ejecucion/nmf.parquet')
df_bert = pl.read_parquet( f'{RUTA_BASE}/ejecucion/crudo_bert.parquet')




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

def calcular_y_graficar(df_modelo, nombre_modelo):
    metricas = calcular_metricas(df_modelo, nombre_modelo, columnas_diversidad)
    df_modelo = metricas['df_modelo'] 
    df_metricas = metricas['df_metricas']
    dict_hiperparametros = metricas['dict_hiperparametros']

    prom_coherencia = metricas['prom_coherencia']            
    prom_diversidad = metricas['prom_diversidad']


        
    plot_pareto_frontier(df_metricas, 
                        nombre_modelo, 
                        f'Modelos {nombre_modelo}: Coherencia (UMass) vs Diversidad',
                        '',
                        prom_coherencia, 
                        prom_diversidad, 
                        f'{nombre_modelo}_sp_todos', 
                        'Columnas',
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
    subtitulos = {
        1:'20 mejores resultados según porcentaje de palabras en dominio',
        2:'20 mejores resultados según coincidencias en dominio',
        3:'20 mejores resultados según diversidad',
        4:'20 mejores resultados según Coherencia Umass'
    }
    dict_mejores = {}
    dict_mejores[1] = obtener_del_dominio(df_modelo, df_metricas, 20, dict_criterios[1])
    dict_mejores[1].write_excel(f'{RUTA_BASE}/graficos/{nombre_modelo}_mejores_porc.xlsx')
    dict_mejores[2] = obtener_del_dominio(df_modelo, df_metricas, 20, dict_criterios[2])
    dict_mejores[2].write_excel(f'{RUTA_BASE}/graficos/{nombre_modelo}_mejores_dominio.xlsx')
    dict_mejores[3] = obtener_del_dominio(df_modelo, df_metricas, 20, dict_criterios[3])
    dict_mejores[3].write_excel(f'{RUTA_BASE}/graficos/{nombre_modelo}_mejores_diversidad.xlsx')
    dict_mejores[4] = obtener_del_dominio(df_modelo, df_metricas, 20, dict_criterios[4])
    dict_mejores[4].write_excel(f'{RUTA_BASE}/graficos/{nombre_modelo}_mejores_umass.xlsx')

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
                            f'{nombre_modelo}: Coherencia UMass vs Diversidad',
                            subtitulos[i],
                            prom_coherencia, 
                            prom_diversidad, 
                            f'{nombre_modelo}_mejores_{dict_criterios[i]}',
                            'Columnas',
                            posicion_coherencia=posicion_anotaciones[i][0], 
                            posicion_diversidad=posicion_anotaciones[i][1])

        df_nube, coherencia, diversidad = crea_df_para_nube(df_modelo, 
                                                            df_metricas, 
                                                            valor['model_id'],
                                                            valor['subconjunto'],
                                                            valor['columna'],
                                                            valor['numero_topicos'])
        generate_wordcloud(df_nube, "palabra", f'{nombre_modelo}_{dict_criterios[i]}__{valor['model_id']}_wc')
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
df_modelo, df_metricas, _ = calcular_y_graficar(df_lda, nombre_modelo)

# NMF ----------------------------------------------
nombre_modelo = 'NMF'
df_modelo, df_metricas, dict_hiperparametros = calcular_y_graficar(df_nmf, nombre_modelo)
print(dict_hiperparametros)

# BERT ----------------------------------------------
nombre_modelo = 'BERT'
df_modelo, df_metricas, dict_hiperparametros = calcular_y_graficar(df_bert, nombre_modelo)
print(dict_hiperparametros)
