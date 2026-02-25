import polars as pl
import os
from IPython.display import display
import math
import plotly.express as px
from pathlib import Path
import math
from unidecode import unidecode
import re
from modulos import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


RUTA_BASE = str(Path(__file__).resolve().parent)
df = pl.read_parquet(f'{RUTA_BASE}/../temporal/datos/resoluciones_bancos_2025.parquet')


df_ = (df.pivot(on='area',
                      index='lugar',
                      values='documento',
                      aggregate_function='len')
  .with_columns(pl.sum_horizontal(
                          pl.all().exclude(pl.Utf8)
                          )
                .alias("total"))
  .sort('total', descending=True)
)

dfw = df.with_columns(
            region=pl.when(pl.col('lugar').str.contains('Sede')).then(pl.lit('Lima')).otherwise(pl.lit('Provincias')),
            largo_contenido=pl.col('contenido').str.len_chars())

dfw.pivot(on='area',
                      index='region',
                      values='largo_contenido',
                      aggregate_function='mean')

df_.write_excel(f'{RUTA_BASE}/graficos/por_area_lugar.xlsx')

def encontrar_archivo(documento):
    ruta = f'{RUTA_BASE}/limpieza/nombres/{documento}.txt'
    return int(os.path.exists(ruta))

df = (df.with_columns(
        pl.col('documento')
        .map_elements(encontrar_archivo, return_dtype=pl.Int8)
        .alias('existen_nombres')
    )
)

def calcular_tamano_muestra(poblacion, nivel_confianza=0.95, margen_error=0.05, proporcion=0.5):
    valores_z = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    
    try:
        valor_z = valores_z[nivel_confianza]

        numerador = (valor_z ** 2) * proporcion * (1 - proporcion)
        denominador = margen_error ** 2
        muestra_inicial = numerador / denominador

        muestra_ajustada = muestra_inicial / (1 + ((muestra_inicial - 1) / poblacion))
        return math.ceil(muestra_ajustada)
    except:
        print('Nivel de confianza no definido')
        print(nivel_confianza, margen_error, proporcion)
        return -1

df_orps = df.filter(pl.col('area')=='ORPS')
df_com = df.filter(pl.col('area')=='Comisión')
df_orps_lsur = df.filter((pl.col('area')=='ORPS') & (pl.col('lugar')=='Sede Lima Sur'))
df_com_central = df.filter((pl.col('area')=='Comisión') & (pl.col('lugar')=='Sede Central'))

n_total = calcular_tamano_muestra(df.shape[0])
n_orps = calcular_tamano_muestra(df_orps.shape[0])
n_com = calcular_tamano_muestra(df_com.shape[0])
n_orps_lsur = calcular_tamano_muestra(df_orps_lsur.shape[0])
n_com_central = calcular_tamano_muestra(df_com_central.shape[0])

print(df.shape[0], n_total)
print(df_orps.shape[0], n_orps)
print(df_com.shape[0], n_com)
print(df_orps_lsur.shape[0], n_orps_lsur)
print(df_com_central.shape[0], n_com_central)

semilla = 42

df_m = df.sample(n=n_total, seed=semilla)
df_orps_m = df_orps.sample(n=n_orps, seed=semilla)
df_com_m = df_com.sample(n=n_com, seed=semilla)
df_orps_lsur_m = df_orps.sample(n=n_orps_lsur, seed=semilla)
df_com_central_m = df_com.sample(n=n_com_central, seed=semilla)

df_m = df_m.with_columns(obs_tot=pl.lit(1))
df_orps_m = df_orps_m.with_columns(obs_orps=pl.lit(1))
df_com_m = df_orps_m.with_columns(obs_com=pl.lit(1))
df_orps_lsur_m  = df_orps_lsur_m.with_columns(obs_orps_lsur=pl.lit(1))
df_com_central_m  = df_com_central_m.with_columns(obs_com_central=pl.lit(1))

df_muestra = (df
        .join(df_m.select('documento','obs_tot'), on='documento', how='left')
        .join(df_orps_m.select('documento','obs_orps'), on='documento', how='left')
        .join(df_com_m.select('documento','obs_com'), on='documento', how='left')
        .join(df_orps_lsur_m.select('documento','obs_orps_lsur'), on='documento', how='left')
        .join(df_com_central_m.select('documento','obs_com_central'), on='documento', how='left')
        .with_columns(pl.col('obs_tot').fill_null(0),
                    pl.col('obs_orps').fill_null(0),
                    pl.col('obs_com').fill_null(0),
                    pl.col('obs_orps_lsur').fill_null(0),
                    pl.col('obs_com_central').fill_null(0),
                    )
        .with_columns((pl.col('obs_tot')+
                    pl.col('obs_orps')+
                    pl.col('obs_com')+
                    pl.col('obs_orps_lsur')+
                    pl.col('obs_com_central')
                    ).alias('obs'))
        .filter(pl.col('obs')>0)
)



df_muestra = (df_muestra
              .with_columns(pl.col('contenido')
                            .map_elements(normalizar_unidecode, return_dtype=pl.Utf8)
                            .alias('contenido_unidecode')
              )
              .with_columns(pl.col('contenido_unidecode')
                            .map_elements(preservar_letras_y_espacios, return_dtype=pl.Utf8)
                            .alias('contenido_letras_espacios'),
                            pl.col('denunciante')
                            .map_elements(normalizar_unidecode, return_dtype=pl.Utf8)
                            .map_elements(preservar_letras_y_espacios, return_dtype=pl.Utf8)
                            .alias('denunciante_letras_espacios')
                           )
              .with_columns(pl.col('contenido')
                            .map_elements(extraer_nombres_propios, return_dtype=pl.Utf8)
                            .alias('nombres_propios')
              )                           
              .with_columns(pl.col('nombres_propios')
                                       .map_elements(normalizar_unidecode_unicos,return_dtype=pl.Utf8)
                                       .alias('nombres_propios_unidecode'))

             )


df_muestra.write_parquet(f'{RUTA_BASE}/../temporal/datos/muestra_2025.parquet')
#-- contenido_letras_espacios

df_muestra = pl.read_parquet(f'{RUTA_BASE}/../temporal/datos/muestra_2025.parquet')