import stanza
import re
import polars as pl
from modseccionar_denunciantes import *
from pathlib import Path
from modulos import *
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


RUTA_BASE = str(Path(__file__).resolve().parent)

stanza.download("es")
nlp = stanza.Pipeline(
    lang="es",
    processors="tokenize,mwt,lemma",
    tokenize_no_ssplit=True,
    use_gpu=True
)


dict_df_llm = conjuntos_de_datos_llm()
modelo = list(dict_df_llm.keys())[0]
df_llm = dict_df_llm[modelo]
columnas_subconjunto = df_llm.columns
columnas_limpiar = ['denuncia']
columna = 'denuncia'

def limpiar(texto):
    respuesta = unidecode(texto).lower()
    try:
        respuesta_letras = re.findall('[a-z]+',respuesta)[0]
    except:
        respuesta_letras = '-'
    if respuesta == respuesta_letras:
        return respuesta
    else:
        return '-'

dict_bows = {}
dict_stopwords = {}



for k in dict_df_llm.keys():
    llave = k
    df_ = dict_df_llm[k]
        
    df_ = df_.with_columns(pl.col(columna).map_elements(unidecode,return_dtype=pl.Utf8).alias(f'{columna}_limpia'))
    df_bow = crear_bow(df_, columna, llave)
    df_bow = df_bow.with_columns(pl.col('palabra')
                                       .map_elements(limpiar,return_dtype=pl.Utf8)
                                       .alias('palabra'),
                                    pl.lit(llave).alias('llave'))
    lst_sw = df_bow.filter((pl.col('porc_documentos')>=0.5) &
                            (~pl.col('palabra').is_in(NO_STOPWORDS))
    )['palabra'].to_list()
    lst_sw = list(set(lst_sw + STOPWORDS_ES))
    lst_sw = [l for l in lst_sw if l!='-']
    df_sw = (pl.DataFrame({'palabra':lst_sw,
                            'control':[1]*len(lst_sw)})
                .with_columns( pl.lit(llave).alias('llave'))
    )
    dict_stopwords[llave] = df_sw
    dict_bows[llave] = df_bow


df_bows = pl.concat(dict_bows.values())
df_bows.write_excel('bow_partes_llm.xlsx')

df_stopwords = pl.concat(dict_stopwords.values())
df_stopwords.write_excel('stopwords_llm.xlsx')
print('Archivos creados')

df_stopwords_dom = (pl.DataFrame({'palabra':STOPWORDS_DOMINIO + STOPWORDS_ES}) 
                      .with_columns(pl.lit(1).alias('control'))
                   )           
def preparar_texto(texto_modelo: str, 
                   dict_stopwords: dict = dict_stopwords, 
                   min_len: int = 3):
    
    texto = texto_modelo.split('~')[0]
    modelo = texto_modelo.split('~')[1]
    df_stopwords = dict_stopwords[modelo]
    # Limpieza básica 
    texto = texto.lower()
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    doc = nlp(texto)

    tokens = {}

    for i, oracion in enumerate(doc.sentences):
        
        palabras = oracion.words
        palabras = [unidecode(p.lemma) for p in palabras if (len(p.lemma) >= min_len) and (p.lemma.isalpha())]
        df_palabras_texto = (pl.DataFrame({'palabra':palabras})
                               .with_columns(pl.col('palabra').cast(pl.Utf8)))
        df_palabras = (df_palabras_texto
                        .join(df_stopwords,
                            on='palabra',
                            how='left')
                        .filter(pl.col('control').is_null())
                    )
        tokens[i] = df_palabras['palabra'].to_list()
    tokens = [p for palabras in tokens.values() for p in palabras]
    return tokens


"""
todos-dictamen_anonimizado
Tiempo transcurrido: 201.2951 segundos
todos-denuncia_anonimizada
Tiempo transcurrido: 211.8869 segundos
todos-contenido_anonimizado
Tiempo transcurrido: 1408.6101 segundos
orps-dictamen_anonimizado
Tiempo transcurrido: 217.3836 segundos
orps-denuncia_anonimizada
Tiempo transcurrido: 227.8992 segundos
orps-contenido_anonimizado
Tiempo transcurrido: 3662.2349 segundos
comisión-dictamen_anonimizado
Tiempo transcurrido: 753.6138 segundos
comisión-denuncia_anonimizada
Tiempo transcurrido: 295.3645 segundos
comisión-contenido_anonimizado
Tiempo transcurrido: 1110.7457 segundos
orps_lim-dictamen_anonimizado
Tiempo transcurrido: 186.4227 segundos
orps_lim-denuncia_anonimizada
Tiempo transcurrido: 77.4217 segundos
orps_lim-contenido_anonimizado
Tiempo transcurrido: 906.4413 segundos
comisión_lim-dictamen_anonimizado
Tiempo transcurrido: 139.4403 segundos
comisión_lim-denuncia_anonimizada
Tiempo transcurrido: 204.5240 segundos
comisión_lim-contenido_anonimizado
Tiempo transcurrido: 1171.1124 segundos

2.99 horas
"""

MARCA_SALTO = "__PUNTO_SALTO__"

def procesar_modelo(modelo, dict_df_llm, columna):

    columna_limpia = f'{columna}_limpia'
    ruta_parquet = f'{RUTA_BASE}/datos/muestra_llm_limpia_{modelo}.parquet'

    inicio = time.perf_counter()

    if os.path.exists(ruta_parquet):
        df_ = pl.read_parquet(ruta_parquet)
    else:
        df_ = (
            dict_df_llm[modelo]
            .select(columnas_subconjunto)
            .with_columns(pl.lit(modelo).alias('llave'))
            .with_columns(
                pl.format('{}~{}',
                            pl.col(columna), 
                            pl.col("llave"))
                .alias('estructura')
            )
            .with_columns(
                pl.col("estructura")
                .map_elements(preparar_texto, return_dtype=pl.List(pl.Utf8))
                .alias(columna_limpia)
            )
            .with_columns(
                pl.col(columna)
                .str.replace_all(r"\.\n", f".{MARCA_SALTO}")
                .str.replace_all(r"\n", " ")
                .str.replace_all(MARCA_SALTO, "\n")
                .alias(f"{columna}_bert")
            )
        )
        df_.write_parquet(ruta_parquet)

    fin = time.perf_counter()
    print(f'{modelo}: fin - inicio')
    return  df_



dict_muestra_limpia = {}
for modelo in dict_df_llm.keys():
    dict_muestra_limpia[modelo] = procesar_modelo(modelo, dict_df_llm, columna)


df_llm_limpia = pl.concat( dict_muestra_limpia.values()).drop('denuncia_bert')

df_llm_limpia.write_parquet(f'{RUTA_BASE}/datos/denuncias_llm_limpias.parquet')

