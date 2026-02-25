from joblib import Parallel, delayed
from tqdm import tqdm
import polars as pl
import polars as pl
import numpy as np
import random
import math
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pathlib import Path
import polars as pl
from modulos import *
NUM_PALABRAS_TOPICO = 10

def ejecutar_lda_desde_tokens(
    df: pl.DataFrame,
    columna_tokens: str,
    numero_topicos: int = 5,
    iteraciones: int = 10,
    random_state: int = 42
):

    # Extraer lista de textos tokenizados
    textos_tokenizados = df[columna_tokens].to_list()

    # Crear diccionario (id -> palabra)
    diccionario = Dictionary(textos_tokenizados)

    # Filtrar palabras muy raras o muy frecuentes
    diccionario.filter_extremes(
        no_below=5
    )

    # Crear corpus en formato Bag-of-Words
    corpus = [diccionario.doc2bow(texto) for texto in textos_tokenizados]

    # Entrenar modelo LDA
    modelo_lda = LdaModel(
        corpus=corpus,
        id2word=diccionario,
        num_topics=numero_topicos,
        passes=iteraciones,
        random_state=random_state
    )

    return modelo_lda, corpus, diccionario


def calcular_coherencias(
    modelo_lda,
    corpus,
    diccionario,
    textos_tokenizados,
    numero_palabras_top: int = NUM_PALABRAS_TOPICO,
    calcular_cv = False,
    calcular_umass = False
):
    """
    Calcula las métricas de coherencia C_v y UMass para un modelo LDA.
    """

    # Extraer palabras principales por tópico
    lista_topicos = []
    for _, topico in modelo_lda.show_topics(
        num_topics=-1,
        num_words=numero_palabras_top,
        formatted=False
    ):
        palabras_topico = [palabra for palabra, _ in topico]
        lista_topicos.append(palabras_topico)

    # Calcular coherencia C_v
    if calcular_cv:
        modelo_coherencia_cv = CoherenceModel(
            topics=lista_topicos,
            texts=textos_tokenizados,
            dictionary=diccionario,
            coherence='c_v'
        )
        coherencia_cv = modelo_coherencia_cv.get_coherence()
    else:
        coherencia_cv = None

    # Calcular coherencia UMass
    if calcular_umass:
        modelo_coherencia_umass = CoherenceModel(
            topics=lista_topicos,
            corpus=corpus,
            dictionary=diccionario,
            coherence='u_mass'
        )
        coherencia_umass = modelo_coherencia_umass.get_coherence()
    else:
        coherencia_umass = None

    return coherencia_cv, coherencia_umass

def asignar_topico_principal(modelo_lda, corpus):

    topicos_principales = []
    for doc in corpus:
        distribucion = modelo_lda.get_document_topics(doc)
        topico_dominante = max(distribucion, key=lambda x: x[1])[0]
        topicos_principales.append(topico_dominante)

    return topicos_principales

def mostrar_topicos(modelo_lda, num_palabras=NUM_PALABRAS_TOPICO):
    """
    Imprime las palabras más representativas de cada tópico.
    """
    for idx, topico in modelo_lda.show_topics(
        num_topics=-1,
        num_words=num_palabras,
        formatted=False
    ):
        palabras = [palabra for palabra, _ in topico]
        print(f"Tópico {idx}: {palabras}")



RUTA_BASE = str(Path(__file__).resolve().parent)

df_llm_limpia = pl.read_parquet(f'{RUTA_BASE}/datos/denuncias_llm_limpias.parquet')
dfs = {k:df_llm_limpia.filter(pl.col('modelo')==k) 
       for k 
       in list(set(df_llm_limpia['modelo'].to_list()))}

# 2. Run the function
# We look for 3 topics (Tech, Sports, Cooking)
topicos = {}
dict_topicos = {}

def procesar_tipo_df(tipo_df):
    df = dfs[tipo_df]
    dict_local = {}
    columna = 'denuncia_limpia'
    print(tipo_df)

    for numero_topicos in tqdm(range(3,16)):
        print(tipo_df, numero_topicos)
        modelo_lda, corpus, diccionario = ejecutar_lda_desde_tokens(
            df,
            columna,
            numero_topicos=numero_topicos
        )

        palabras_principales = modelo_lda.show_topics(
            num_topics=-1,
            num_words=NUM_PALABRAS_TOPICO,
            formatted=False
        )

        metricas = calcular_coherencias(
            modelo_lda,
            corpus,
            diccionario,
            df[columna].to_list(),
            numero_palabras_top=NUM_PALABRAS_TOPICO,
            calcular_cv=False,
            calcular_umass=True
        )

        for topico in palabras_principales:
            llave = f'{tipo_df}_{numero_topicos}_{topico[0]}'

            dict_local[llave] = pl.DataFrame({
                'tipo_df': [tipo_df for _ in topico[1]],
                'numero_topicos': [numero_topicos for _ in topico[1]],
                'coherencia_umass': [metricas[1] for _ in topico[1]],
                'num_topico': [topico[0] for _ in topico[1]],
                'palabra': [palabra[0] for palabra in topico[1]],
                'peso': [palabra[1] for palabra in topico[1]],
            })

    return dict_local

tipos_df = list(dfs.keys())

resultados = Parallel(n_jobs=-1)(
    delayed(procesar_tipo_df)(tipo_df)
    for tipo_df in tqdm(tipos_df, desc="Procesando tipo_df")
)



for resultado in resultados:
    dict_topicos.update(resultado)

df_lda_llm = pl.concat([df for df in dict_topicos.values()])

df_lda_llm.write_parquet( f'{RUTA_BASE}/ejecucion/LDA_llm_{NUM_PALABRAS_TOPICO}.parquet')