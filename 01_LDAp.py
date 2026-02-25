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


def asignar_topico_principal(modelo_lda, corpus):

    topicos_principales = []
    for doc in corpus:
        distribucion = modelo_lda.get_document_topics(doc)
        topico_dominante = max(distribucion, key=lambda x: x[1])[0]
        topicos_principales.append(topico_dominante)

    return topicos_principales

def mostrar_topicos(modelo_lda, num_palabras=10):
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


from gensim.models import CoherenceModel


def calcular_coherencias(
    modelo_lda,
    corpus,
    diccionario,
    textos_tokenizados,
    numero_palabras_top: int = 10,
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

def calcular_coherencias_rapidas(
    modelo_lda,
    corpus,
    diccionario,
    textos_tokenizados,
    numero_palabras_top=5,
    porc_documentos=0.1,
    semilla = 42,
    calcular_cv = False,
    calcular_umass = False
):
    """
    Versión optimizada para cálculo rápido de coherencias.
    """

    random.seed(semilla)

    max_documentos = int(porc_documentos * len(textos_tokenizados))
    # Muestreo de textos
    textos_muestra = (
        textos_tokenizados
        if len(textos_tokenizados) <= max_documentos
        else random.sample(textos_tokenizados, max_documentos)
    )

    # Extraer tópicos
    lista_topicos = [
        [palabra for palabra, _ in topico]
        for _, topico in modelo_lda.show_topics(
            num_topics=-1,
            num_words=numero_palabras_top,
            formatted=False
        )
    ]

    if calcular_cv:
        # C_v (lento, pero acotado)
        modelo_cv = CoherenceModel(
            topics=lista_topicos,
            texts=textos_muestra,
            dictionary=diccionario,
            coherence='c_v'
        )
        coherencia_cv = modelo_cv.get_coherence()
    else:
        coherencia_cv = None

    if calcular_umass:
        modelo_umass = CoherenceModel(
            topics=lista_topicos,
            corpus=corpus,
            dictionary=diccionario,
            coherence='u_mass'
        )
        coherencia_umass = modelo_umass.get_coherence()
    else:
        coherencia_umass = None

    return coherencia_cv, coherencia_umass


# ==========================================
# 2. Main Execution
# ==========================================

RUTA_BASE = str(Path(__file__).resolve().parent)

lst_llaves, dfs, _ = conjuntos_de_datos()

# 2. Run the function
# We look for 3 topics (Tech, Sports, Cooking)
topicos = {}
dict_topicos = {}

def procesar_tipo_df(tipo_df):
    df = dfs[tipo_df]
    dict_local = {}
    columna_base = tipo_df.split('-')[1] + '_limpio'
    columnas = [columna_base, columna_base+'_dom']
    print(tipo_df)
    combinaciones = [(columna, n_topicos) 
        for columna in columnas
        for n_topicos in range(3,36)]
    for columna, numero_topicos in tqdm(
        combinaciones,
        desc=f"  {tipo_df}",
        leave=False
    ):
            
        modelo_lda, corpus, diccionario = ejecutar_lda_desde_tokens(
            df,
            columna,
            numero_topicos=numero_topicos
        )

        topicos_principales = asignar_topico_principal(
            modelo_lda,
            corpus
        )

        palabras_principales = modelo_lda.show_topics(
            num_topics=-1,
            num_words=10,
            formatted=False
        )

        metricas = calcular_coherencias(
            modelo_lda,
            corpus,
            diccionario,
            df[columna].to_list(),
            numero_palabras_top=10,
            calcular_cv=False,
            calcular_umass=True
        )

        for topico in palabras_principales:
            llave = f'{tipo_df}_{numero_topicos}_{topico[0]}_{columna}'

            dict_local[llave] = pl.DataFrame({
                'tipo_df': [tipo_df for _ in topico[1]],
                'columna': [columna for _ in topico[1]],
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

df_lda = pl.concat([df for df in dict_topicos.values()])

df_lda.write_parquet( f'{RUTA_BASE}/ejecucion/LDA.parquet')