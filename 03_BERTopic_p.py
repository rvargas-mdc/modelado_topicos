import polars as pl
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from pathlib import Path
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from glob import glob
import torch
from modulos import *
from tqdm import tqdm

def ejecutar_bertopic(
    dfs: dict,
    dfsw: dict,
    llave: str,
    columna_texto: str,
    porc_min_tamano_topico: float,
    idioma: str = "spanish"    
):
    """
    Ejecuta BERTopic sobre una columna de texto de un DataFrame.

    Retorna:
    - DataFrame con tópico asignado
    - Modelo BERTopic entrenado
    """
    df_ = dfs[llave]
    df_sw = dfsw[llave]
    lst_stopwords = df_sw['palabra'].to_list()
    min_tamano_topico = int(df_.shape[0]*porc_min_tamano_topico)

    # =========================================================
    # 1. Extraer textos
    # =========================================================
    textos = df_[columna_texto].to_list()

    # =========================================================
    # 2. Modelo de embeddings (multilingüe)
    # =========================================================
    modelo_embeddings = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2",
        device="cuda"
    )
    # Creamos el vectorizador que se encargará de "contar" las palabras para los tópicos.
    # Esto NO afecta a los embeddings (el modelo sigue leyendo frases completas),
    # solo limpia las palabras clave que verás en el resultado final.
    vectorizador_modelo = CountVectorizer(
        stop_words=lst_stopwords
    )

    # =========================================================
    # 3. Configuración y entrenamiento de BERTopic
    # =========================================================
    modelo_bertopic = BERTopic(
        embedding_model=modelo_embeddings,
        vectorizer_model=vectorizador_modelo,
        language=idioma,
        min_topic_size=min_tamano_topico,
        calculate_probabilities=True,
        verbose=True
    )

    try:
        topicos, probabilidades = modelo_bertopic.fit_transform(textos)
        df_resultado = df_.with_columns(
            pl.Series(name="topico_principal", values=topicos)
        )
    except:
        df_resultado, topicos, probabilidades = None, None, None
    # =========================================================
    # 4. Agregar tópico dominante al DataFrame
    # =========================================================

    return df_resultado, modelo_bertopic


def calcular_coherencia_umass_bertopic(
    modelo_bertopic: BERTopic,
    lst_documentos: list[list[str]],
    corpus: list,
    diccionario: Dictionary
) -> float:
    """
    Calcula la coherencia U_Mass de un modelo BERTopic entrenado.

    Parámetros
    ----------
    modelo_bertopic : BERTopic
        Modelo BERTopic ya entrenado.
    lst_documentos : list[list[str]]
        Lista de documentos tokenizados (cada documento es una lista de tokens).

    Retorna
    -------
    float
        Valor de coherencia U_Mass.
    """

    # --------------------------------------------------
    # 1. EXTRAER PALABRAS DE CADA TÓPICO (excluyendo -1)
    # --------------------------------------------------
    topicos = modelo_bertopic.get_topics()

    try:
        palabras_topicos = [
            [palabra for palabra, _ in topicos[id_topico]]
            for id_topico in topicos
            if id_topico != -1  # excluir outliers
        ]

        # --------------------------------------------------
        # 2. CALCULAR COHERENCIA U_MASS
        # --------------------------------------------------
        modelo_coherencia = CoherenceModel(
            topics=palabras_topicos,
            corpus=corpus,
            dictionary=diccionario,
            coherence="u_mass"
        )

        coherencia_umass = modelo_coherencia.get_coherence()
    except:
        coherencia_umass = None

    return coherencia_umass

COLUMNAS_BERT = ['tipo_df',
                'columna',
                'coherencia_umass',
                'porc_min_tamano_topico',
                'Topic',
                'Count',
                'Name',
                'Representation']

def actualizar_resultado():
    archivos_bert = glob(f'{RUTA_BASE}/ejecucion/bert_*.parquet')
    if len(archivos_bert)>0:
        df_p = pl.read_parquet(archivos_bert)
        print(df_p.select('tipo_df','columna').unique())
        df_p.select(COLUMNAS_BERT
            ).write_parquet( f'{RUTA_BASE}/ejecucion/crudo_bert.parquet')
    return df_p

RUTA_BASE = str(Path(__file__).resolve().parent)

lst_llaves, dfs, dfsw = conjuntos_de_datos()
lst_llaves = sorted(lst_llaves)
try:
    df_p = actualizar_resultado()
except:
    df_p = pl.DataFrame({'llave':['-']})

dict_topicos = {}

for i,llave in tqdm(enumerate(lst_llaves)):
    print(i,llave)
    columna_base = llave.split('-')[1]
    tipo_df = llave.split('-')[0]
    df_ = dfs[llave]
    columnas = [columna_base, columna_base+'_bert']
    for columna in columnas:
        lst_documentos = (
            df_
            .select(
                pl.col(columna)
                .str.to_lowercase()
                .str.replace_all(r"[^\w\s]", " ")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .str.split(" ")
            )
            .to_series()
            .to_list()
        )        
        diccionario = Dictionary(lst_documentos)
        corpus = [diccionario.doc2bow(doc) for doc in lst_documentos]
        for porc_min_tamano_topico in [0.05, 0.01]:
            hecho = df_p.filter((pl.col('tipo_df')==tipo_df) &
                                (pl.col('columna')==columna) &
                                (pl.col('porc_min_tamano_topico')==porc_min_tamano_topico)
                        ).shape[0]
            if hecho==0:
                print(f'Ejecutando: {llave} {porc_min_tamano_topico}')    
                df_topicos, modelo_bertopic = ejecutar_bertopic(dfs, 
                                                                dfsw, 
                                                                llave,
                                                                columna,
                                                                porc_min_tamano_topico)
                try:
                    informacion_topicos = modelo_bertopic.get_topic_info()
                except:
                    informacion_topicos = None

                if isinstance(informacion_topicos, pd.core.frame.DataFrame):
                    metrica = calcular_coherencia_umass_bertopic(    modelo_bertopic,
                        lst_documentos,
                        corpus,
                        diccionario
                    )
                    df_t = pl.from_pandas(modelo_bertopic.get_topic_info())
                    df_t = (df_t.with_columns(tipo_df=pl.lit(tipo_df),
                                                columna=pl.lit(columna),
                                                coherencia_umass = pl.lit(metrica),
                                                porc_min_tamano_topico = pl.lit(porc_min_tamano_topico)
                                                )
                           )
                    df_t = df_t.select(COLUMNAS_BERT)
                else:
                    df_t = pl.DataFrame({
                         'tipo_df':[tipo_df],
                         'columna':[columna],
                         'coherencia_umass':[None],
                         'porc_min_tamano_topico':[None],
                         'Topic':[None],
                         'Count':[None],
                        'Name':[None],
                        'Representation':[None]
                                        })

                llave = f'{tipo_df}-{columna}'
                dict_topicos[llave] = df_t

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            else:
                print(f'Previamente hecho {llave} {porc_min_tamano_topico}')

print(i)
lst_df = [df for df in dict_topicos.values() if df.shape[0]>1]
df_palabras = pl.concat(lst_df).sort('coherencia_umass',descending=True)

#df_palabras.group_by(['tipo_df','numero_topicos']).agg(pl.col("palabra").str.join(", ").alias("palabras"))
#df_palabras.write_parquet( f'{RUTA_BASE}/ejecucion/bert_1_26.parquet')
df_palabras.write_parquet( f'{RUTA_BASE}/ejecucion/bert_{int(i)}.parquet')

df_p = actualizar_resultado()

print(df_p.select('columna',
                         'tipo_df',
                         'coherencia_umass',
                         'porc_min_tamano_topico')
                 .unique()
                 .sort('coherencia_umass',
                       descending=True))
