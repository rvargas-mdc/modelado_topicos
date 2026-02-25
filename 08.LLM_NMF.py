import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from pathlib import Path
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from joblib import Parallel, delayed
from tqdm import tqdm
from modulos import *

NUM_PALABRAS_TOPICO = 10

dict_hiperparametros = {
    2: {'n_hip':2,'n_c':25,'init':'random','random_state':42,'solver':'mu','beta_loss':'kullback-leibler','alpha_W':0.0,'alpha_H':0.0, 'shuffle':True, 'tol':0.001},
    3: {'n_hip':3,'n_c':25,'init':'random','random_state':42,'solver':'cd','beta_loss':'frobenius','alpha_W':0.001,'alpha_H':0.001, 'shuffle':True, 'tol':0.001},
    4: {'n_hip':4,'n_c':25,'init':'random','random_state':42,'solver':'mu','beta_loss':'kullback-leibler','alpha_W':0.001,'alpha_H':0.001, 'shuffle':True, 'tol':0.001},
    1: {'n_hip':1,'n_c':25,'init':'random','random_state':42,'solver':'cd','beta_loss':'frobenius','alpha_W':0.0,'alpha_H':0.0, 'shuffle':True, 'tol':0.001},
    5: {'n_hip':5,'n_c':25,'init':'nndsvda','random_state':42,'solver':'cd','beta_loss':'frobenius','alpha_W':0.0,'alpha_H':0.0, 'shuffle':True, 'tol':0.001},
    6: {'n_hip':6,'n_c':25,'init':'nndsvda','random_state':42,'solver':'mu','beta_loss':'kullback-leibler','alpha_W':0.0,'alpha_H':0.0, 'shuffle':True, 'tol':0.001},
    7: {'n_hip':7,'n_c':25,'init':'nndsvda','random_state':42,'solver':'cd','beta_loss':'frobenius','alpha_W':0.001,'alpha_H':0.001, 'shuffle':True, 'tol':0.001},
    8: {'n_hip':8,'n_c':25,'init':'nndsvda','random_state':42,'solver':'mu','beta_loss':'frobenius','alpha_W':0.001,'alpha_H':0.001, 'shuffle':True, 'tol':0.001},
}


def ejecutar_nmf_modelado_topicos(
    df: pl.DataFrame,
    vectorizador: TfidfVectorizer,
    matriz_tfidf: any,
    hiperparametros: dict,
    n_palabras_topico:int = NUM_PALABRAS_TOPICO
):

    print(hiperparametros)
    modelo_nmf = NMF(
        n_components=hiperparametros['n_c'],
        init=hiperparametros['init'],
        random_state=hiperparametros['random_state'],
        solver=hiperparametros['solver'],
        beta_loss=hiperparametros['beta_loss'],
        alpha_W=hiperparametros['alpha_W'],
        alpha_H=hiperparametros['alpha_H'],
        shuffle=hiperparametros['shuffle'],
        tol=hiperparametros['tol']
    )

    try:
        matriz_topicos = modelo_nmf.fit_transform(matriz_tfidf)
    except Exception as e:
        return None, None, None

    topico_dominante = np.argmax(matriz_topicos, axis=1)

    df_resultado = df.with_columns(
        pl.Series(name="topico_principal", values=topico_dominante)
    )

    nombres_caracteristicas = vectorizador.get_feature_names_out()
    diccionario_topicos = {}

    for idx_topico, pesos in enumerate(modelo_nmf.components_):
        indices_top = pesos.argsort()[:-n_palabras_topico - 1:-1]
        palabras_top = [nombres_caracteristicas[i] for i in indices_top]
        diccionario_topicos[idx_topico] = palabras_top

    return df_resultado, modelo_nmf, diccionario_topicos


def calcular_coherencias_nmf(
    modelo_nmf,
    vectorizador,
    textos_tokenizados,
    n_palabras_topico: int = NUM_PALABRAS_TOPICO,
    calcular_cv: bool = False,
    calcular_umass: bool = False    
):

    # =========================================================
    # 1. Extraer palabras principales por tópico
    # =========================================================
    nombres_caracteristicas = vectorizador.get_feature_names_out()
    lista_topicos = []

    for pesos_topico in modelo_nmf.components_:
        indices_top = np.argsort(pesos_topico)[-n_palabras_topico:]
        palabras_topico = [nombres_caracteristicas[i] for i in indices_top]
        lista_topicos.append(palabras_topico)

    # =========================================================
    # 2. Crear diccionario y corpus (Gensim)
    # =========================================================
    diccionario = corpora.Dictionary(textos_tokenizados)
    corpus = [diccionario.doc2bow(texto) for texto in textos_tokenizados]

    # =========================================================
    # 3. Coherencia C_v (usa textos originales)
    # =========================================================
    if calcular_cv:
        modelo_coherencia_cv = CoherenceModel(
            topics=lista_topicos,
            texts=textos_tokenizados,
            dictionary=diccionario,
            coherence="c_v",
            processes=1  # IMPORTANTE para evitar cuelgues
        )

        coherencia_cv = modelo_coherencia_cv.get_coherence()
    else:
        coherencia_cv = None

    # =========================================================
    # 4. Coherencia UMass (usa corpus BoW)
    # =========================================================
    if calcular_umass:
        modelo_coherencia_umass = CoherenceModel(
            topics=lista_topicos,
            corpus=corpus,
            dictionary=diccionario,
            coherence="u_mass"
        )

        coherencia_umass = modelo_coherencia_umass.get_coherence()
    else:
        coherencia_umass = None

    return coherencia_cv, coherencia_umass


RUTA_BASE = str(Path(__file__).resolve().parent)

df_llm_limpia = pl.read_parquet(f'{RUTA_BASE}/datos/denuncias_llm_limpias.parquet')
dfs = {k:df_llm_limpia.filter(pl.col('modelo')==k) 
       for k 
       in list(set(df_llm_limpia['modelo'].to_list()))}

# 2. Run the function
# We look for 3 topics (Tech, Sports, Cooking)
topicos = {}
dict_topicos = {}


columna = 'denuncia_limpia'



def procesar_tipo_df(tipo_df):
    print(tipo_df)
    df = dfs[tipo_df]
    
    lst_textos = [' '.join(d) for d in df[columna].to_list()]

    vectorizador = TfidfVectorizer(
        min_df=5
    )

    matriz_tfidf = vectorizador.fit_transform(
        lst_textos
    )

    combinaciones = [
        (n_com, n_hip)
        for n_com  in range(3,16)
        for n_hip in list(dict_hiperparametros.keys())
    ]

    dict_local = {}
    for (n_com, n_hip) in tqdm(
        combinaciones,
        desc=f"  {tipo_df}",
        leave=False
    ):
        hiperparametros = dict_hiperparametros[n_hip]
        hiperparametros['n_c'] = n_com
        df_resultado, modelo_nmf, topicos = ejecutar_nmf_modelado_topicos(
            df,
            vectorizador=vectorizador,
            matriz_tfidf=matriz_tfidf,
            hiperparametros=hiperparametros
        )
        
        if isinstance(df_resultado, pl.DataFrame):
            metricas = calcular_coherencias_nmf(
                modelo_nmf,
                vectorizador,
                df[columna].to_list(),
                n_palabras_topico=10,
                calcular_cv=False,
                calcular_umass=True
            )

            

            str_hiperparametro = str(hiperparametros)
            numero_topicos = len(topicos.keys())
            for topico, palabras in list(topicos.items()):
                llave = f'{tipo_df}_{str_hiperparametro}_{topico}'

                dict_local[llave] = pl.DataFrame({
                    'tipo_df': [tipo_df] * len(palabras),
                    'columna': [columna] * len(palabras),
                    'numero_topicos': [numero_topicos] * len(palabras),
                    'coherencia_umass': [metricas[1]] * len(palabras),
                    'num_topico': [topico] * len(palabras),
                    'palabra': palabras,
                    'hiperparametros':[str_hiperparametro] * len(palabras)
                })
        else:
            dict_local[llave] = pl.DataFrame({
                'tipo_df': [tipo_df],
                'columna': [columna],
                'numero_topicos': [0],
                'coherencia_umass': [],
                'num_topico': [],
                'palabra': [],
                'hiperparametros':[str_hiperparametro]
            })

    return dict_local

tipos_df = list(dfs.keys())

if True:
    resultados = Parallel(n_jobs=-1)(
        delayed(procesar_tipo_df)(tipo_df)
        for tipo_df in tqdm(tipos_df, desc="Procesando tipo_df")
    )

    dict_topicos = {}
    for resultado in [r for r in resultados ]:
        dict_topicos.update(resultado)
    df_palabras = (
        pl.concat(dict_topicos.values())
        .sort('coherencia_umass', descending=True)
    )

else:
    dict_topicos = {tipo_df:procesar_tipo_df(tipo_df)
                    for tipo_df in tqdm(tipos_df)}

    df_palabras = pl.concat([pl.concat(dict_topicos[k].values()) for k in dict_topicos])


df_palabras.write_parquet(
    f'{RUTA_BASE}/ejecucion/NMF_LLM_{NUM_PALABRAS_TOPICO}.parquet'
)

print(
    df_palabras
    .select(
        'columna',
        'tipo_df',
        'numero_topicos',
        'coherencia_umass'
    )
    .unique()
    .sort('coherencia_umass', descending=True)
)
