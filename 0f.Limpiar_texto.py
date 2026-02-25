import stanza
import re
import polars as pl
from modseccionar_denunciantes import *
from pathlib import Path
from modulos import *
import time
import os
RUTA_BASE = str(Path(__file__).resolve().parent)

stanza.download("es")
nlp = stanza.Pipeline(
    lang="es",
    processors="tokenize,mwt,lemma",
    tokenize_no_ssplit=True,
    use_gpu=True
)


columnas_subconjunto = ['documento',
 'area',
 'lugar',
 'zona',
 'contenido_anonimizado',
 'denuncia_anonimizada',
 'dictamen_anonimizado',
 'denuncia_dictamen_anonimizado']

df_muestra = pl.read_parquet(f'{RUTA_BASE}/../temporal/datos/muestra_denuncias_2025.parquet')
df_muestra = (df_muestra.with_columns(pl.concat_str(pl.col('denuncia_anonimizada'),
                                                   pl.lit('\n'),
                                                   pl.col('dictamen_anonimizado'))
                                        .alias('denuncia_dictamen_anonimizado')
                                    )
                       )
(df_muestra.select(columnas_subconjunto)
          .write_parquet(f'{RUTA_BASE}/../temporal/datos/muestra_denuncia_dictamen_2025.parquet')
)
dfs = {'todos':df_muestra.filter(pl.col('obs_tot')==1), 
       'orps':df_muestra.filter(pl.col('obs_orps')==1), 
       'comisión':df_muestra.filter(pl.col('obs_com')==1), 
       'orps_lim':df_muestra.filter(pl.col('obs_orps_lsur')==1), 
       'comisión_lim':df_muestra.filter(pl.col('obs_com_central')==1) }


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

columnas_limpiar = [ 
 'dictamen_anonimizado', 
 'denuncia_anonimizada',
 'contenido_anonimizado',
 'denuncia_dictamen_anonimizado'
]
dict_bows = {}
dict_stopwords = {}


for columna in columnas_limpiar:
    for k in dfs.keys():
        llave = f'{k}-{columna}'
        df_ = dfs[k]
        

        df_ = df_.with_columns(pl.col(columna).map_elements(unidecode,return_dtype=pl.Utf8).alias(f'{columna}_limpia'))
        df_bow = crear_bow(df_, columna, llave)
        df_bow = df_bow.with_columns(pl.col('palabra')
                                       .map_elements(limpiar,return_dtype=pl.Utf8)
                                       .alias('palabra'),
                                    pl.lit('llave').alias('llave'))
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
#df_bows.filter(pl.col('palabra').is_in(STOPWORDS_ES))
df_bows.write_excel('bow_partes.xlsx')

df_stopwords = pl.concat(dict_stopwords.values())
#df_stopwords.filter(pl.col('palabra').is_in(STOPWORDS_ES))
df_stopwords.write_excel('stopwords.xlsx')
print('Archivos creados')

df_stopwords_dom = (pl.DataFrame({'palabra':STOPWORDS_DOMINIO + STOPWORDS_ES}) 
                      .with_columns(pl.lit(1).alias('control'))
                   )           
def preparar_texto(dict_texto: dict, 
                   dict_stopwords: dict = dict_stopwords, 
                   min_len: int = 3):
    
    #print(dict_texto)
    texto = dict_texto[list(dict_texto.keys())[0]]
    llave = dict_texto[list(dict_texto.keys())[1]]

    df_stopwords = dict_stopwords[llave]
    # Limpieza básica 
    texto = texto.lower()
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    doc = nlp(texto)

    tokens = {}
    tokens_dominio = {}
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

        
        df_palabras = (df_palabras_texto
                        .join(df_stopwords_dom,
                            on='palabra',
                            how='left')
                        .filter(pl.col('control').is_null())
                    )
        tokens_dominio[i] = df_palabras['palabra'].to_list()


    tokens1 = [p for palabras in tokens.values() for p in palabras]
    tokens_dominio1= [p for palabras in tokens_dominio.values() for p in palabras]
    return [tokens1, tokens_dominio1]





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
comisión-denuncia_dictamen_anonimizado
Tiempo transcurrido: 404.1796 segundos
comisión_lim-denuncia_dictamen_anonimizado
Tiempo transcurrido: 0.0692 segundos
orps-denuncia_dictamen_anonimizado
Tiempo transcurrido: 472.6412 segundos
orps_lim-denuncia_dictamen_anonimizado
Tiempo transcurrido: 421.8143 segundos
todos-denuncia_dictamen_anonimizado
2.99 horas
"""

inicio = time.perf_counter()

MARCA_SALTO = "__PUNTO_SALTO__"
dict_muestra_limpia = {}
llaves = sorted(list(dfs.keys()))

columnas_limpiar_itera = [l for l in columnas_limpiar if 'denuncia_dictamen' in l]
#columnas_limpiar_itera = columnas_limpiar
for k in llaves:
    for columna in columnas_limpiar_itera:
          
        llave = f'{k}-{columna}'
        columna_limpia = f'{columna}_limpio'
        ruta_parquet = f'{RUTA_BASE}/datos/muestra_limpia_{llave}.parquet'
        print(llave)
        if os.path.exists(ruta_parquet) and False:
            df_ = pl.read_parquet(ruta_parquet)
            dict_muestra_limpia[llave] = df_
            #print(df_[columna_limpia].to_list()[0][:1])
        else:
            df_ = dfs[k]
            df_ = (df_
                .select(columnas_subconjunto)
                .with_columns(
                    pl.lit(llave).alias('llave')
                )
                .with_columns(
                    pl.struct(pl.col(columna),pl.col("llave")).alias('estructura')
                )
            )
            df_ = df_.with_columns(
                    pl.col("estructura")
                    .map_elements(preparar_texto, return_dtype=pl.List(pl.List(pl.Utf8)))
                    .alias(f"{columna}_limpio_doble")
                )
            df_ = df_.with_columns([
                pl.col(f"{columna}_limpio_doble").list.get(0).alias(f"{columna}_limpio"),
                pl.col(f"{columna}_limpio_doble").list.get(1).alias(f"{columna}_limpio_dom")
            ])
            df_ = df_.with_columns(
                pl.col(columna)
                # 1. Proteger ".\n"
                .str.replace_all(r"\.\n", f".{MARCA_SALTO}")
                # 2. Reemplazar todos los saltos restantes
                .str.replace_all(r"\n", " ")
                # 3. Restaurar ".\n"
                .str.replace_all(MARCA_SALTO, "\n")
                .alias(f"{columna}_bert")
            )
            df_ = df_.drop('estructura',f"{columna}_limpio_doble").write_parquet(ruta_parquet)
        dict_muestra_limpia[llave] = df_
        fin = time.perf_counter()
        lapso = fin - inicio
        print(f"Tiempo transcurrido: {lapso:.4f} segundos")
        inicio = fin


todas_las_columnas = (['llave'] 
                      + columnas_subconjunto 
                      + [c+'_limpio' for c in columnas_limpiar]
                      + [c+'_limpio_dom' for c in columnas_limpiar]                      
                      + [c+'_bert' for c in columnas_limpiar])

def completar_columnas(llave,df_t):
    print(llave)
    columnas_faltantes = list(set(todas_las_columnas).difference(set(df_t.columns)))
    print('.........', len(columnas_faltantes))
    for c in columnas_faltantes:
        if 'limpio' in c:
            df_t = df_t.with_columns(pl.lit(['']).alias(c))
        else:
            df_t = df_t.with_columns(pl.lit('').alias(c))
    df_t = df_t.select(todas_las_columnas)
    return df_t

df_muestra_limpia = pl.concat([completar_columnas(llave,df_) 
                               for llave,df_ in dict_muestra_limpia.items()] )

df_muestra_limpia.write_parquet(f'{RUTA_BASE}/datos/muestra_2025_limpia.parquet')

for i,c in enumerate(columnas_limpiar):
    df_muestra_limpia = df_muestra_limpia.with_columns(pl.when(pl.col(f'{c}_limpio')==[''])
                                                         .then(pl.lit(0))
                                                         .otherwise(pl.lit(1))
                                                         .alias(f'largo_{i}'),
                                                        pl.when(pl.col(f'{c}_limpio_dom')==[''])
                                                         .then(pl.lit(0))
                                                         .otherwise(pl.lit(1))
                                                         .alias(f'largo_{i}_dom'))
for row in (df_muestra_limpia.group_by('llave').agg(pl.col('largo_0').sum(),
                                        pl.col('largo_1').sum(),
                                        pl.col('largo_2').sum(),
                                        pl.col('largo_0_dom').sum(),
                                        pl.col('largo_1_dom').sum(),
                                        pl.col('largo_2_dom').sum()
                                        ).sort('llave')
).iter_rows():
    print(row)

"""
('comisión-contenido_anonimizado', 0, 0, 322)
('comisión-denuncia_anonimizada', 0, 322, 0)
('comisión-dictamen_anonimizado', 322, 0, 0)
('comisión_lim-contenido_anonimizado', 0, 0, 281)
('comisión_lim-denuncia_anonimizada', 0, 281, 0)
('comisión_lim-dictamen_anonimizado', 281, 0, 0)
('orps-contenido_anonimizado', 0, 0, 322)
('orps-denuncia_anonimizada', 0, 322, 0)
('orps-dictamen_anonimizado', 322, 0, 0)
('orps_lim-contenido_anonimizado', 0, 0, 280)
('orps_lim-denuncia_anonimizada', 0, 280, 0)
('orps_lim-dictamen_anonimizado', 280, 0, 0)
('todos-contenido_anonimizado', 0, 0, 352)
('todos-denuncia_anonimizada', 0, 352, 0)
('todos-dictamen_anonimizado', 352, 0, 0)
"""

verificando = False
if verificando:
    contenidos = df_muestra['contenido_anonimizado'].to_list()
    palabra = 'transacción'
    [[o for o in c.split('\n') if palabra in o] for c in contenidos if palabra in c]

