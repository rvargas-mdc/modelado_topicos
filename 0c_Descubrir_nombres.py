from glob import glob 
import os
import polars as pl
from pathlib import Path
from modulos import *

RUTA_BASE = str(Path(__file__).resolve().parent)

df_muestra = pl.read_parquet(f'{RUTA_BASE}/../temporal/datos/muestra_2025.parquet')


muestras = ['obs_tot',
            'obs_orps',
            'obs_com',
            'obs_orps_lsur',
            'obs_com_central']

dict_muestras = {col:df_muestra.filter(pl.col(col)==1) 
                 for col in muestras}
dict_bow = {col:crear_bow(dict_muestras[col],'contenido_letras_espacios',col.replace('obs_','')) 
               for col in muestras}

df_bow = pl.concat([df_ for df_ in dict_bow.values()])
df_bow.write_excel('bow.xlsx')
df_bow.write_parquet(f'{RUTA_BASE}/limpieza/bow.parquet')

# Buscar nombres propios en denunciantes
lst_nombres_propios = df_muestra['nombres_propios']
lst_denunciantes = df_muestra['denunciante']
lst_denunciantes = [n.replace('.',' ').replace(';',' ').replace(',',' ').replace(')',' ').replace('(',' ') for n in lst_denunciantes]
denunciantes = ' '.join(lst_denunciantes)
lst_denunciantes = list(set([normalizar_unidecode(p) for p in denunciantes.split(' ') if len(p)>0]))
df_nombres = (pl.DataFrame({'nombre':[l for l in lst_denunciantes if len(l)>1]})     
   .sort('nombre')
   .write_csv(f'{RUTA_BASE}/limpieza/nombres_denunciantes.csv')
)
lst_nombres_propios = ' '.join(df_muestra['nombres_propios'])
lst_nombres_propios = sorted(list(set(lst_nombres_propios.split(' '))))

df_nombres = (pl.read_csv(f'{RUTA_BASE}/limpieza/nombres_denunciantes_corregido.csv')
                .with_columns(control = pl.lit(1))
)
# Buscar nombres propios en Contenido
# 1. Identificar palabras que inicien en maypuscula
dict_bow_nombres = {col:crear_bow(dict_muestras[col],'nombres_propios',col.replace('obs_','')) 
               for col in muestras}

def extraer_ejemplo(palabra, df=df_muestra):
    lst_frases = palabra_en_columna(df, palabra)
    try:
        return lst_frases[0]
    except:
        return ''

df_bow_nombres = pl.concat([df_ for df_ in dict_bow_nombres.values()])
df_bow_nombres = (df_nombres.join(df_bow_nombres.select('palabra'), 
                                 left_on = 'nombre',
                                 right_on = 'palabra',
                                 how='right')
                            .unique()
                            .filter(pl.col('control').is_null()))
df_bow_nombres = (df_bow_nombres
                  .with_columns(indice=pl.Series(range(df_bow_nombres.shape[0]))))

from tqdm import tqdm
for i in tqdm(range(32)):
    df_bow_nombres2 = (df_bow_nombres.filter((pl.col('indice')>=i*500) &
                                             (pl.col('indice')<(i+1)*500))
                                .with_columns(pl.col('palabra')
                                                .map_elements(extraer_ejemplo, return_dtype=pl.Utf8)
                                                .alias('ejemplo')
                                            )
    )

    df_bow_nombres2.write_excel(f'{RUTA_BASE}/limpieza/bow_nombres_{i:02d}.xlsx')

dict_bow_nombres2 = {}
for i in tqdm(range(9)):
    df_ = (pl.read_excel(f'{RUTA_BASE}/limpieza/bow_nombres_{i:02d}.xlsx')
             .with_columns(pl.col('control').cast(pl.Utf8))
    )
    dict_bow_nombres2[f'{i:02d}'] = df_

df_bow_nombres2 = pl.concat(dict_bow_nombres2.values())
df_bow_nombres2.write_excel(f'{RUTA_BASE}/limpieza/bow_nombres.xlsx')

nombres_propios = lst_nombres_propios[0].split(' ')
denunciantes = lst_denunciantes[0].split(' ')
busqueda = [any([d==w.lower() for d in denunciantes]) for w in nombres_propios]
[n for (n, b) in zip(nombres_propios, busqueda) if not b]

resultado = [w for w in nombres_propios if w.lower() in busqueda]

print(resultado)

palabra_en_columna(df_muestra, 'Agip')

#AN`LISIS