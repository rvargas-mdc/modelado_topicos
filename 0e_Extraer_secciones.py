from glob import glob 
import os
import polars as pl
import ast
import re
import numpy as np
from modseccionar_denunciantes import *
from modcrear_dataset import *
from pathlib import Path
from modulos import *

RUTA_BASE = str(Path(__file__).resolve().parent)
df_muestra = pl.read_parquet(f'{RUTA_BASE}/../temporal/datos/m2025_anonimizados_numeros.parquet')


lst_limites_1 = ['antecedentes', 
                 'antecedentes. -', 
                 'hecho', 
                 'hechos',
                 'antecedente']

lst_limites_2 = ['analisis',
                 'an´lisis',
                 'analisis del caso',
                 'cuestion en discusion',
                 'cuestion previa',
                 'cuestiones en discusion',
                 'aplicación al caso en concreto',
                 'marco jurídico aplicable a la noción de consumidor final',
                 'sobre el error material en la resolución n° 1',
                 'sobre la improcedencia parcial',
                 'sobreel desistimiento',
                 'marco-jurídico'
                 ]

lst_limites_3 = ['PRIMERO:',
                 'Primero:',
                 'RESUELVE:',
                 'DECISIÓN DE LA COMISIÓN',
                 'SE HA RESUELTO'
]


def obtener_antecedente(contenido):
    contenido_cam = contenido
    contenido_min = destildar(contenido).lower()
    antecedente = None
    for limite in lst_limites_2:
        lst_borde = [f'{limite}\n',
                     f'{limite}:\n',
                     f'{limite} \n',
                     f'{limite} :\n'
                     ]
        for borde in lst_borde:
            if borde in contenido_min:
                antecedente = contenido_min.split(borde)[0]
                break
            else:
                borde = f'{limite}:\n'
                if borde in contenido_min:
                    antecedente = contenido_min.split(borde)[0]
                    break
    if antecedente:
        contenido_cam = contenido_cam[:len(antecedente)]
        contenido_min = destildar(contenido_cam).lower()
        for limite in lst_limites_1:
            borde = f'{limite}\n'
            if borde in contenido_min:
                antecedente = contenido_min.split(borde)[1]
                break
            else:
                borde = f'{limite}:\n'
                if borde in contenido_min:
                    antecedente = contenido_min.split(borde)[0]
                    break
        antecedente = contenido_cam[-len(antecedente):]
    else:
        antecedente = None
    return antecedente


df_muestra = (df_muestra.with_columns(
        pl.col('contenido_anonimizado')
        .map_elements(obtener_antecedente, return_dtype=pl.Utf8)
        .alias('antecedente_anonimizado')
    )
)
print('Sin antecedente:',df_muestra.filter(pl.col('antecedente_anonimizado').is_null()).shape[0])

crear_revision(df_muestra, 
               'antecedente_anonimizado', 
               'contenido_anonimizado', 
               limitar_a=None)

df_muestra = actualizar_valores_nulos(df_muestra, 'antecedente_anonimizado')


def obtener_resolucion(contenido):
    dictamenes = [contenido.split(l)[-1] for l in lst_limites_3]
    longitudes =[len(d) for d in dictamenes]
    dictamen = dictamenes[np.argmin(longitudes)]
    if len(dictamen)>len(contenido)*0.9:
        return None
    limite = lst_limites_3[np.argmin(longitudes)]
    if limite[0]=='P':
        dictamen = limite + ' ' + dictamen
    return dictamen

df_muestra = (df_muestra.with_columns(
        pl.col('contenido_anonimizado')
        .map_elements(obtener_resolucion, return_dtype=pl.Utf8)
        .alias('dictamen_anonimizado')
    )
)
print('Sin dictamen:',df_muestra.filter(pl.col('dictamen_anonimizado').is_null()).shape[0])

crear_revision(df_muestra, 
               'dictamen_anonimizado', 
               'contenido_anonimizado', 
               limitar_a=None)

df_muestra.write_parquet(f'{RUTA_BASE}/../temporal/datos/muestra_secciones_2025.parquet')

