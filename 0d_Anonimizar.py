from glob import glob 
import os
import polars as pl
from pathlib import Path
import re
from modulos import *
import math
from multiprocessing import Pool

RUTA_BASE = str(Path(__file__).resolve().parent)

df_muestra = pl.read_parquet(f'{RUTA_BASE}/../temporal/datos/muestra_2025.parquet')


def generar_nombres():
    df_nombres_cuerpo = pl.read_excel(f'{RUTA_BASE}/limpieza/bow_nombres_corregido.xlsx')
    df_nombres_denunciantes = (pl.read_csv(f'{RUTA_BASE}/limpieza/nombres_denunciantes_corregido.csv')
                    .with_columns(control = pl.lit(1))
    )
    df_nombres_comunes = pl.read_csv(f'{RUTA_BASE}/limpieza/nombres_propios_y_comunes.csv',
                                     separator=';')

    lst_nombres = (df_nombres_cuerpo
                .filter(pl.col('control')=='1')['palabra']
                .to_list() +
                df_nombres_denunciantes['nombre']
                .to_list())
    df_nombres = (pl.DataFrame({'nombre':lst_nombres})
                    .join(df_nombres_comunes,
                            on='nombre',
                            how='left')
                    .with_columns(pl.col('permitir_minuscula')
                                    .fill_null(True),
                                  pl.col('excepcion')
                                    .fill_null('-'))    
    )
    return df_nombres

resultados = {}
for ntupla in df_muestra.select('documento','contenido').iter_rows():
    documento = ntupla[0]
    contenido = ntupla[1]

    resultado = {}

    dnirucs = re.findall(r'[0-9]{6,20}',
                        contenido
                     )
    resultado['DNIRUC'] = dnirucs

    rucs = re.findall(r"[12]0(?:,\d{3}){3}",
                        contenido
                     )
    resultado['RUC'] = rucs

    resoluciones = re.findall(
        r"[0-9]+-[\n]{0,1}[0-9]{4}",
        contenido
    )

    resoluciones = list(set(resoluciones))
    resultado['resoluciones'] = resoluciones

    telefonos = [o.lower().split('te')[1:] for o in contenido.split('\n') if 'telf' in o.lower() or 'telé' in o.lower() or 'telef' in o.lower()]
    central = [o.lower().split('central:')[1:] for o in contenido.split('\n') if 'central:' in o.lower()]
    telefonos = [s for sublist in telefonos + central for s in sublist]
    telefonos = [re.findall(r"[^a-zA-Zéá´´ióúÁÉÍÓÚñÑ]+", t) for t in telefonos]
    telefonos = [('~'+s).replace('~.','~').replace('~:','').replace('~','').strip() for sublist in telefonos for s in sublist]
    telefonos = [t[:-1].strip() if t[-1] in [')','/','-','(','+'] else t for t in telefonos if len(t)>2]
    telefonos = [t[1:].strip() if t[0] in [')','°',':'] else t for t in telefonos if len(t)>2]
    telefonos = [t[1:].strip() if t[0] in [')','°',':'] else t for t in telefonos if len(t)>2]

    telefonos = [t for t in telefonos if len(re.findall('[0-9]{3}',t))>0]
    
    telefonos = list(set(telefonos))
    resultado['telefonos'] = telefonos

    patron_email = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    emails = re.findall(patron_email,
                        contenido
                     )

    webs = [[p for p in o.split(' ') if ('www' in p) or ('http' in p)] 
            for o in contenido.split('\n') if ('www' in o) or ('http' in o)]
    webs = [w for sublist in webs for w in sublist]
    webs = sorted(list(set(webs)))
    webs.reverse()



    resultado['correo'] = list(set(emails))
    resultado['web'] = webs

    emails = re.findall(patron_email, contenido)    

    resultados[documento] = resultado

dict_contenidos = {}
lst_datos_sensibles = ['correo','web', 'DNIRUC','RUC','resoluciones','telefonos']
for documento in resultados.keys():
    contenido = df_muestra.filter(pl.col('documento')==documento)['contenido'][0]
    for clave_sensible in lst_datos_sensibles:
        for valor_sensible in resultados[documento][clave_sensible]:
            valor_sensible
            contenido = re.sub(re.escape(valor_sensible.strip()), 
                            '*****', 
                            contenido, flags=re.IGNORECASE)
    dict_contenidos[documento] = contenido

df_anonimizado_numeros = pl.DataFrame({'documento':list(dict_contenidos.keys()),
                               'contenido_anonimizado_numeros':list(dict_contenidos.values())})

df_muestra = df_muestra.join(df_anonimizado_numeros, on='documento')

df_nombres = generar_nombres()
dict_nombres_extra = {'Banco Internacional': 'Banco ****', 
                     'Banco de Crédito': 'Banco ****',
                     'Banco de la Nación': 'Banco ****',
                     'Bank Trust Company': 'Bank ****',
                     'Banco de la Microempresa': 'Banco ****',
                     'Banco de la Micro': 'Banco ****',
                     'Mi Banco': 'Banco ****',
                     'el-banco-de-la-nacion': 'el Banco ****',
                     'Interbank': '****',
                     'Págalo.pe': '****',
                     'Pequeña Empresa': 'Pequeña~Empresa',
                     'MARCO JURÍDICO': 'MARCO~JURÍDICO',
                     'SERVICIOS GENERALES': '****',
                     'Responsabilidad administrativa': 'Responsabilidad~administrativa',
                     'Marco legal': 'Marco~legal',
                     ' : SEGUNDO': ' : ****',
                     ', Segundo': ', ****',
                     'señor Segundo': 'señor ****',
                     }

dict_nombres_extra2 = {
                     'Responsabilidadadministrativa' : 'Responsabilidad administrativa',
                     'Marcolegal' : 'Marco legal'
                     }


def anonimizar_nombres(palabra_minuscula_excepcion):
    palabra = palabra_minuscula_excepcion['palabras']
    permitir_minuscula = palabra_minuscula_excepcion['permitir_minuscula']
    excepcion = palabra_minuscula_excepcion['excepcion']
    if (( permitir_minuscula or 
        (palabra[0]==palabra[0].upper())
        ) and
        (palabra != excepcion)):
            patron = r'[^\W\d_]{2,}'
            return re.sub(patron, '****', palabra)
    else:
        return palabra

def anonimizar_contenido(contenido_anonimizado_numeros):
    for k in dict_nombres_extra.keys():
        try:
            contenido_anonimizado_numeros = re.sub(k, 
                                                dict_nombres_extra[k], 
                                                contenido_anonimizado_numeros,
                                                flags=re.IGNORECASE)
        except:
            print(k)

    lineas = contenido_anonimizado_numeros.split('\n')
    dict_lineas = {}
    for i, linea in enumerate(lineas):
        lst_palabras = linea.split(' ')
        df_ = pl.DataFrame({'palabras' : lst_palabras,
        'num_palabra' : range(len(lst_palabras))})
        df_ = df_.with_columns(num_linea = pl.lit(i))
        dict_lineas[i] = df_

    df_palabras = (pl.concat(dict_lineas.values())
                    .with_columns(pl.col('palabras')
                                    .map_elements(normalizar_unidecode, return_dtype=pl.Utf8)
                                    .str.to_lowercase()
                                    .alias('palabras_unidecode')
                    )
                    .with_columns(pl.col('palabras_unidecode')
                                    .map_elements(preservar_letras_y_espacios, return_dtype=pl.Utf8)
                                    .alias('palabras_unidecode'),
                                 (pl.col('num_palabra')
                                            .max()
                                            .over(pl.col('num_linea')) == 
                                        pl.col('num_palabra'))
                                        .alias('ult_pal_linea')
                    )
                    .join(df_nombres,
                        left_on = 'palabras_unidecode',
                        right_on = 'nombre')
                    .with_columns(pl.struct(pl.col('palabras'),
                                            pl.col('permitir_minuscula'),
                                            pl.col('excepcion')
                    )
                                    .map_elements(anonimizar_nombres, return_dtype=pl.Utf8)
                                    .alias('palabras_anonimizadas')
                    )                       
    )

    contenido_anonimizado = contenido_anonimizado_numeros
    for palabra, num_palabra, num_linea, ult_pal_linea, palabra_anonimizada in df_palabras.select('palabras','num_palabra','num_linea','ult_pal_linea','palabras_anonimizadas').iter_rows():
        if num_palabra==0:
            if num_linea>0:
                palabra_anonimizada = '\n' + palabra_anonimizada
                palabra = '\n' + palabra
            else:
                palabra_anonimizada = ' ' + palabra_anonimizada
                palabra = ' ' + palabra
        if ult_pal_linea:
            palabra_anonimizada = palabra_anonimizada + '\n'
            palabra = palabra + '\n'
        else:
            palabra_anonimizada = palabra_anonimizada + ' '
            palabra = palabra + ' '
        if len(palabra)>2:
            contenido_anonimizado = contenido_anonimizado.replace(palabra, palabra_anonimizada)
        contenido_anonimizado = contenido_anonimizado.replace('~',' ')
    
    return contenido_anonimizado

def chunkify(lst, n):
    k = math.ceil(len(lst) / n)
    return [lst[i:i+k] for i in range(0, len(lst), k)]

def procesar_chunk(chunk):
    return [anonimizar_contenido(x) for x in chunk]



df_muestra = df_muestra.with_columns(
    pl.col('contenido_anonimizado_numeros')
      .map_elements(anonimizar_contenido, return_dtype=pl.Utf8)
      .alias('contenido_anonimizado'))

df_muestra.write_parquet(f'{RUTA_BASE}/../temporal/datos/m2025_anonimizados_numeros.parquet')

df_muestra = pl.read_parquet(f'{RUTA_BASE}/../temporal/datos/m2025_anonimizados_numeros.parquet')

verificar = False
if verificar:

    dict_muestra = {d:df_muestra.filter(pl.col('documento')==d)['contenido_anonimizado'][0].split('\n') 
     for d in df_muestra['documento'].to_list()}

    dict_muestra = {d:[o for o in dict_muestra[d] if ('arqu' in o.lower())
                       and ('mides' in o.lower())
                       ]
     for d in list(dict_muestra.keys()) }

    dict_muestra = {d:dict_muestra[d]
     for d in list(dict_muestra.keys()) if  len(dict_muestra[d])>0}


    llave = 'DNI'
    sub_dict_resultado = {doc:valores[llave] for doc, valores 
                in zip(resultados.keys(),resultados.values()) if len(valores[llave])>0}
    print(len(sub_dict_resultado), sub_dict_resultado)

    sub_dict_contenido = {doc:[o for o in contenido.split('\n') if llave in o] for (doc,contenido) in zip(df_muestra['documento'].to_list(),df_muestra['contenido'].to_list())}
    sub_dict_contenido = {k:v for (k,v) in zip(sub_dict_contenido.keys(),sub_dict_contenido.values()) if len(v)>0}
    print(len(sub_dict_contenido), sub_dict_contenido)

    len([l for l in contenido.split('\n') for contenido in df_muestra['contenido_anonimizado'].to_list() if '****' in l])

    for documento,contenido in df_muestra.select('documento','contenido_anonimizado').iter_rows():
        for linea in contenido.split('\n'):
            if '****' in linea:
                print(f'{documento} | {linea}')

    doc = '3f2a1059-f27b-4b47-b405-7dbe57f758bc'
    contenido = df_muestra.filter(pl.col('documento')==doc)['contenido'][0]
    contenido_anonimizado = df_muestra.filter(pl.col('documento')==doc)['contenido_anonimizado'][0]
    contenido_anonimizado_numeros = df_muestra.filter(pl.col('documento')==doc)['contenido_anonimizado_numeros'][0]
    lst_contenido = contenido.split('\n')
    lst_contenido_anonimizado = contenido_anonimizado.split('\n')
    lst_contenido_anonimizado_numeros = contenido_anonimizado_numeros.split('\n')
    lst_contenido_dif1 = [o for o in lst_contenido if o not in lst_contenido_anonimizado]
    lst_contenido_dif2 = [o for o in lst_contenido_anonimizado if o not in lst_contenido]
    for i in range(max([len(lst_contenido_dif1),len(lst_contenido_dif2)])):
        try:
            print(f'o{i}...',lst_contenido_dif1[i])
        except:
            pass
        try:
            print(f'n{i}...',lst_contenido_dif2[i])
        except:
            pass

    [o for o in contenido_anonimizado_numeros.split('\n') if 'legal' in o]

    lst_contenido_anonimizado =  df_muestra['contenido_anonimizado'].to_list()
    df_bow = crear_bow(df_muestra, 'contenido_anonimizado', 'a')



    df_tok = df_muestra.with_columns(
        pl.col("contenido_anonimizado")
        .str.replace_all(re.escape('****'),'ANONIMI')
        .str.replace_all(re.escape('\n'),' ')
        .alias("contenido_anonimizado")
    ).with_columns(
        pl.col("contenido_anonimizado")
        .str.replace_all(r"[^\w\s]", "")
        .str.split(" ")
        .alias("tokens")
    )

    df_bigrams = df_tok.with_columns(
        pl.col("tokens")
        .list.eval(
            pl.when(pl.element().shift(-1).is_not_null())
                .then(
                    pl.concat_str(
                        [pl.element(), pl.element().shift(-1)],
                        separator=" "
                    )
                )
                .otherwise(None)
        )
        .list.drop_nulls()
        .alias("bigrams")
    )

    def tiene_anom(lst):
        return [b for b in lst if ('ANONIM' in b) and
                (b.replace('ANONIMI','').lower()!=b.replace('ANONIMI',''))]
    df_bigrams = df_bigrams.select('bigrams').with_columns(
        pl.col('bigrams').map_elements(tiene_anom,return_dtype=pl.List(pl.Utf8)).alias('bigrams2')
    )

    anonimos = [s for sublist in df_bigrams['bigrams2'].to_list() for s in sublist]

    df_b = pl.DataFrame({'big':anonimos})
    df_b['big'].value_counts().sort('count',descending=True).write_csv(f'{RUTA_BASE}/limpieza/bigramas.txt')


