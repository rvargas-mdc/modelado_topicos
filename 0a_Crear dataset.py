import os
import requests
import polars as pl
from glob import glob
import ast
from pathlib import Path
from modulos import *
RUTA_BASE = str(Path(__file__).resolve().parent)

def prueba():
    contenido = df['contenido'][0]
    print(contenido)

# 1. Leer los archivos
archivos = glob(f'{RUTA_BASE}/../temporal/datos/crudos/*.txt')
def leer_archivo(ruta):
    with open(ruta, encoding="utf-8") as f:
        contenido = f.read()
        f.close()
    return contenido

df = pl.DataFrame({'archivo':archivos})
df = df.with_columns(
    pl.col("archivo")
    .map_elements(leer_archivo, return_dtype=pl.String)
    .alias("contenido")
)

# 2. Crear columna documento
def nombrar_documento(ruta):
    return ruta.split('\\')[-1][:-4]
df = df.with_columns(
    pl.col("archivo")
    .map_elements(nombrar_documento, return_dtype=pl.String)
    .alias("documento")
)

# 3. Extarer cabecera
def extraer_cabecera_con_fecha(contenido):
    lst_palabras = ['antecedentes\n', 
                    'antecedentes:\n',
                    'antecedentes. -\n',
                    '\n1. antecedentes'
                    'antecedente\n',
                    'antecedente:\n',
                    'hechos\n',
                    'hechos:\n'
                    ]
    antecedentes = ''
    contenido = contenido.lower()
    for palabra in lst_palabras:
        antecedentes = contenido.split(palabra)
        if len(antecedentes)==2:
            antecedentes = antecedentes[0]
            try:
                antecedentes = [l for l in antecedentes.split('\n') if len(l.strip())>5]
                antecedentes = '\n'.join(antecedentes)
            except:
                pass
            break
        else:
            antecedentes = None
    if antecedentes:
        lst_palabras = ['sumilla', 
                        'resumen',
                        '\nsanción ',
                        '\nsancion ',
                        '\nsanción\n',
                        '\nsanción:\n',
                        '\nsanciones:',
                        '\nsanciones\n',
                        '\nsanciones ',
                        '\nsanciónes ',
                        '\nsanción: ',
                        '\nsancion: ',
                        '\nsanción :',
                        '2025\n1. ',
                        '\nresuelto:\n',
                        '\nsanciones a ',
                        '\nsanciones ix',
                        '\nsanciones-',
                        '\nsanciones -',
                        '\ncomisión ha resuelto:',
                        '\nen el procedimiento administrativo',
                        '\nse desestima el pedido',
                        '\nse declara ',
                        '\ncuestión previa:',
                        ]
        lst_antecedentes2 = [antecedentes.split(palabra)[0]
                                for palabra in lst_palabras]
        len_antecedentes2 = [len(a) for a in lst_antecedentes2]
        min_len = min(len_antecedentes2)
        antecedentes2 = lst_antecedentes2[len_antecedentes2.index(min_len)]
        lst_fechas = [l for l in antecedentes.split('\n') 
                    if ('202' in l) 
                    & ('expediente' not in l) 
                    & ('resoluci' not in l)
                    & (l[-4:-1]=='202')]
        if (len(lst_fechas)==1) & (len(antecedentes2)<len(antecedentes)):
            antecedentes = antecedentes2 + '\n' + lst_fechas[0]
        else:
            antecedentes = antecedentes2
    return antecedentes

df = df.with_columns(
    pl.col("contenido")
    .map_elements(extraer_cabecera_con_fecha, return_dtype=pl.String)
    .alias("cabecera_con_fecha")
)

print('Archivos', len(archivos))
print('Filas:', df.shape[0])
print('Sin cabecera con fecha:', df.filter(pl.col('cabecera_con_fecha').is_null()).shape[0])

df = (df.with_columns(pl.col('cabecera_con_fecha').str.len_chars().alias('largo'))
        .with_columns(pl.when((pl.col('largo')>900) |
                                ((pl.col('cabecera_con_fecha').str.contains('\nsanci'))) &
                                ~(pl.col('cabecera_con_fecha').str.contains('\nsancionador')
                                )
                                ) 
                        .then(pl.lit(None))
                        .otherwise(pl.lit('Longitud adecuada'))
                        .alias('control longitud')))
    
df2 = df.filter(pl.col('cabecera_con_fecha').is_null())

# 3.1 Revisar manualmente qué documentos no tienen el campo cabecera lleno


df = df.with_columns(pl.when((pl.col('control longitud').is_null())|
                                (pl.col('cabecera_con_fecha').is_null()) |
                                ((pl.col('cabecera_con_fecha').str.contains('\nsanci'))) &
                                ~(pl.col('cabecera_con_fecha').str.contains('\nsancionador')
                                )
                                )
                        .then(None)
                        .otherwise(pl.col('cabecera_con_fecha'))
                        .alias('cabecera_con_fecha'))

crear_revision(columna_con_nulos='cabecera_con_fecha', 
            columna_origen='contenido', 
            limitar_a=2000)

# 3.2 Después de revisar manualmente el archivo 'revision.txt'
#     actualizar las cabeceras 



df = actualizar_valores_nulos(df, 'cabecera_con_fecha')

df = df.with_columns(cabecera_con_fecha=pl.col('cabecera_con_fecha').str.to_lowercase())

df = df.select([c for c in df.columns if c not in ['largo', 'control longitud']])
# 4. Identificar el área
def identificar_area(cabecera):
    cabecera = cabecera.lower()
    if 'tribunal' in cabecera:
        return 'Tribunal'
    if 'comisi' in cabecera:
        return 'Comisión'
    if 'resolutivo' in cabecera:
        return 'ORPS'
df = df.with_columns(
    pl.col("cabecera_con_fecha")
    .map_elements(identificar_area, return_dtype=pl.String)
    .alias("area")
)
print('Sin área:', df.filter(pl.col('area').is_null()).shape[0])

print(df.shape[0])
crear_revision(columna_con_nulos='area', 
                columna_origen='contenido', 
                limitar_a=2000)
   
df = actualizar_valores_nulos(df, 'area')
print(df.shape[0])


# 5. Identificar la oficina ---
def identificar_oficina(cabecera):
    dict_sedes = {'arequipa':'Arequipa',
                'huancayo':'Huancayo',
                'piura':'Piura',
                'trujillo':'Trujillo',
                'la libertad':'Trujillo',
                'chiclayo':'Chiclayo',
                'lambayeque':'Chiclayo',
                'cajamarca':'Cajamarca',
                'puno':'Puno',
                'cusco':'Cusco',
                'chimbote':'Chimbote',
                'tacna':'Tacna',
                'tarapoto':'Tarapoto',
                'san martín':'Tarapoto',
                'tribunal':'Lima',
                'huancavelica':'Huancavelica',
                'ica':'Ica',
                'huaraz':'Huaraz',
                'tacna':'Tacna',
                'lima sur':'Sede Lima Sur',
                'lima norte':'Sede Lima Norte',
                'central lima sur':'Sede Lima Sur',
                'los olivos':'Sede Lima Norte',
                'sede central':'Sede Central',
                'lima':'Lima',}
    cabecera = cabecera.lower().strip()
    for llave_sede, nombre_sede in zip(dict_sedes.keys(), dict_sedes.values()):
        if ((f'indecopi de {llave_sede}' in cabecera) |
            (f'indecopi {llave_sede}' in cabecera) | 
            (f'indecopi - {llave_sede}' in cabecera) | 
            (f'oficina regional de {llave_sede}' in cabecera) |
            (f'sede {llave_sede}' in cabecera)):
            return nombre_sede
    for llave_sede, nombre_sede in zip(dict_sedes.keys(), dict_sedes.values()):
        if llave_sede in cabecera:
            return nombre_sede
    ult_linea = cabecera.split('\n')[-1]
    if ',' in ult_linea:
        lugar = ult_linea.split(',')[0]
    else:
        lugar = ult_linea.split(' ')[0]
    try:
        return dict_sedes[lugar]
    except:
        return None
df = df.with_columns(
    pl.col("cabecera_con_fecha")
    .map_elements(identificar_oficina, return_dtype=pl.String)
    .alias("lugar")
)

df = df.with_columns(pl.when(pl.col('lugar')=='Lima')
                       .then(pl.lit('Sede Central'))
                       .otherwise(pl.col('lugar'))
                       .alias('lugar'))

print('Sin lugar:', df.filter(pl.col('lugar').is_null()).shape[0])

print('Resoluciones: ',df.shape[0])


# 6. Identificar denunciante, denunciado y actividad
def identificar_denunciante_y_otros(cabecera):
    cabecera = cabecera.lower()
    try:
        ante_ado_actividad = cabecera.split('denuncian')[1]
        denunciante = ante_ado_actividad.split('denunciad')[0]
        ado_mat_actividad = ante_ado_actividad.split('denunciad')[1]
    except:
        return None
    
    try:
        denunciante = denunciante.split(':')[1].split('\n')
        denunciante = [d.split('(')[0].strip() for d in denunciante]
        denunciante = [d for d in denunciante if len(d)>1]
        denunciante = ';'.join(denunciante)
    except:
        denunciante = None

    try:
        denunciado = ado_mat_actividad.split('materia')[0]
        denunciado = denunciado.split(':')[1].split('\n')
        denunciado = [d.split('(')[0].strip() for d in denunciado]
        denunciado = [d for d in denunciado if len(d)>1]
        denunciado = ';'.join(denunciado)
    except:
        denunciado = None

    try:
        materia = ado_mat_actividad.split('materia')[1]
        materia = materia.split('actividad')[0]
        materia = materia.split(':')[1].split('\n')
        materia = [d.split('(')[0].strip() for d in materia]
        materia = [d for d in materia if len(d)>1]
        materia = ';'.join(materia)
    except:
        materia = None


    try:       
        actividad = ado_mat_actividad.split('actividad')[1]
        actividad = actividad.split(':')[1].split('\n')
        actividad = [d.split('(')[0].strip() for d in actividad]
        actividad = [d for d in actividad if len(d)>1]
        actividad = ';'.join(actividad[:-1])
    except:
        actividad = None

    return str({'denunciante':denunciante, 
                'denunciado':denunciado, 
                'materia':materia,
                'actividad':actividad})
    
# 6.1 Revisión manual de datos nulos
df = df.with_columns(
        pl.col("cabecera_con_fecha")
        .map_elements(identificar_denunciante_y_otros, return_dtype=pl.String)
        .alias("ante_ado_vidad")
    ).with_columns(
        pl.when((pl.col('ante_ado_vidad').is_null()) &
                ~(pl.col('cabecera_con_fecha').str.contains('denunciante'))
                )
          .then(pl.lit('Sin denunciante'))
          .otherwise(pl.lit(''))
          .alias('observacion_ado')
    ).with_columns(
        pl.when((pl.col('ante_ado_vidad').is_null()) &
                ~(pl.col('cabecera_con_fecha').str.contains('denunciad'))
                )
          .then(pl.concat_str(pl.col('observacion_ado'), pl.lit(' - '), pl.lit('Sin denunciado')))
          .otherwise(pl.col('observacion_ado'))
          .alias('observacion_ado')
    ).with_columns(
        pl.when((pl.col('ante_ado_vidad').is_null()) &
                ~(pl.col('cabecera_con_fecha').str.contains('actividad'))
                )
          .then(pl.concat_str(pl.col('observacion_ado'), pl.lit(' - '), pl.lit('Sin actividad')))
          .otherwise(pl.col('observacion_ado'))
          .alias('observacion_ado')
    ).with_columns(
        pl.when((pl.col('observacion_ado')=='') &
                (pl.col('ante_ado_vidad').is_null())
                )
          .then(None)
          .otherwise(pl.col('observacion_ado'))
          .alias('observacion_ado')
    )


df = df.filter(~pl.col('cabecera_con_fecha').str.contains('\nqueja'))

df = df.with_columns(pl.when(pl.col('observacion_ado')=="")
                       .then(pl.lit(None))
                       .otherwise(pl.col('observacion_ado'))
                       .alias('observacion_ado'))

print('Sin ante_ado_vidad:', df.filter(pl.col('ante_ado_vidad').is_null()).shape[0])
print('Sin ante_ado_vidad y sin causa:', df.filter((pl.col('ante_ado_vidad').is_null()) & 
                                                   (pl.col('observacion_ado').is_null() )).shape[0])

df = df.filter(~(pl.col('observacion_ado').is_not_null() &
                             pl.col('ante_ado_vidad').is_null()))

print(df.shape[0])
crear_revision(columna_con_nulos='ante_ado_vidad', 
            columna_origen='cabecera_con_fecha', 
            limitar_a=None)

# 6.2 Actualización de datos corregidos manualmente
df = actualizar_valores_nulos(df, 'ante_ado_vidad')
print(df.shape[0])

# 6.3 Expandir en cuatro columnas independientes
def actualiza_ado_vidad_parte(par_valores): 
    valores = par_valores['ante_ado_vidad']
    llave_ = par_valores['literal']
    try:
        return ast.literal_eval(valores)[llave_]
    except:
        return None

for llave in ['denunciante', 'denunciado', 'materia', 'actividad']:
    print(llave)
    df = (df.with_columns(
            pl.struct(pl.col("ante_ado_vidad"),pl.lit(llave)).alias(llave)
        )
            .with_columns(
            pl.col(llave)
            .map_elements(actualiza_ado_vidad_parte, return_dtype=pl.String)
            .alias(llave)
        )
    )

# 6.4 Eliminar columnas temporales
df = df.select([c for c in df.columns if c not in [
                'ante_ado_vidad',
                'observacion_ado']])


# 7. Conservar denuncias a bancos que no sean del Tribunal 
df = df.filter(
        pl.col('denunciante').is_not_null() &
        (pl.col('denunciado').str.contains('banco') |
        pl.col('denunciado').str.contains('scotiabank'))
    )

print('Resoluciones con Tribunal', df.shape[0])
df = df.filter(
        (pl.col('area')!='Tribunal') 
    )

print('Resoluciones sin tribunal', df.shape[0])


df = df.with_columns(pl.when(pl.col('lugar').str.contains('Sede'))
                            .then(pl.lit('Lima'))
                            .otherwise(pl.lit('Provincias'))
                            .alias('zona')      ,
                     pl.col("contenido")
                       .str.replace_all("\n", "~")
                       .str.replace_all(r"\s+", " ")
                       .str.replace_all(" ~", "~")
                       .str.replace_all("~ ", "~")
                       .str.replace_all("~", "\n")
                       .alias('contenido')
                    )

df.write_parquet(f'{RUTA_BASE}/../temporal/datos/resoluciones_bancos_2025.parquet')

df = pl.read_parquet(f'{RUTA_BASE}/../temporal/datos/resoluciones_bancos_2025.parquet')
df = (df.with_columns(
    # Conteo de palabras
    num_palabras=
        pl.col("contenido")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.split(" ")
        .list.len(),
    # Conteo de párrafos
    num_parrafos=
        pl.col("contenido")
        .str.split("\n")
        .list.len()
            )
    .with_columns(
        (pl.col('num_palabras')/
         pl.col('num_parrafos')).alias('pal_por_parrafo')
    )
    )

print(
df.pivot(on='area',
                      index='zona',
                      values='documento',
                      aggregate_function='len')
)

print(
df.pivot(on='area',
                      index='zona',
                      values='num_palabras',
                      aggregate_function='mean')
)

print(
df.pivot(on='area',
                      index='zona',
                      values='num_parrafos',
                      aggregate_function='mean')
)

print(
df.pivot(on='area',
                      index='zona',
                      values='pal_por_parrafo',
                      aggregate_function='mean')
)

