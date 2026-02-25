import os
import polars as pl
from openai import OpenAI
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from modulos import *

# 9.52

OPEN_ROUTER_KEY = os.getenv('OPEN_ROUTER_KEY')

RUTA_BASE = str(Path(__file__).resolve().parent)

lst_llaves, dfs, dfsw = conjuntos_de_datos()
df = dfs['todos-denuncia_anonimizada']

modelos = ['openai/gpt-5-mini',
           'google/gemini-2.5-flash-lite',
           'anthropic/claude-3-haiku',
           'deepseek/deepseek-v3.2-speciale'
           ]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_KEY,
)

instruccion_sistema ="""
Eres un asistente legal inteligente. Tu objetivo es resumir las denunciadas 
en una resolución de Indecopi. Reglas: Extrae solo el hecho denunciado. 
Usa como máximo 10 palabras por denuncia. 
OMITE: Nombres, Fechas, Montos, Números de operación. 

Salida: Texto plano separado por 'Enter'. 

Ejemplos de cómo debes procesar la información: 

    Entrada 1: “El señor Juan Pérez indica que el 12 de enero el banco le cobró 500 soles por una membresía que él nunca solicitó”. 
    Salida esperada 1: “Cobro de membresía no solicitada por el usuario.” 

    Entrada 2: “La denunciante afirma que el cajero automático no le dispensó el dinero pero sí se descontó de su cuenta de ahorros.” 
    Salida esperada 2:” Cajero no dispensó efectivo pero descontó saldo” 
    
    Entrada 3: “La señora López indica que el banco le cobró 300 soles por la membresía de su tarjeta de crédito y adicionalmente le otorgaron un crédito falsificando su firma”. 
    Salida esperada 3: “Cobro de membresía no solicitada por el usuario. 
                        Falsificación de firma en crédito no solicitado" 
                        
Si el texto no revela el tema denunciado, indica "Hechos no revelados explícitamente"
    """

def obtener_denuncias( modelo, denuncia):
    completion = client.chat.completions.create(

                    model=modelo, # You can change this to any supported model
                    messages=[
                        {"role":"system",
                        "content":instruccion_sistema},
                        {
                            "role": "user",
                            "content": denuncia
                        }
                    ],
                    temperature=0.1,
                    top_p=0.9
                )
    respuesta = completion.choices[0].message.content
    return respuesta

def procesar_documento(documento, modelo):
    archivo_resumido = f"{RUTA_BASE}/ejecucion/{modelo}_{documento}.txt"
    if os.path.exists(archivo_resumido):
        return f"Previamente procesado {documento}"

    denuncia = df.filter(pl.col('documento')==documento)['denuncia_anonimizada'][0]
    denuncia_resumida = obtener_denuncias(modelo, denuncia)

    with open(archivo_resumido, "w", encoding="utf-8") as f:
        f.write(denuncia_resumida)

    archivos = glob(f"{RUTA_BASE}/ejecucion/{modelo}*")
    n = len(archivos) + 1
    return f"{n}: Procesado {documento}"

lst_documentos = sorted(df['documento'].to_list())

# Ejecutar en paralelo
for modelo in modelos:
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(procesar_documento, doc, modelo) for doc in lst_documentos]

        for future in as_completed(futures):
            print(future.result())



dict_denuncias = {}
for modelo in modelos:
    ruta_modelo = modelo.split('/')[0]
    nombre_modelo = modelo.split('/')[1]
    ruta_archivo = RUTA_BASE + '/ejecucion/' + modelo
    archivos = glob( f'{ruta_archivo}*')
    for archivo in archivos:
        documento_txt = archivo.split('\\')[-1]
        documento = documento_txt.replace(f'{nombre_modelo}_','')[:-4]
        with open(archivo, 'r', encoding='Utf-8') as f:
            contenido = f.read()
            f.close()
            oraciones = contenido.split('\n')
            for i, oracion in enumerate(oraciones):
                dict_denuncias[f'{documento}_{i+1}_{nombre_modelo}'] = oracion

df_llm = pl.DataFrame({'documento' : [k.split('_')[0] for k in dict_denuncias.keys()],
                   'num_denuncia' : [k.split('_')[1] for k in dict_denuncias.keys()],
                   'modelo' : [k.split('_')[2] for k in dict_denuncias.keys()],
                   'denuncia':[str(v) for v in dict_denuncias.values()]})

df_llm_oraciones = df.join(df_llm, on='documento',how='left')

df_llm_oraciones = df_llm_oraciones.select(['llave',
 'documento',
 'area',
 'lugar',
 'zona',
 'denuncia_anonimizada',
 'num_denuncia',
 'modelo',
 'denuncia'])
df_llm_oraciones.pivot(index='documento',on='modelo',values='denuncia',aggregate_function='len' ).write_excel('enuncias_llm.xlsx')
df_llm_oraciones.write_parquet( f'{RUTA_BASE}/datos/denuncias_llm.parquet')


df_llm_oraciones = pl.read_parquet(f'{RUTA_BASE}/datos/denuncias_llm.parquet')
lst_modelos = df_llm_oraciones.select('modelo').unique()['modelo'].to_list()
dict_df_llm = {modelo: df_llm_oraciones.filter(pl.col('modelo')==modelo) for modelo in lst_modelos}



print('Fin')