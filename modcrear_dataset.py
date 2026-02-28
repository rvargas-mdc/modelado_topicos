import polars as pl
def crear_revision(df, columna_con_nulos, columna_origen, limitar_a=None):
    archivo_revision = f'./datos/limpieza/revision_{columna_con_nulos}.txt'
    with open(archivo_revision, 'w', encoding='Utf-8') as f:
        for doc in df.filter(pl.col(columna_con_nulos).is_null()).iter_rows():
            documento = doc[2]
            df_ = df.filter(pl.col('documento')==documento)
            contenido = df_[columna_origen][0]
            if limitar_a:
                lineas = contenido[:limitar_a]
            else:
                lineas = contenido
            f.write(f'{documento}' + '-'*30 )
            f.writelines(lineas)
            f.write('='*30)
        f.close()


def actualizar_valores_nulos(df, columna_con_nulos):
    archivo_revision = f'./datos/limpieza/revision_{columna_con_nulos}_corregido.txt'
    with open(archivo_revision, 'r', encoding='Utf-8') as f:
        documentos = f.read()
    lst_documentos = [d for d in documentos.split('='*30) if '-'*30 in d]

    lst_documentos = [(texto.split('-'*30)[0],texto.split('-'*30)[1]) 
                    for texto 
                    in lst_documentos 
                    if len(texto)>1]

    columna_temporal = f'{columna_con_nulos}_temporal'
    df_fix = pl.DataFrame(
        {'documento':[d.replace('\n','') for d,c in lst_documentos],
        columna_temporal:[c for d,c in lst_documentos]
        }
    )

    df = (
        df
        .join(df_fix, on="documento", how="left")
        .with_columns(
            pl.coalesce([columna_con_nulos, columna_temporal]).alias(columna_con_nulos)
        )
        .drop(columna_temporal)
    )
    print(f'Sin {columna_con_nulos}:', df.filter(pl.col(columna_con_nulos).is_null()).shape[0])
    return df