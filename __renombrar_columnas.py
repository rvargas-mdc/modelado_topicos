for o,d in {'denuncia_anonimizada':'antecedentes_anonimizados','denuncia_dictamen_anonimizada':'antecedentes_dictamen_anonimizados','denuncia_dictamen_anonimizado':'antecedentes_dictamen_anonimizados'}.items():
    df_lda = df_lda.with_columns(pl.col('tipo_df').str.replace_all(o,d).alias('tipo_df'),
                                 pl.col('columna').str.replace_all(o,d).alias('columna') )
    df_nmf  = df_nmf.with_columns(pl.col('tipo_df').str.replace_all(o,d).alias('tipo_df'),
                                 pl.col('columna').str.replace_all(o,d).alias('columna'))
    df_bert = df_bert.with_columns(pl.col('tipo_df').str.replace_all(o,d).alias('tipo_df').alias('tipo_df'),
                                 pl.col('columna').str.replace_all(o,d).alias('columna'))

    df_lda.write_parquet( f'{RUTA_BASE}/ejecucion/lda.parquet')
    df_nmf.write_parquet( f'{RUTA_BASE}/ejecucion/nmf.parquet')
    df_bert.write_parquet( f'{RUTA_BASE}/ejecucion/bert.parquet')