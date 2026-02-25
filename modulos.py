import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from unidecode import unidecode
import re
from pathlib import Path

RUTA_BASE = str(Path(__file__).resolve().parent)


STOPWORDS_ES = [
    # pronombres personales
    "yo","me","mí","conmigo",
    "tú","te","ti","contigo",
    "él","ella","ello","lo","la","le","se","sí","consigo",
    "nosotros","nosotras","nos","nuestro","nuestra","nuestros","nuestras",
    "vosotros","vosotras","os","vuestro","vuestra","vuestros","vuestras",
    "ellos","ellas","los","las","les",

    # pronombres posesivos
    "mío","mía","míos","mías",
    "tuyo","tuya","tuyos","tuyas",
    "suyo","suya","suyos","suyas",

    # pronombres / adjetivos demostrativos
    "este","esta","estos","estas",
    "ese","esa","esos","esas",
    "aquel","aquella","aquellos","aquellas",
    "tal", "tales"

    # pronombres relativos
    "que","quien","quienes","cual","cuales","cuyo","cuya","cuyos","cuyas",
    "donde","cuando","cuanto","cuanta","cuantos","cuantas",

    # pronombres indefinidos
    "algo","alguien","nadie","nada",
    "uno","una","unos","unas",
    "otro","otra","otros","otras",
    "alguno","alguna","algunos","algunas",
    "ninguno","ninguna","ningunos","ningunas",
    "mucho","mucha","muchos","muchas",
    "poco","poca","pocos","pocas",
    "todo","toda","todos","todas"

    # preposiciones
    "a","ante","bajo","cabe","con","contra",
    "de","desde","durante","en","entre",
    "hacia","hasta","mediante","para","por",
    "según","sin","sobre","tras","versus","vía"

    # otros
    "ademas", "asi", "tanto", 
    "mas", "pero", "como",
    "via","aun","que",
    "iii", "vii", "viii",
    "enero", "febrero", "marzo", "abril",
    "mayo", "junio", "julio", "agosto",
    "setiembre", "octubre", "noviembre", "diciembre",
    "uno", "dos", "tres", "cuatro", 
    "cinco", "seis", "siete", "ocho", 
    "nueve", "diez"

]

STOPWORDS_UNICODE = [unidecode(p) for p in STOPWORDS_ES]

NO_STOPWORDS = ['cargo',
 'coercitivo',
 'credito',
 'cuenta',
 'embargo',
 'fondo',
 'gasto',
 'importe',
 'interes',
 'operacion',
 'pagar',
 'pago',
 'reclamar',
 'reclamo',
 'tarjeta',
 'tasa',
 'transaccion',
 'transparencia']

STOPWORDS_DOMINIO = [
 'acceso', 'accion', 'acordar', 'acreditar', 'acreditir',
 'actividad', 'acto', 'actuacion', 'actuado', 'actuar',
 'acuerdo', 'adelante', 'ademas', 'adjunto', 'administrar',
 'administrativo', 'admitir', 'adoptar', 'adquirir', 'advertir',
 'afectacion', 'afectado', 'afectar', 'agotar', 'alcance',
 'allanamiento', 'amonestacion', 'analisis', 'analizar', 'ano',
 'antecedente', 'anterior', 'antes', 'apelacion', 'apercibimiento',
 'aplicable', 'aplicacion', 'aplicar', 'apreciar', 'aprobar',
 'arbitral', 'articular', 'articulo', 'asimismo', 'asumida',
 'asumir', 'atencion', 'atender', 'atenuante', 'automaticamente',
 'autoridad', 'bancario', 'banco', 'bien', 'brindar',
 'buscar', 'caber', 'cada', 'calidad', 'caracteristica',
 'caso', 'cautelar', 'central', 'circunstancia', 'citado',
 'civil', 'clausula', 'coactivo', 'cobranza', 'codigo',
 'colegiado', 'comercial', 'comision', 'comisionado', 'competencia',
 'competente', 'complementario', 'concluir', 'concreto', 'condicion',
 'conducta', 'conforme', 'conformidad', 'conjunto', 'conocimiento',
 'consecuencia', 'consejo', 'consentido', 'consentimiento', 'consideracion',
 'considerar', 'consistir', 'constancia', 'constituir', 'consumidor',
 'consumo', 'contado', 'contar', 'contencioso', 'contenido',
 'contrario', 'contratar', 'contribuir', 'controversia', 'copia',
 'correctiva', 'correctivo', 'corresponder', 'correspondiente', 'costa',
 'costo', 'criterio', 'cualquiera', 'cuarto', 'cuestion',
 'cumplimiento', 'cumplir', 'dar', 'deber', 'decir',
 'decision', 'declaracion', 'declarar', 'decreto', 'defensa',
 'dejar', 'dentro', 'denuncia', 'denunciado', 'denunciante',
 'denunciar', 'derecho', 'descargo', 'determinado', 'determinar',
 'devolver', 'dia', 'diario', 'dicho', 'dictar',
 'direccion', 'directiva', 'directo', 'disponer', 'disposicion',
 'dispuesto', 'distinto', 'documento', 'duplicar', 'efectivo',
 'efecto', 'efectuar', 'ejecucion', 'emitir', 'empresa',
 'encontrar', 'entidad', 'entregar', 'envase', 'escrito',
 'especial', 'establecer', 'establecido', 'establecimiento', 'estado',
 'estar', 'evaluar', 'evitar', 'existencia', 'existir',
 'expediente', 'expresamente', 'expresar', 'expreso', 'expuesto',
 'extremo', 'factor', 'facultad', 'facultado', 'falta',
 'favor', 'fecha', 'fin', 'final', 'finalidad',
 'financiero', 'firme', 'forma', 'formulado', 'formular',
 'formule', 'funcion', 'fundamento', 'fundar', 'futuro',
 'general', 'generar', 'gerencia', 'graduacion', 'gratuito',
 'haber', 'habil', 'hacer', 'hecho', 'identificar',
 'idoneidad', 'implicar', 'imponer', 'impongan', 'impositiva',
 'impositivo', 'impugnacion', 'impugnar', 'impugnativo', 'imputacion',
 'incluir', 'incumplimiento', 'incurrir', 'indebido', 'indecopi',
 'indicar', 'informacion', 'informar', 'infraccion', 'infractor',
 'infractora', 'iniciar', 'inmediato', 'inscripcion', 'instancia',
 'instituto', 'intelectual', 'intermediacion', 'interponer', 'interposicion',
 'interpuesto', 'intervencion', 'jefa', 'jefatura', 'judicial',
 'juridico', 'lado', 'lapso', 'legal', 'legislativo',
 'legitimidad', 'ley', 'limite', 'literal', 'llevar',
 'lpag', 'mandato', 'manera', 'marco', 'materia',
 'maximo', 'mayor', 'medida', 'medio', 'menor',
 'mercado', 'merito', 'mes', 'ministro', 'mismo',
 'modificado', 'modificar', 'momento', 'monetario', 'monto',
 'motivo', 'multa', 'nacional', 'naturaleza', 'necesario',
 'norma', 'normativa', 'notificacion', 'notificado', 'nuevamente',
 'nuevo', 'numeral', 'objeto', 'obligacion', 'obligado',
 'obrar', 'obstante', 'ocasionado', 'oficial', 'oficina',
 'oficio', 'orden', 'ordenar', 'organo', 'orientar',
 'otorgar', 'pagina', 'parrafo', 'parte', 'particular',
 'partir', 'patrimonial', 'pedir', 'perder', 'perdida',
 'perjuicio', 'permitir', 'persistir', 'persona', 'pertinente',
 'peru', 'peruano', 'plantear', 'plazo', 'poder',
 'poner', 'posible', 'potestad', 'precedente', 'precisar',
 'presentacion', 'presentado', 'presentar', 'presente', 'presidenta',
 'prestacion', 'presunto', 'pretension', 'previo', 'previsto',
 'primero', 'principio', 'probar', 'probatorio', 'procedencia',
 'proceder', 'procedimiento', 'procesal', 'proceso', 'producir',
 'producto', 'pronunciamiento', 'pronunciar', 'propiedad', 'propio',
 'prorrogable', 'proteccion', 'proveedor', 'publicar', 'publico',
 'puder', 'pues', 'puesto', 'punto', 'quedar',
 'quince', 'quinto', 'realizar', 'recibir', 'reconocer',
 'reconocido', 'recurso', 'referido', 'referir', 'registrado',
 'registro', 'reglamento', 'regular', 'relacion', 'reparador',
 'requerimiento', 'requerir', 'requisito', 'resarcir', 'resolucion',
 'resolutivo', 'resolver', 'respectivo', 'respecto', 'responsabilidad',
 'responsable', 'resultar', 'revertir', 'revision', 'salud',
 'salvo', 'sancion', 'sancionador', 'sancionar', 'secretaria',
 'sede', 'seguir', 'segundo', 'seguridad', 'senalar',
 'senor', 'senora', 'sentido', 'ser', 'servicio',
 'sexto', 'siempre', 'siguiente', 'similar', 'sino',
 'sistema', 'situacion', 'solicitar', 'solicitud', 'solo',
 'sub', 'sucesivamente', 'sumar', 'sumarisimo', 'supremo',
 'supuesto', 'tambien', 'tecnica', 'telefono', 'tener',
 'tercero', 'termino', 'texto', 'tipo', 'titulo',
 'tomar', 'total', 'tramitacion', 'tramite', 'tratar',
 'traves', 'tributario', 'tuo', 'uit', 'ultimo',
 'unicamente', 'unico', 'unidad', 'utilizar', 'vencer',
 'vencido', 'vencimiento', 'ver', 'verificar', 'vez',
 'vida', 'vigencia', 'vigente', 'virtud', 'web'
]



def crear_bow(df_, columna, nombre_df):
    texts = df_[columna].to_list()
    if isinstance(texts[0],list):
        texts = ' '.join(texts)

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=STOPWORDS_UNICODE  # optional, remove if you want all tokens
    )

    bow_matrix = vectorizer.fit_transform(texts)

    arr_palabras = vectorizer.get_feature_names_out()

    df_bow = pl.DataFrame(
        bow_matrix.T.toarray()
    )

    columnas_doc = [f"doc_{i:03d}" for i in range(bow_matrix.shape[0])]
    df_bow.columns = columnas_doc
    df_bow_u = df_bow
    for col in df_bow.columns:
        df_bow = (df_bow
                   .with_columns(pl.col(col).cast(pl.Int128)))
        df_bow_u = (df_bow_u
                   .with_columns(pl.when(pl.col(col)>0)
                                   .then(pl.lit(1))
                                   .otherwise(pl.lit(0)) 
                                   .alias(col) 
                                   .cast(pl.Int16))
                    )     

    df_bow = (df_bow
                .with_columns(
                    pl.Series("palabra", arr_palabras),
                    pl.sum_horizontal(columnas_doc)
                    .alias('ocurrencias_totales')
                )
                .select('palabra','ocurrencias_totales')
    )
    df_bow_u = (df_bow_u
                .with_columns(
                    pl.Series("palabra", arr_palabras),
                    pl.sum_horizontal(columnas_doc)
                    .alias('documentos'),
                    (pl.sum_horizontal(columnas_doc)/
                    len(columnas_doc))
                    .alias('porc_documentos'),
                    pl.lit(nombre_df).alias('muestra')
                            )
                .select('muestra','palabra','documentos','porc_documentos')
    )
    df_bow_t = df_bow_u.join(df_bow, on='palabra')

    return df_bow_t.sort('porc_documentos',descending=True)

def palabra_en_columna(df, palabra):

    pattern = rf"(^|[^a-zA-ZÁÉÍÓÚáéíóúÑñ]){palabra}([^a-zA-ZÁÉÍÓÚáéíóúÑñ]|$)"
    regex = re.compile(pattern, re.IGNORECASE)

    lst_contenido = df["contenido"].to_list()

    hallazgos = [
        linea
        for c in lst_contenido
        for linea in c.split("\n")
        if regex.search(linea)
    ]

    return list(set(hallazgos))    


def normalizar_unidecode(texto: str) -> str:
    return unidecode(texto)

def normalizar_unidecode_unicos(texto: str) -> str:
    lst_texto = [unidecode(t) for t in texto.split('/n')]
    return ' '.join(lst_texto)

def preservar_letras_y_espacios(texto: str) -> str:
    texto = texto.replace('\n','^').lower()
    texto = texto.replace(' ','~').lower()
    nuevo_texto = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚüïÜÏ~\^]+", "", texto)
    nuevo_texto = nuevo_texto.replace('^','\n')
    nuevo_texto = nuevo_texto.replace('~',' ')
    return nuevo_texto

def extraer_nombres_propios(texto: str) -> list:
    nombres_propios = list(set(re.findall(r'\b[A-Z][a-zA-Z]*\b', texto)))
    nombres_propios = ' '.join([n for n in nombres_propios if (len(n)>1) and n.lower not in STOPWORDS_ES])
    return nombres_propios


def anonimizar_nombres(texto):
    patron = r'[^\W\d_]{2,}'
    return re.sub(patron, '****', texto)

def crear_revision(df,columna_con_nulos, columna_origen, limitar_a=None):
    archivo_revision = f'{RUTA_BASE}/limpieza/revision_{columna_con_nulos}.txt'
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
    archivo_revision = f'{RUTA_BASE}/limpieza/revision_{columna_con_nulos}_corregido.txt'
    with open(archivo_revision, 'r', encoding='Utf-8') as f:
        documentos = f.read()
    lst_documentos = documentos.split('='*30)
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


def conjuntos_de_datos():
    df_total = pl.read_parquet( f'{RUTA_BASE}/datos/muestra_2025_limpia.parquet')
    df_stopwords = pl.read_excel(f'{RUTA_BASE}/limpieza/stopwords.xlsx')

    lst_llaves = df_total.select('llave').unique()['llave'].to_list()
    dfs = {}
    dfsw = {}

    for llave in lst_llaves:
        dfs[llave] = df_total.filter(pl.col('llave')==llave)
        dfsw[llave] = df_stopwords.filter(pl.col('llave')==llave)
    
    return lst_llaves, dfs, dfsw


def conjuntos_de_datos_llm():
    df_llm_oraciones = pl.read_parquet(f'{RUTA_BASE}/datos/denuncias_llm.parquet')
    lst_modelos = df_llm_oraciones.select('modelo').unique()['modelo'].to_list()
    dict_df_llm = {modelo: df_llm_oraciones.filter(pl.col('modelo')==modelo) for modelo in lst_modelos}



    return dict_df_llm