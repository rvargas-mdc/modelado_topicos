# Comparativa de Modelado de Tópicos

Comparativa de modelado de tópicos en un cojunto de resoluciones de Indecopi empleando LDA, NMF, BERTopic y diversos LLM

## Componentes

### Scripts de preparación y limpieza de datos

**0a_Crear dataset.py**: 
- Crea una base de datos de los documentos a partir de archivos de texto creados a partir de los PDF originales.
- Crea la columna cabecera que contiene las cabeceras de los docuemntos.
- Identifica el "área" resolutiva (ORPS, Comisión de Protección al Consumidor, Tribunal de Protección al Consumidor)
- Identifica el lugar dende se encuentra el área resolutiva (Sede Central, Lima Norte, Lima Sur, provincias)
- Identifica al denunciante, al denunciado, la materia sobre la que versa la resolución y la actividad económica. La materia tiene textos como, por ejemplo: procedimiento administrativo sancionador, medida correctiva, graduación de la sanción, costas y costos, falta de idoneidad y solicitud de nulidad.

**0b_Extraer_muestra.py**:
- Genera cinco muestras a partir del conjunto de datos global:


|Muestra|Contenido        |
|-----|------------------|
|todos|Muestra de todos los documentos|
|comisión|Muestra de todas las resoluciones provenientes de comisiones|
|orps|Muestra de todas las resoluciones provenientes de las oficinas de resolución de procesos sumarísimos|
|com-lima|Muestra de todas las resoluciones provenientes de comisiones de la sede Lima|
|orps|Muestra de todas las resoluciones provenientes de la ORPS de Lima| 



**0c_Descubrir_nombres.py**:
- Crea lista de nombres de los denunciantes en un archivo CSV que sirve para retirar de ella palabras como "viuda".

**0d_Anonimizar.py**
- Anonimiza números de documentos nacionales de identidad (DNI), números únicos de contribuyentes tributarios (RUC), números telefónicos, correos, direcciones web, nombres de denunciantes y nombres de denunciados.
- Crea la columna `contenido_anonimizado`

**0e_Extraer_secciones.py**
- Extrae la sección antecedentes y la sección dictamen en cada documento. Crea tres columnas `antecedentes_anonimizado`, `dictamen_anonimizado`,
`antecedentes_dictamen_anonimizado`

**0f_Limpiar_texto.py**
- Convierte en minúsculas, retira *stopwords* y lematiza el contenido de `contenido_anonimizado`, `antecedentes_anonimizado`, `dictamen_anonimizado` y `antecedentes_dictamen_anonimizado` y se genera las columnas `contenido_anonimizado_limpio`, `antecedentes_anonimizado_limpio`, `dictamen_anonimizado_limpio`, `antecedentes_dictamen_anonimizado_limpio`
`contenido_anonimizado_limpio_bert`, `antecedentes_anonimizado_limpio_bert`, `dictamen_anonimizado_limpio_bert` y `antecedentes_dictamen_anonimizado_limpio_bert`

Las columnas terminadas en "bert" elimnan los saltos de línea que no están precedidos de un punto.

**01.LDA.py**
- Varaiables independientes:

|Variable|Valores|
|--------|----------|
|Muestra|Todas las muestras: "todos", "orps", "comisión","com-lima","orps-lima"|
|Columnas| `contenido_anonimizado_limpio`, `antecedentes_anonimizado_limpio`, `dictamen_anonimizado_limpio` y `antecedentes_dictamen_anonimizado_limpio`|
|Número de tópicos|Todos lo números enteros entre 3 y 35 inclusive|

**02.NMF.py**
- Variables independientes:

|Variable|Valores|
|--------|----------|
|Muestra|Todas las muestras: "todos", "orps", "comisión","com-lima","orps-lima"|
|Columnas| `contenido_anonimizado_limpio`, `antecedentes_anonimizado_limpio`, `dictamen_anonimizado_limpio` y `antecedentes_dictamen_anonimizado_limpio`|
|Hiper parámetros|Ver tabla|
|Número de tópicos|Todos lo números enteros entre 3 y 35 inclusive|

Tabla de hiper parámetros:
|Id|Init|Randon_state|Solver|Beta_loss|Alpha_W|Alpha_H|Shuffle|Tol|
|--|----|----|----|----|----|----|----|----|
|1|random |42|cd|frobenius|0.0|0.0|True|0.001|
|2|random|42|mu|kullback-leibler|0.0|0.0|True|0.001|
|3|random|42|cd|frobenius|0.001|0.001|True|0.001|
|4|random|42|mu|kullback-leibler|0.001|0.001|True|0.001|
|5|nndsvda|42|cd|frobenius|0.0|0.0|True|0.001|
|6|nndsvda|42|mu|kullback-leibler|0.0|0.0|True|0.001|
|7|nndsvda|42|cd|frobenius|0.001|0.001|True|0.001|
|8|nndsvda|42|mu|frobenius|0.001|0.001|True|0.001|


- Se genera un modelo para cada compibación de hiperparámetros.


**03.BERTopic.py**
- Variables inependientes:


|Variable|Valores|
|--------|----------|
|Muestra|Todas las muestras: "todos", "orps", "comisión","com-lima","orps-lima"|
|Columnas| `contenido_anonimizado_limpio`, `antecedentes_anonimizado_limpio`, `dictamen_anonimizado_limpio`, `antecedentes_dictamen_anonimizado_limpio`,`contenido_anonimizado_limpio_bert`, `antecedentes_anonimizado_limpio_bert`, `dictamen_anonimizado_limpio_bert` y `antecedentes_dictamen_anonimizado_limpio_bert`|
|Mínimo número de dosumentos en grupo|1% y 5% del tamaño de la muestra|
|Número de tópicos|Todos lo números enteros entre 3 y 35 inclusive|
