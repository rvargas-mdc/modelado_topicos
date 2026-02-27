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
  a) Considerando todos los datos
  b) Considerando solo las resoluciones provenientes de ORPS de todo el país.
  c) Considerando solo las resoluciones provenientes de comisiones de todo el país.
  d) Considerando solo las resoluciones provenientes de ORPS de Lima.
  e) Considerando solo las resoluciones provenientes de comisiones de Lima.

**0c_Descubrir_nombres.py**:
- Crea lista de nombres de los denunciantes en un archivo CSV que sirve para retirar de ella palabras como "viuda".

**0d_Anonimizar.py**
- Anonimiza números de documentos nacionales de identidad (DNI), números únicos de contribuyentes tributarios (RUC), números telefónicos, correos, direcciones web, nombres de denunciantes y nombres de denunciados.

**0e_Extraer_denuncia.py**

