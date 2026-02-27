# Comparativa de Modelado de Tópicos

Comparativa de modelado de tópicos en un cojunto de resoluciones de Indecopi empleando LDA, NMF, BERTopic y diversos LLM

## Componentes

### Scripts de preparación y limpieza de datos

**0a_Crear dataset.py**: 
- Crea una base de datos de los documentos a partir de archivos de texto creados a partir de los PDF originales.
- Crea la columna cabecera que contiene las cabeceras de los docuemntos.
- Identifica el "área" resolutiva (ORPS, Comisión de Protección al Consumidor, Tribunal de Protección al Consumidor)
- Identifica el lugar dende se encuentra el área resolutiva (Sede Central, Lima Norte, Lima Sur, provincias)
- Identifica al denunciante, al denunciado, la materia sobre la que versa la resolución y la actividad económica.
- 