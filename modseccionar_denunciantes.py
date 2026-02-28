import unicodedata
import re

def identificar_empresas(nombre):
    palabras = ['empresa',
                'corporación',
                'grupo',
                'corporacion',
                'asociación',
                'asociacion',
                'sociedad',
                'responsabilidad',
                'instituto',
                'comercial',
                'servicios',
                'restaurante',
                'constructora',
                'consultora',
                'congregación',
                'consumidor',
                'usuario',
                'congregacion',
                'limitada',
                'industria',
                'junta']
    for palabra in palabras:
        if palabra in nombre:
            return True
    if ('.' in nombre) and not('iniciales' in nombre) and not('vda' in nombre):
        return True
    else:
        return False

def eliminar_textos_en_nombre(nombre):
    textos = ['representación',
              'sucesión',
              'intestada',
              'señora',
              'señor',
              'calidad',
              'vda. de',
              'viuda',
              ' y '
              ]
    for texto in textos:
        nuevo_nombre = nombre.replace(texto,' ').replace('  ',' ')
        return nuevo_nombre

def destildar(texto):
    texto = unicodedata.normalize("NFD", texto)
    return "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    ).lower()

def extrae_lista_nombres(denunciante):
    lst_nombres = [eliminar_textos_en_nombre(nombre).split('(')[0]
                        for nombre 
                        in denunciante.split(';')]
    lst_nombres = [nombre
                        for nombre 
                        in lst_nombres
                        if not identificar_empresas(nombre) ]

    for i, nombre in enumerate(lst_nombres):
            bigramas = [' de ', 
                        ' del ',
                        ' viuda de ', 
                        ' los ', 
                        ' la ', 
                        ' e ',
                        ' san ',
                        ' de la ',
                        ' de los ',
                        ]
            
            nuevo_nombre = nombre
            for bigrama in bigramas:
                try:
                    if bigrama in nuevo_nombre:
                        nuevo_bigrama = bigrama[:-1] + '-'
                        nuevo_nombre = re.sub(re.escape(bigrama), 
                               nuevo_bigrama, 
                               nombre, flags=re.IGNORECASE)
                        lst_nombres[i] = nuevo_nombre
                except:
                    print('--', denunciante, nombre)
                    pass
            
            lst_nombres[i] = nuevo_nombre.split(' ')

    lst_nombres = [nombre for denunciante_ in lst_nombres 
                          for nombre in denunciante_]
    bigramas = ['de', 'del', 'la', 'los', 'san', 'de la', 'de los']
    for bigrama in bigramas:
        try:
            posicion = lst_nombres.index(bigrama)
            if posicion < (len(lst_nombres)-1):
                lst_nombres[posicion] = lst_nombres[posicion] + ' ' + lst_nombres[posicion + 1]
                lst_nombres[posicion + 1] = ''
        except:
            pass
    lst_nombres = [nombre for nombre in lst_nombres 
                   if (len(nombre)>0) & 
                      ('señor' not in nombre) &
                      ('sucesión' not in nombre) &
                      (nombre not in ['sucesión', 'intestada', 'calidad', 'representación']) ]

    for bigrama in bigramas:
        lst_nombres = [nombre.replace('-',' ') 
                       if f'{bigrama}-' in nombre else nombre 
                       for nombre in lst_nombres]

    if 'e-ingunza' in lst_nombres:
        posicion = lst_nombres.index('e-ingunza')
        lst_nombres[posicion] = 'e ingunza'
        lst_nombres.append('ingunza')
    if 'lopez-' in lst_nombres:
        lst_nombres.append('lopez-raygada')
        lst_nombres.append('lopez')
    

    nombres_compuestos = [nombre for nombre in lst_nombres if '-' in nombre]
    for nombre_compuesto in nombres_compuestos:
        for nombre in nombre_compuesto.split('-'):
            lst_nombres.append(f'{nombre}-')
            lst_nombres.append(f'-{nombre}')

    nombres_compuestos = [nombre for nombre in lst_nombres if ' ' in nombre]
    for nombre_compuesto in nombres_compuestos:
        nombre = nombre_compuesto.split(' ')[-1]
        lst_nombres.append(nombre)
    

    lst_nombres = [destildar(nombre) for nombre in lst_nombres]
    lst_nombres = list(set(lst_nombres))
    bigramas = bigramas + ['en','y']
       
    return lst_nombres