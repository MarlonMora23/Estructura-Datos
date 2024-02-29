#Funciones para el ordenamiento de los datos
def bubble_sort(data, condition, reverse=False):
    for i in range(0, len(data)-1):	
        for j in range(0, len(data) - i - 1):	
            if data[j][condition] > data[j+1][condition]:	
                data[j], data[j+1] = data[j+1], data[j]
    
    return data if reverse == False else data[::-1]

def improved_bubble_sort(data, condition, reverse=False) -> list:
    i = 0
    control = True
    while(i <= len(data) - 2) and control:
        control = False
        for j in range(0, len(data) - 1 - i):
                if data[j][condition] > data[j+1][condition]:
                    data[j+1], data[j] = data[j], data[j+1]
                    control = True

        i += 1 

    return data if reverse == False else data[::-1]

def bidirectional_bubble_sort(data, condition, reverse=False):
    izquierda = 0	
    derecha = len(data)-1	
    control = True	

    while (izquierda < derecha) and control:	
        control = False	

        for i in range(izquierda, derecha):	
            if(data[i][condition] > data[i+1][condition]):	
                control = True	
                data[i], data[i+1] = data[i+1], data[i]	

        derecha -= 1	

        for j in range(derecha, izquierda, -1):	
            if(data[j][condition] < data[j-1][condition]):	
                control = True	
                data[j], data[j-1] = data[j-1], data[j]	

        izquierda += 1	

    return data if reverse == False else data[::-1]

def selection_sort(data, condition, reverse=False) -> list:
    for i in range(0, len(data)-1):
        minimo = i
        for j in range(i+1, len(data)):
            if(data[j][condition] < data[minimo][condition]):
                minimo= j	

        data[i], data[minimo] = data[minimo], data[i]
    
    return data if reverse == False else data[::-1]

def quick_sort(data, first, last, condition, reverse) -> list:
    left = first
    right = last - 1
    pivote = last

    while(left < right):
        while (data[left][condition] < data[pivote][condition]) and (left <= right):
            left += 1
        while (data[right][condition] > data[pivote][condition]) and (right >= left):
            right -= 1
        if(left < right):
            data[left], data[right] = data[right], data[left]
    
    if(data[pivote][condition] < data[left][condition]):
            data[left], data[pivote] = data[pivote], data[left]
    
    if(first < left):
        quick_sort(data, first, left-1, condition, reverse)
    
    if(last>left):
        quick_sort(data, left+1, last, condition, reverse)

    return data if reverse == False else data[::-1]

def handle_quicksort(data, condition, reverse=False) -> list:
    first = 0
    last = len(data) - 1
    return quick_sort(data, first, last, condition, reverse)

def insertion_sort(data, condition, reverse=False) -> list:
    for i in range(1, len(data)+1):
        k=i-1
        while (k>0) and (data[k][condition]<data[k-1][condition]):
            data[k], data[k-1] = data[k-1], data[k]
            k -= 1

    return data if reverse == False else data[::-1]

def merge_sort(data, condition, reverse=False) -> list:
    if len(data) <= 1:
        return data
    else:
        middle = len(data) // 2
        left = []
        right = []

        for i in range(0, middle):
            left.append(data[i])

        for i in range(middle, len(data)):
            right.append(data[i])

        left = merge_sort(left, condition, reverse)
        right = merge_sort(right, condition, reverse)

        if left[middle - 1][condition] <= right[0][condition]:
            left += right
            return left
        
        result = merge(left, right, condition)

        return result if reverse == False else result[::-1]

def merge(left, right, condition) -> list:
    mixed_list = []
    while len(left) > 0 and len(right) > 0:
        if left[0][condition] < right[0][condition]:
            mixed_list.append(left.pop(0))
        else:
            mixed_list.append(right.pop(0))

    if len(left) > 0:
        mixed_list += left

    if len(right) > 0:
        mixed_list += right

    return mixed_list

def count_sort(data, condition, max_value) -> list:
    count_list  = [0] * (max_value + 1)
    sorted_list  = [None] * len(data)

    for i in data:
        count_list[i[condition]] += 1

    total = 0
    for i in range(len(count_list)):
        count_list[i], total = total, total + count_list[i]

    for index in data:
        sorted_list[count_list[index[condition]]] = index
        count_list[index[condition]] += 1

    return sorted_list

def handle_count_sort(data, condition, reverse=False) -> list:
    max_value = max(data, key=lambda x: x[condition])[condition]
    result = count_sort(data, condition, max_value)

    return result if  reverse == False else result[::-1]

def bucket_sort(data, condition, reverse=False) -> list:
    num_buckets = 10
    buckets = [[] for _ in range(num_buckets)]

    # Distribuir elementos en las cubetas
    for element in data:
        bucket_index = int(element[condition] * num_buckets)
        buckets[bucket_index].append(element)

    # Ordenar cada cubeta usando otro algoritmo 
    for i in range(num_buckets):
        buckets[i] = sorted(buckets[i], key=lambda x: x[condition])

    # Combinar las cubetas ordenadas
    sorted_data = []
    for bucket in buckets:
        sorted_data.extend(bucket)

    return sorted_data if reverse == False else sorted_data[::-1]

def counting_sort(data, exp, condition):
    n = len(data)
    output = [0] * n
    count = [0] * 10

    # Contar la frecuencia de cada dígito en la posición actual
    for i in range(n):
        index = data[i][condition] // exp
        count[index % 10] += 1

    # Calcular las posiciones finales acumuladas
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Construir la lista ordenada
    i = n - 1
    while i >= 0:
        index = data[i][condition] // exp
        output[count[index % 10] - 1] = data[i]
        count[index % 10] -= 1
        i -= 1

    # Copiar la lista ordenada de vuelta a la lista original
    for i in range(n):
        data[i] = output[i]

def radix_sort(data, condition, reverse=False) -> list:
    
    max_num = max(data, key=lambda x: x[condition])[condition]
    exp = 1

    while max_num // exp > 0:
        counting_sort(data, exp, condition)
        exp *= 10
    
    return data if reverse == False else data[::-1]

def shell_sort(data, condition, reverse=False) -> list:
    n = len(data)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = data[i]
            j = i
            while j >= gap and data[j - gap][condition] > temp[condition]:
                data[j] = data[j - gap]
                j -= gap
            data[j] = temp

        gap //= 2
    
    return data if reverse == False else data[::-1]

def tim_sort(data, condition, reverse=False) -> list:
    min_run = 32

    def insertion_sort(arr, left, right):
        for i in range(left + 1, right + 1):
            j = i
            while j > left and arr[j - 1][condition] > arr[j][condition]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                j -= 1

    def merge(arr, l, m, r):
        len1, len2 = m - l + 1, r - m
        left, right = [], []

        for i in range(0, len1):
            left.append(arr[l + i])
        for i in range(0, len2):
            right.append(arr[m + 1 + i])

        i, j, k = 0, 0, l
        while i < len1 and j < len2:
            if left[i][condition] <= right[j][condition]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len1:
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len2:
            arr[k] = right[j]
            j += 1
            k += 1

    n = len(data)
    for i in range(0, n, min_run):
        insertion_sort(data, i, min((i + min_run - 1), n - 1))

    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min((left + size - 1), (n - 1))
            right = min((left + 2 * size - 1), (n - 1))

            if mid < right:
                merge(data, left, mid, right)

        size *= 2

    return data if reverse == False else data[::-1]

# Imprimir el título de la característica
def print_title(title) -> None:
    #Asignar dinámicamente el tamaño de la decoración del titulo
    print('\n' + '-'*len(title)*2)
    print(f'{title:^{len(title)*2}}')
    print('-'*len(title)*2)

# Imprimir mensaje de invalidez
def print_not_valid_msg() -> None:
    print('\n///////////////////////////////////////')
    print('!Por favor, ingrese una opción válida!')
    print('///////////////////////////////////////\n')


# Obtener titulos y su cantidad de espacios
def get_titles_spaces(cant_spaces) -> list[list]:
    return [
        ['Deporte', cant_spaces[0]],
        ['Año', cant_spaces[1]],
        ['Espectadores', cant_spaces[2]],
        ['Participantes', cant_spaces[3]],
        ['País de', cant_spaces[4]],
        ['Ganancias', cant_spaces[5]],
        ['Torneo más importante', cant_spaces[6]],
        ['Frecuencia', cant_spaces[7]],
        ['Duración', cant_spaces[8]],
        ['Torneos', cant_spaces[9]],
        ['Cantidad', cant_spaces[10]],
    ]

# Obtener subtitulos y su cantidad de espacios
def get_subtitle_spaces(cant_spaces) -> list[list]:
    return [
        ['', cant_spaces[0]],
        ['Creación', cant_spaces[1]],
        ['(Millones)', cant_spaces[2]],
        ['', cant_spaces[3]],
        ['origen', cant_spaces[4]],
        ['(Millones)', cant_spaces[5]],
        ['', cant_spaces[6]],
        ['(Años)', cant_spaces[7]],
        ['(Días)', cant_spaces[8]],
        ['(Hechos)', cant_spaces[9]],
        ['Reglas', cant_spaces[10]],
    ]

# Imprimir el resultado de la búsqueda
def print_search_result(data_spaces, results) -> None:
    for position, spaces in data_spaces:
        print(f'{results[0][position]:^{spaces}}', end=' ')
    print()

# Imprimir el resultado del ordenamiento
def print_sorting_result(data_spaces, results) -> None:
    for i in range(len(results)):
            for position, spaces in data_spaces:
                # Imprimir datos ordenados
                print(f'{results[i][position]:^{spaces}}', end=' ')
            print() 

#Imprimir el encabezado
def print_header(titles_spaces, subtitle_spaces) -> None:
    print('-'*170)
    for title, space in titles_spaces:
        print(f'{title:^{space}}', end=' ')
    print()

    for subtitle, spaces in subtitle_spaces:
        print(f'{subtitle:^{spaces}}', end=' ')  
    print()
    print('-'*170)

# Posicionar el encabezado en la posición correcta
def organize_header(cant_spaces, position) -> None:
    titles_spaces = get_titles_spaces(cant_spaces)
    subtitle_spaces = get_subtitle_spaces(cant_spaces)
    
    #Poner en la segunda posicion el titulo del elemento que se está buscando
    titles_spaces[1][0], titles_spaces[position][0] = titles_spaces[position][0], titles_spaces[1][0]
    titles_spaces[1][1], titles_spaces[position][1] = titles_spaces[position][1], titles_spaces[1][1]

    #Poner en la segunda posicion el subtitulo del elemento que se está buscando
    subtitle_spaces[1][0], subtitle_spaces[position][0] = subtitle_spaces[position][0], subtitle_spaces[1][0]
    subtitle_spaces[1][1], subtitle_spaces[position][1] = subtitle_spaces[position][1], subtitle_spaces[1][1]

    # Retornar la cantidad de espacios correspondientes a la nueva columna
    return titles_spaces, subtitle_spaces

# Imprimir la tabla de resultados ordenada
def print_results(results, label, position, search=False) -> None:
    # Mostrar la etiqueta de que se está ordenando o buscando
    print(f'{label:^170}')

    #Si se seleccionó antiguedad, imprimir los años que han pasado
    if position == 1:
        print(f'El deporte {results[0][0]} tiene {2024 - int(results[0][1])} años')

    #Definir la cantidad de espacios de cada columna
    cant_spaces = [18, 8, 15, 18, 14, 15, 32, 12, 10, 10, 10]

    titles_spaces, subtitle_spaces = organize_header(cant_spaces, position)
    #Imprimir encabezados (titulos y subtitulos)
    print_header(titles_spaces, subtitle_spaces)

    #Asignar a cada posicion, la cantidad de espacios que ocupa
    data_spaces = [[i, cant_spaces[i]] for i in range(len(cant_spaces))]

    #Poner en la segunda posición el elemento que se está buscando
    for i in range(len(results)):
        results[i][1], results[i][position] = results[i][position], results[i][1]

    #Cambiar los espacios de la segunda columna por el tamaño del elemento que se ordeno
    data_spaces[1][1], data_spaces[position][1] = data_spaces[position][1], data_spaces[1][1]
    
    #Si se quiere hacer una búsqueda
    if search:
        # Imprimir búsqueda 
        print_search_result(data_spaces, results)
    #Sino, mostrar toda la lista
    else:
        print_sorting_result(data_spaces, results)

    print('-'*170)


#Mostrar menú de opciones disponibles
def show_options(options) -> None:
    for index, option in options.items():
        print(f'{index}. {option}')

    print('0. Volver')

# Obtener nombres de los algoritmos de ordenamiento
def get_sort_algorithms_names() -> dict:
    return {
        1 : 'Burbuja',
        2 : 'Bubuja mejorado',
        3 : 'Burbuja bidireccional',
        4 : 'Seleccion',
        5 : 'QuickSort',
        6 : 'Insercion',
        7 :  'MergeSort',
        8 : 'CountSort',
        9 : 'BuckerSort',
        10 : 'RadixSort',
        11 : 'ShellSort',
        12 : 'TimSort'
    }

# Obtener algoritmos de ordenamiento
def get_sort_algorithms() -> dict:
    return {
        1 : bubble_sort,
        2 : improved_bubble_sort,
        3 : bidirectional_bubble_sort,
        4 : selection_sort,
        5 : handle_quicksort,
        6 : insertion_sort,
        7 : merge_sort,
        8 : handle_count_sort,
        9 : bucket_sort,
        10 : radix_sort,
        11 : shell_sort,
        12 : tim_sort
    }

# Seleccionar algoritmo de ordenamiento
def select_sort_algorithm() -> callable:
    #Obtener algoritmos de ordenamiento
    sort_algorithms = get_sort_algorithms_names()
    #Mostrar los algoritmos de ordenamiento
    show_options(sort_algorithms)
    
    try:
        sort_algorithm = int(input('\nDigita el numero del ordenamiento a usar: '))
    except ValueError:
        # Si no se digita un número, lanzar error
        print_not_valid_msg()
        return select_sort_algorithm()

    # Si se selecciona volver, regresar al menu anterior
    if sort_algorithm == 0:
        return sort_algorithm
    
    # Si la opcion seleccionada no está dentro de las opciones disponibles, lanzar error
    if sort_algorithm not in range(1, len(sort_algorithms) + 1):
        print_not_valid_msg()
        return select_sort_algorithm()
    
    # Retornar el algoritmo seleccionado
    return get_sort_algorithms()[sort_algorithm]

# Obtener argumentos para cada tipo de ordenamiento
def sort_algorithms_args(data, position) -> dict:
    return {
        1 : [data, position],
        2 : [data, position, True],
        3 : [data, position],
        4 : [data, position, True]
    }

# Obtener opciones que tienen búsqueda (para ordenamientos numéricos)
def if_search() -> dict:
    return {
        1 : False,
        2 : False,
        3 : True,
        4 : True
    }

# Obtener ordenamientos numéricos
def get_numeric_sorts(description) -> dict:
    return {
        1 : f'Ordenar de {description[0]} a {description[1]}', 
        2 : f'Ordenar de {description[1]} a {description[0]}',
        3 : f'Buscar deporte {description[2]}{description[0]}',
        4 : f'Buscar deporte {description[2]}{description[1]}',
    }

# Obtener la etiqueta de como se ordena
def get_numeric_sorts_label(description) -> dict:
    return {
        1 : f'-'*170 + '\n' + f'{("Ordenando de " + description[0] + " a " + description[1]):^170}',
        2 : f'-'*170 + '\n' + f'{("Ordenando de " + description[1] + " a " + description[0]):^170}',
        3 : f'-'*170 + '\n' + f'{("Deporte " + description[2] + description[0]):^170}',
        4 : f'-'*170 + '\n' + f'{("Deporte " + description[2] + description[1]):^170}'
    }

# Obtener ordenamientos alfabéticos
def get_alphabetic_sorts(_) -> dict:
    return {
        1 : 'Ordenar de A hasta Z',
        2 : 'Ordenar de Z hasta A'
    }

# Obtener la etiqueta de como se ordena
def get_alphabetic_sorts_label(_) -> dict:
    return {
        1 : '-'*170 + '\n' + f'{"Ordenando de A hasta Z":^170}',
        2 : '-'*170 + '\n' + f'{"Ordenando de Z hasta A":^170}'
    }

# Obtener el tipo de ordenamiento selecionado
def select_order_or_search(title, description, get_type_sorts) -> int:
    print_title(title)
    # Obtener opciones de ordenamiento según si es numérico o alfabético
    sort_options = get_type_sorts(description)
    # Mostrar opciones disponibles para el tipo correspondiente
    show_options(sort_options)

    try:
        option = int(input('\nDigite el número de la opción que quieres averiguar: '))
    except ValueError:
        # Si no se digita un número, lanzar error
        print_not_valid_msg()
        return select_order_or_search(title, description, get_type_sorts)
    
    # Si se selecciona volver, regresar al menú anterior
    if option == 0:
        return option
    
    # Si la opción seleccionada no está dentro de las opciones disponibles, lanzar error
    if option not in range(1, len(sort_options) + 1):
        print_not_valid_msg()
        return select_order_or_search(title, description, get_type_sorts)
    
    return option

#Funcion para manejar los casos de ordenamiento o busqueda
def handle_order_or_search(data, title, description, position, get_type_sorts, get_type_sorts_label, has_search) -> None:
    # Seleccionar en base a qué se va a ordenar o buscar
    sort_option = select_order_or_search(title, description, get_type_sorts)
    # Si la opción es 0, volver al menú anterior
    if sort_option == 0:
        return
    
    # Seleccionar algoritmo de ordenamiento
    sort_algorithm = select_sort_algorithm()
    # Si la opción es 0, repetir funcion
    if sort_algorithm == 0:
        return handle_order_or_search(data, title, description, position, get_type_sorts, get_type_sorts_label, has_search)

    # Obtener argumentos de la función necesarios
    sort_algorithm_args = sort_algorithms_args(data, position)[sort_option]
    # Llamar a la función de ordenamiento
    results = sort_algorithm(*sort_algorithm_args)
    # Obtener etiqueta del título
    label = get_type_sorts_label(description)[sort_option]

    #Si la característica tiene busqueda
    if has_search:
        # Determinar si se va a ordenar o buscar
        search = if_search()[sort_option]
    else:
        search = False

    # Imprimir el resultado
    print_results(results, label, position, search)

# Funcion para casos numéricos
def handle_numeric_features(data, title, description, position) -> None:
    handle_order_or_search(data, title, description, position, get_numeric_sorts, get_numeric_sorts_label, has_search=True)

# Funcon para casos alfabéticos
def handle_alphabetic_features(data, title, position) -> None:
    handle_order_or_search(data, title, '', position, get_alphabetic_sorts, get_alphabetic_sorts_label, has_search=False)

                
# Obtener los nombres de las características
def get_features() -> dict:
    return {
        1 : 'Antiguedad de los deportes',
        2 : 'Cantidad de espectadores (2023)' ,
        3 : 'Cantidad de participantes por juego',
        4 : 'País de origen',
        5 : 'Ganancias (2023)',
        6 : 'Torneo más importante',
        7 : 'Frecuencia del torneo mas importante',
        8 : 'Tiempo de duración del torneo mas importante',
        9 : 'Torneos realizados (2023)',
        10 : 'Cantidad de reglas',
    }

#Obtener la funcion requerida para cada característica
def features_handlers() -> dict:
    return {
        1 : handle_numeric_features,
        2 : handle_numeric_features,
        3 : handle_numeric_features,
        4 : handle_alphabetic_features,
        5 : handle_numeric_features,
        6 : handle_alphabetic_features,
        7 : handle_numeric_features,
        8 : handle_numeric_features,
        9 : handle_numeric_features,
        10 : handle_numeric_features
    }

#Obtener los parámetros de cada característica
def get_features_args() -> dict:
    return {
            1 : ['Antiguedad', ['mas antiguo', 'mas nuevo', ''], 1],
            2 : ['Cantidad de espectadores', ['menos espectadores', 'mas espectadores', 'con '], 2],
            3 : ['Cantidad de Participantes', ['menos participantes', 'mas participantes', 'con '], 3],
            4 : ['País de origen', 4],
            5 : ['Ganancias', ['menos ingresos', 'mas ingresos', 'con '], 5],
            6 : ['Mejor Torneo', 6],
            7 : ['Frecuencia del torneo mas importante', ['menos frecuente', 'mas frecuente', ''],  7],
            8 : ['Duración del torneo mas importante', ['menor duracion', 'mas duracion', 'con '], 8],
            9 : ['Cantidad de torneos mas importantes realizados', ['menos torneos realizados', 'mas torneos realizados', 'con '], 9],
            10 : ['Cantidad de reglas', ['menor cantidad de reglas', 'mayor cantidad de reglas', 'con '], 10],
    }

#Mostrar las características
def show_features(features) -> None:
    for index, feature in features.items():
        print(f'{index}. {feature}')

    print(f'-1. Salir')

#Seleccionar la característica de ordenamiento         
def select_sort_feature() -> int:
    features = get_features()
    show_features(features)

    try:
        sort_feature = int(input('\nDigite el número de la opción que quieres averiguar: '))
    except ValueError:
        print_not_valid_msg()
        return select_sort_feature()
    
    if sort_feature == -1:
        return sort_feature

    if sort_feature not in range(1, len(features) + 1):
        print_not_valid_msg()
        return select_sort_feature()
    
    return sort_feature

#Inicializar la informacion
def initialize_data() -> list[list]:
    """
    Información de cada columna: 
    1. Deporte, 2. Antiguedad, 3. Espectadores, 4. Participantes, 5. Pais origen, 6. Ganancias, 
    7. Torneo mas importante, 8. Frecuencia torneo mas importante, 9. Duracion torneo mas importante, 
    10. Torneos mas importantes realizados, 11. Cantidad de reglas
    """
    return [
        ['Futbol', 1863, 3.5, 11, 'Inglaterra', 529.9, 'World cup', 4, 29, 21, 22], 
        ['Baloncesto', 1891, 820, 5, 'Estados Unidos', 10, 'Copa Mundial de baloncesto', 4, 17, 13, 18], 
        ['Tenis', 1200, 1.0, 2, 'Francia', 1.589, 'Grand Slam', 1, 16, 25, 400], 
        ['Beisbol', 1845, 12, 9, 'Estados Unidos', 10.3, 'Clasico mundial de beisbol', 2, 13, 10, 5], 
        ['Boxeo', 1681, 0.132, 2, 'Inglaterra ', 2.155, 'Campeonato mundial de boxeo', 2, 13, 12, 68], 
        ['Hockey', 1893, 2000, 6, 'Estados Unidos', 1.28, 'National Hockey League', 1, 212, 42, 106], 
        ['Futbol americano', 1869, 17, 11, 'Estados Unidos', 571, 'Super bowl', 1, 152, 42, 57], 
        ['Atletismo', 776, 0.321, 4, 'Grecia', 70.0, 'Campeonato mundial de atletismo', 2, 10, 42, 19], 
        ['Karate', 1500, 40, 7, 'Japon', 60.0, 'Campeonato mundial de karate', 2, 6, 5, 26], 
        ['Ciclismo', 1817, 715, 8, 'Alemania', 1.9, 'Tour de francia', 1, 24, 10, 110]
    ]


#Función principal
def Main() -> None:
    while True:
        print('\n' + '-'*50)
        print(f'{"Cosas que no sabías sobre deportes":^50}')
        print('-'*50 + '\n')

        #Obtener la información
        data = initialize_data() 
        #Obtener caracteristica a buscar u ordenar
        sort_feature = select_sort_feature()

        #Si el usuario digita salir, se termina el programa
        if sort_feature == -1:
            return
        
        #Obtener argumentos de la caracteristica esogida
        sort_feature_args = get_features_args()[sort_feature]
        #Obtener funcion requerida para la caracteristica escogida
        feature_handler = features_handlers()[sort_feature]
        #llamar la funcion correspondiente con sus argumentos
        feature_handler(data, *sort_feature_args)

if __name__ == '__main__':
    Main()