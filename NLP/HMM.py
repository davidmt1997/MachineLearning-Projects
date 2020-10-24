import random


class HMM:
    def __init__(self, corpus):
        self.corpus = corpus

    # Devuelve un array de tuplas (token, pos_tag)
    def parse_corpus(self):
        ret = []
        # Abrir el fichero
        with open(self.corpus) as f:
            data = f.read()
            # Dividir el fichero por lineas
            data = data.split('\n')
            for i in data:
                # Por cada linea, ignorar las que empiezan por <
                if '<' in i:
                    continue
                # Si la linea esta vacia, añadir el start token
                if i == '':
                    i = "start start <s>"

                # Separar las demas lineas por palabras
                words = i.split(" ")

                # De esas palabras no quedamos solo con la primera (token) y la tercera (pos tag)
                token, pos_tag = words[0], words[2]
                # Las añadimos al array, cogiendo solo la primera letra del pos_tag, que es la que indica la categoria gramatical de la palabra
                ret.append((token, pos_tag[0]))
            return ret

    # Cuenta el numero de veces que aparece una categoria gramatical en el corpus
    def count_category(self, words, tag):
        count = 0
        for i in words:
            if i[1] == tag:
                count += 1
        return count

    # Devuelve un diccionario con el conteo de veces que aparece cada tag en el corpus
    def get_count_for_category(self, words):
        # De momento todos los signos de puntuacion los he metido en el grupo F
        categories = ['<', 'A', 'C', 'D', 'N', 'P', 'R', 'S', 'V', 'Z', 'W', 'I', 'F']
        # categories = ['A', 'C', 'D', 'N', 'P', 'R', 'S', 'V', 'Z', 'W', 'I', 'Fd', 'Fc', 'Flt', 'Fla', 'Fs', 'Fat', 'Faa',
        # 'Fg', 'Fz', 'Fpt', 'Fpa', 'Ft', 'Fp', 'Fit', 'Fia', 'Fe', 'Fre', 'Fra', 'Fx', 'Fh', 'Fct', 'Fca']
        ret = {}
        for i in categories:
            ret[i] = self.count_category(words, i)
        return ret

    # Devuelve una lista con todas las palabras del corpus (sin repetir)
    # Esta lista es la que se usará para calcular las probabilidades de observación
    def get_distinct_words(self, words):
        ret = []
        for i in words:
            if i[0] not in ret:
                ret.append(i[0])
        return ret

    # Cuenta las veces que dada un palabra (token) su categoria gramatical es el tag
    # Numerador de la formula de la probabilidad de observacion
    def count_category_given_token(self, words, token, tag):
        count = 0
        for word in words:
            if word[0] == token and word[1] == tag:
                count += 1
        return count

    '''
    Devuelve la probabilidad de observacion o emision asociado al estado (tag) de una palabra (token) en el corpus (words)
    Argumentos:
    words: array del corpus entero con sus categorias morfologicas
    dic: diccionario con las diferentes categorias y el numero de veces que aparecen en el corpus
    token: palabra de la que se calcula la probabilidad de observacion
    tag1: estado correspondiente al token
    '''
    def calculate_probabilidad_observacion(self, words, dic, token, tag):
        # Numero de veces que el estado tag se asocia con la palabra token
        numerator = self.count_category_given_token(words, token, tag)
        # Numero de veces que aparece el estado tag en el corpus
        denominator = dic[tag]
        # Para evitar la division por 0
        if denominator == 0:
            denominator += 0.01
        return numerator / denominator

    '''
    Funcion de utilidad que cuenta las veces que se transiciona de un estado (tag1) a un estado (tag2)
    '''
    def count_tag_transitions(self, words, tag1, tag2):
        count = 0
        for i in range(len(words) - 1):  # -1 para evitar salir del tamaño del array
            if words[i][1] is tag1 and words[i + 1][1] is tag2:
                count += 1
        return count

    '''
    Devuelve la probabilidad de transicion de un estado (tag1) a otro estado (tag2)
    Argumentos:
    words: array del corpus entero con sus categorias morfologicas
    dic: diccionario con las diferentes categorias y el numero de veces que aparecen en el corpus
    tag1: estado del que se transiciona
    tag2: estado al que se transiciona
    '''
    def calculate_probabilidad_transicion(self, words, dic, tag1, tag2):
        numerator = self.count_tag_transitions(words, tag1, tag2)
        # Numero de veces que aparece el estado tag1 en el corpus
        denominator = dic[tag1]
        # Para evitar la division por 0
        if denominator == 0:
            denominator += 0.01
        return numerator / denominator

    # Hace un print de la matriz de observacion para que sea mas facil interpretarla
    def pretty_print_observacion(self, words, array):
        categories = ['<', 'A', 'C', 'D', 'N', 'P', 'R', 'S', 'V', 'Z', 'W', 'I', 'F']
        print(categories)
        for i in range(len(array)):
            print(words[i], array[i])

    # Hace un print de la matriz de transicion para que sea mas facil interpretarla
    def pretty_print_transicion(self, array):
        categories = ['<', 'A', 'C', 'D', 'N', 'P', 'R', 'S', 'V', 'Z', 'W', 'I', 'F']
        print(categories)
        for i in range(len(categories)):
            print(categories[i], array[i])

    '''
    Funcion que calcula la matriz de viterbi
    Argumentos: 
    dis: array con todas las palabras del corpus sin repetir
    matriz de probabilidades de observacion
    matriz de probabilidades de transicion
    Frase que se desea analizar
    '''
    def calculate_viterbi_matrix(self, dis, array_observacion, array_transicion, sentence):
        ret = []
        categories = ['<', 'A', 'C', 'D', 'N', 'P', 'R', 'S', 'V', 'Z', 'W', 'I', 'F']
        # Dividimos la frase por palabras
        frase = sentence.split(" ")
        max_num = 1
        # Filas: cada palabra de la frase a analizar
        for i in frase:
            a = []
            # Columnas: cada una de las categorias gramaticales
            for j in range(1, len(categories)):
                # Si una palabra no  esta en el corpus, utilizamos valores aleatorios para calcular la matriz de viterbi
                # ToDo: igual es buena idea quitar lo de aleatorio
                if i not in dis:
                    print("Palabra fuera del corpus")
                    a.append(random.random())
                else:
                    # Calculamos el indice de la columna en la que se encuentra la palabra en la matriz de observacion
                    index = dis.index(i.lower())
                    #print("%s %s max: %s observacion: %s transicion: %s" %(i, categories[j], max_num, array_observacion[index][j], array_transicion[j-1][j]))
                    #print("Resultado: %s" %(max_num * array_observacion[index][j] * array_transicion[j-1][j]))
                    # Aplicamos la formula del algoritmo de viterbi
                    a.append(max_num * array_observacion[index][j] * array_transicion[j-1][j])
            # Calculamos el maximo de la iteracion anterior para usarlo en el calculo en la siguiente iteracion
            if len(a) != 0:
                max_num = max(a)
            ret.append(a)
        return ret

    # Hace un print de cada palabra y su categoria morfologica encontrada segun el algoritmo
    def get_analisis_morfosintactico(self, words, array):
        categories = ['A', 'C', 'D', 'N', 'P', 'R', 'S', 'V', 'Z', 'W', 'I', 'F']
        dict = {'A': 'Adjetivo', 'C':'conjuncion', 'D': 'determinante', 'N': 'nombre', 'P': 'pronombre',
                      'R': 'adverbio', 'S': 'adposicion', 'V': 'verbo', 'Z': 'numero', 'W': 'Fecha', 'I': 'interjeccion', 'F':'signo de puntuacion'}
        for i in range(len(array)):
            index = array[i].index(max(array[i]))
            print(words[i], dict[categories[index]])



def main():
    # Inicializamos una instancia de la clase HMM
    my_hmm = HMM("corpus.txt")
    # Obtenemos un array con los tokens y tags del corpus
    corpus = my_hmm.parse_corpus()
    print(corpus[:15])
    # Obtenemos un diccionario con las veces que aparece cada categoria gramatical en el corpus
    dict = my_hmm.get_count_for_category(corpus)
    for x,y in dict.items():
        print(x, y)
    # Obtenemos una lista de las palabras del corpus sin repetir
    dis = my_hmm.get_distinct_words(corpus)

    probs_observacion = []
    # Por cada palabra del corpus
    for i in dis:
        row = []
        # Por cada categoria gramatical
        for j in dict.keys():
            # Obtenemos un aray con la probabilidad de observacion de estas
            row.append(my_hmm.calculate_probabilidad_observacion(corpus, dict, i, j))
        probs_observacion.append(row)

    my_hmm.pretty_print_observacion(dis, probs_observacion)
    probs_transicion = []
    # Por cada categoria gramatical
    for i in dict.keys():
        row = []
        # Por cada categoria gramatical
        for j in dict.keys():
            # Obtenemos un array con la probabilidad de transicion de estas
            row.append(my_hmm.calculate_probabilidad_transicion(corpus, dict, i, j))
        probs_transicion.append(row)

    my_hmm.pretty_print_transicion(probs_transicion)
    sentence = "El enfermo grave habla de transplantes"
    words = sentence.split(" ")
    res = my_hmm.calculate_viterbi_matrix(dis, probs_observacion, probs_transicion, sentence)
    print(dict.keys())
    for i in range(len(res)):
        print(words[i], res[i])
    my_hmm.get_analisis_morfosintactico(words, res)



if __name__ == "__main__":
    main()
