import numpy
import random
from pprint import pprint
from functions import pretty_print

import numpy as np

with open('RUS.txt', 'r') as words_txt:
    dictionary = words_txt.read().split()[:3]

counter = 0
alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя '

activated_neurons = []


def activation(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, sent_weight, sent_parents):
        self.weight = sent_weight
        self.parents = sent_parents[0]
        self.znach = 0

    def send_inpuls(self):
        # если нейрон возбудился
        self.znach = activation(self.znach)
        if self.znach >= 0.7:
            for i in range(len(self.parents)):
                for j in range(len(self.parents[i])):
                    # отправляем взвешенные сигналы соответсвующим родителям
                    # сейчас отправляется активированное значение (стр. после первого if).
                    # Но все равно суммарное значение на родиель приходит слишком большое. Возможно нужно найти среднее ирфметическое или средневзвешанное.
                    self.parents[i][j].znach += self.znach * self.weight[i][j]


    def thinking(self):

        # процесс обработки сигнала. Импульс идет с начала от самого последнего (выходного)
        # к начальным (входным) те возвращают свои значения, которые перемножаются на веса,
        # которые свои у каждой связи(от одного нейрона идет несколько связей к следующему
        # слою и у каждой связи свой вес).

        self.activated_childs = set()
        if self.parents:
            znach = [neur.thinking() for neur in self.parents]
            come_in = []
            for i in znach:
                if i[0]:
                    come_in.append(i[0] * self.weight)
                    self.activated_childs = self.activated_childs.union(i[1])
            znach = activation(sum(come_in))
            if znach >= 0.6:
                #    print('send', come_in * self.weight)
                activated_neurons.append(self)
                return [znach, self.activated_childs.union({self})]
            else:
                return [None, {}]
        elif self.is_active:
            return [1, {self}]  # нужна тут ф-ция активации или нет?
        else:
            return [None, {}]

    def good(self):
        # повышает веса у всех
        self.weight += 0.1


def train(training_set_inputs, training_set_outputs, output):
    error = training_set_outputs - output
    adjustment = np.dot(training_set_inputs.T, error * (output * (1 - output)))
    return adjustment


# x1 = 0
# x2 = 0
# x3 = 0
# massiv = []

neurons = np.ndarray((3, 3, 3), dtype=Neuron)
for i in range(len(neurons)):
    for j in range(len(neurons[i])):
        for k in range(len(neurons[i][j])):
            if i == 1:
                pass
            if i == len(neurons) - 1:
                # в ячейке создается нейрон в которм входной массив неронов без родительских и соответственно весов
                neurons[i][j][k] = Neuron([], [[]])
            else:
                # в нейрон записывается значение весов для следующего слоя (у каждого родителя свой вес). Слой индексируется i. Неронка 3-х мерная.
                neurons[i][j][k] = Neuron(np.random.rand(3, 3), [neurons[i + 1]])

inp = 'я не что'.split()
for i in range(len(inp)):
    # в зависимости на каком месте i стоит конкретное слово (dictionary.index(inp[i])) полученый нерон возбуждается
    neurons[0][i][dictionary.index(inp[i])].znach = 1

for k in neurons:
    for i in k:
        for j in i:
            j.send_inpuls()

answ = [[] for i in range(3)]
for i in range(len(neurons[-1])):
    for j in range(len(neurons[-1][i])):
        zn = neurons[-1][i][j].znach
        if neurons[-1][i][j].znach >= 0.7:
            answ[i].append(dictionary[j])
for i in answ:
    if i:
        print(random.choice(i), end=' ')
    else:
        print(None, end=' ')
print()
print()
pretty_print(neurons, 'znach')

# for i in range(4):  # степень
#     massiv.append([])
#     for j in range(3):  # число
#         if i == 0:
#             massiv[i].append(Neuron(random.choice([0.1, 0.7]), []))
#         else:
#             massiv[i].append(Neuron(random.choice([0.1, 0.7]), massiv[i - 1]))
#
# weights = []
# for i in range(len(massiv)):
#     weights.append([])
#     for j in range(len(massiv[i])):
#         weights[i].append(massiv[i][j].weight)
#
# pprint(weights)
#
# all_neurons = []
# for i in massiv:
#     all_neurons += i
# count = 0
#
# while count != 10000:
#     count += 1
#     vvod = 'я'.split()
#     right_answ = 'что не что'.split()
#
#     train_right_answ = [[0 for j in range(len(dictionary))] for _ in range(3)]
#
#     train_input = [0 for j in range(len(dictionary))]
#     train_neurons = []
#
#     for i in range(len(right_answ)):
#         train_right_answ[i][dictionary.index(right_answ[i])] = 1
#
#     text = vvod.copy()
#     answ_neur = []
#     viviod = []
#     for i in range(3):
#         text = vvod.copy() + viviod
#         for j in text:
#             massiv[0][dictionary.index(j)].is_active = 1
#             train_input[dictionary.index(j)] = 1
#         for j in massiv[-1]:
#             answ_neur.append(j.thinking()[0])
#         clean_answ = []
#         for j in answ_neur:
#             if j:
#                 clean_answ.append(j)
#         # word = dictionary[answ_neur.index(max(clean_answ))]
#
#         for j in range(len(answ_neur)):
#             if not answ_neur[j]:
#                 answ_neur[j] = 0
#         if clean_answ:
#             viviod.append(dictionary[answ_neur.index(max(clean_answ))])
#             train_neurons = list(massiv[-1][answ_neur.index(max(clean_answ))].activated_childs)
#             # adjustment = train(array(train_input), array(train_right_answ)[i], array(answ_neur))
#         else:
#             viviod.append('')
#             train_neurons = []
#         print('vivod', viviod, 'right_answ', right_answ, 'i', i)
#         if viviod and viviod[i] == right_answ[i]:
#             adjustment = 0.1
#             train_neurons = list(massiv[-1][answ_neur.index(max(clean_answ))].activated_childs)
#         else:
#             adjustment = -0.1
#             if clean_answ:
#                 train_neurons = list(massiv[-1][answ_neur.index(max(clean_answ))].activated_childs)
#             else:
#                 train_neurons = []
#         for j in train_neurons:
#             j.weight += adjustment
#         not_train_neurons = []
#         for j in all_neurons:
#             if j not in train_neurons:
#                 not_train_neurons.append(j)
#         for j in not_train_neurons:
#             j.weight -= adjustment
#         # for j in text:
#         #    massiv[0][dictionary.index(j)].is_active = 0
#         answ_neur = []
#
#         for l in massiv:
#             for j in l:
#                 print(round(j.weight, 3), end=' ')
#             print()
#         print('------------------------------')
#
#
#     print(' '.join(viviod))
