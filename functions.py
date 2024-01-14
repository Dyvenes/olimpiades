import random


def pretty_print(arr, value):
    for i in arr:
        for j in i:
            for k in j:
                if value:
                    s = f'print(k.{value}, end=" ")'
                    eval(s)
                else:
                    print(k, end=' ')
            print()
        print()


# a = [[[random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)],
#       [random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)],
#       [random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)]],
#      [[random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)],
#       [random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)],
#       [random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)]],
#      [[random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)],
#       [random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)],
#       [random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)]]]
# pretty_print(a, None)
