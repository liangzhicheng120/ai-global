import string
import random

digits = string.digits
operators = '+-*'
other = '() '
characters = digits + operators + other


def generate():
    '''
    运算符生成器
    :return:
    '''
    ts = ['{}{}{}{}{}', '({}{}{}){}{}', '{}{}({}{}{})']
    ds = '0123456789'
    os = '+-*'
    cs = [random.choice(ds) if x % 2 == 0 else random.choice(os) for x in range(5)]
    return random.choice(ts).format(*cs)


[print(generate()) for item in range(10)]
