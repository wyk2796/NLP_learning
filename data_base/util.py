# coding:utf-8
import os
import re


def _operation_single_file(path, func, *args, **kwds):
    try:
        print('operation file path: %s' % path)
        func(path, *args, **kwds)
    except Exception as e:
        print('operation file %s encounter an error:%s' % (path, e))


def operation_file(path, func, filter_func=lambda x: True, *args, **kwds):
    if os.path.isdir(path):
        [_operation_single_file(path + f_name, func, *args, **kwds)
         for f_name in os.listdir(path) if filter_func(f_name)]
    else:
        _operation_single_file(path, func, *args, **kwds)


def save_to_file(path, content, encoding='utf-8'):
    with open(path, encoding=encoding, mode='w') as out:
        for line in content:
            out.write(' '.join(line) + '\n')


def transform_lines_list(content):
    """
    cope with content to a definite form
    :param content: original text, like [I love you!]
    :return: handled text like [[I],[love],[you],[!]]
    """
    data = []
    for line in content:
        line_transform = _handler_not_char(line).strip()
        words = line_transform.split(' ')
        fina_w = []
        for w in words:
            if w != '' and w != '':
                fina_w.append(w)
        data.append(fina_w)
    return data


def word_frp_statistic(content):
    word_map = {}
    for line in content:
        for w in line:
            if w != '':
                if w in word_map:
                    word_map[w] += 1
                else:
                    word_map[w] = 1
    return word_map


def _handler_not_char(line):
    def _add_white_space(matched):
        value = str(matched.group('value').strip())
        if value != ' ' and value != '-':
            value = ' ' + value + ' '
        return value
    a_line = re.sub('(?P<value>\W)', _add_white_space, line)
    return a_line


def is_num(word):
    """
    the function discriminate a word whether is a number or not.
    :param word:
    :return: if word is a number, return True or False
    """
    a=re.match('[+-]?\d+$', word)
    b=re.match('[+-]?(\d+)\.(\d+)$', word)
    c=re.match('[+-]?\d{1,3}(,\d{3})*$', word)
    d=re.match('[+-]?(\d+)(\.(\d+))?[-+]?e\d+$', word)
    return a is not None or \
           b is not None or \
           c is not None or \
           d is not None


def is_date(word):
    a = re.match('[0-1]?\d-[0-3]?\d(-\d{4})?$', word)
    b = re.match('[0-1]?\d:[0-3]?\d:\d{4}?$', word)
    c = re.match('[0-2]?\d(:[0-6]?\d){1,2}$', word)
    return a is not None or \
           b is not None or \
           c is not None
