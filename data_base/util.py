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
        # line_words = line.strip().split(' ')
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
    return re.match('^[\d.:,e-]+$', word) is not None
