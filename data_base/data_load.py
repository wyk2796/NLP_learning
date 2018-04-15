# coding:utf-8
import os


class Read(object):
    def __init__(self, path, encoding='utf-8'):
        self.content = []
        self.encoding = encoding
        self.path = path
        self._load_text(self.path)

    def _read_line(self, line):
        pass

    def _begin_read(self):
        pass

    def _end_read(self):
        pass

    def get_content(self):
        return self.content

    def _load_text(self, path):
        if os.path.isdir(path):
            [self._load_text(path + '/' + f_name)
             for f_name in os.listdir(path)]
        else:
            self._operation_single_file(path)

    def _operation_single_file(self, path):
        try:
            print('operation file path: %s' % path)
            with open(path, encoding=self.encoding, mode='r') as f:
                temp_read = f.readlines()
                self._begin_read()
                for line in temp_read:
                    reader_str = self._read_line(line)
                    if reader_str:
                        self.content.append(reader_str)
                self._end_read()
        except Exception as e:
            print('operation file %s encounter an error:%s' % (path, e))


class ShakespeareRead(Read):

    def _read_line(self, line):
        line = line.strip()
        if line.find('<') > -1 or line.find('>') > -1:
            return None
        elif '' == line:
            return None
        else:
            return line





