"""
在测试时，一批一批地顺序读取测试集轨迹
"""
from loader.data_utils import *


class DataOrderScaner():

    def __init__(self, srcfile, batch):
        self.srcfile = srcfile
        self.batch = batch
        self.srcdata = []
        self.start = 0
        self.size = 0

    def load(self, max_num_line=0):
        num_line = 0
        with open(self.srcfile, 'r') as srcstream:
            for s in srcstream:
                s = [int(x) for x in s.split()]
                self.srcdata.append(np.array(s, dtype=np.int32))
                num_line += 1
                if 0 < max_num_line <= num_line:
                    break
        self.size = len(self.srcdata)
        self.start = 0

    def getbatch(self):
        """
        Output:
        src (seq_len, batch)
        lengths (1, batch)
        invp (batch,): inverse permutation, src.t()[invp] gets original order
        """
        if self.start >= self.size:
            return None, None, None
        src = self.srcdata[self.start:self.start + self.batch]
        # update `start` for next batch
        self.start += self.batch
        return pad_arrays_keep_invp(src)

