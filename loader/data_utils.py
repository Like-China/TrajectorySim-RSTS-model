import numpy as np
import torch
import settings

from collections import namedtuple


def argsort(seq):
    """
    sort by length in reverse order
    ex. src=[[1,2,3],[3,4,5,6],[2,3,4,56,3]] ，return 2，1，0
    :param seq: (list[array[int32]])
    :return: the reversed order
    """
    return [x for x, y in sorted(enumerate(seq), key=lambda x: len(x[1]), reverse=True)]


def pad_array(a, max_length, PAD=settings.PAD):
    """
    :param a: 一条待补位操作的轨迹 (array[int32])
    :param max_length: 该条轨迹所在批次轨迹中 轨迹的最大长度，按该长度标准补位
    :param PAD: 补位值，设置为0
    :return: 补位后的轨迹
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))


def pad_arrays(a):
    """
    多条轨迹(一个批次的轨迹)补位操作，每条轨迹的长度补0，使其长度和该batch最长轨迹的长度相同
    :param a: 一批轨迹
    :return:
    """
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)


def pad_arrays_pair(src, trg, keep_invp=False):
    """
    1. 对轨迹补零操作，使得所有轨迹的长度都一样长
    2. 对轨迹长度从大到小进行排序
    3. 返回TD类，其中轨迹点列表进行了转置操作，每列代表一个轨迹点
    4. 返回形式 ['src', 'lengths', 'trg', 'invp']

    :param src: (list[array[int32]])
    :param trg: (list[array[int32]])
    :param keep_invp: 是否需要保留原来的轨迹长度排序索引
    :return:

    src (seq_len1, batch)
    trg (seq_len2, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    TD = namedtuple('TD', ['src', 'lengths', 'trg', 'invp'])
    assert len(src) == len(trg), "source and target should have the same length"
    idx = argsort(src)
    src = list(np.array(src)[idx])
    trg = list(np.array(trg)[idx])

    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    trg = pad_arrays(trg)
    if keep_invp == True:
        invp = torch.LongTensor(invpermute(idx))
        return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=invp)
    else:
        return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=[])


def invpermute(p):
    """
    输入p,返回p的每个位置的值的索引invp
    idx = [5, 7, 8, 9, 6, 1, 2, 0, 3, 4]
    invp(idx) = [7, 5, 6, 8, 9, 0, 4, 1, 2, 3]
    invp[p[i]] = i 如 p中有个数是45，我现在想知道45在p的第几个位置，那么invp[45]会告诉我们答案
    invp[i] = p.index(i)

    inverse permutation
    """
    p = np.asarray(p)
    invp = np.empty_like(p)
    for i in range(p.size):
        invp[p[i]] = i
    return invp


def pad_arrays_keep_invp(src):
    """
    用于结果验证的时候

    Pad arrays and return inverse permutation

    Input:
    src (list[array[int32]])
    ---
    Output:
    src (seq_len, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    idx = argsort(src) # [5, 7, 8, 9, 6, 1, 2, 0, 3, 4]
    src = list(np.array(src)[idx])
    lengths = list(map(len, src))  # [13, 13, 12, 12, 10, 5, 5, 4, 4, 3]
    lengths = torch.LongTensor(lengths)  
    # 对位补齐
    src = pad_arrays(src) 
    invp = torch.LongTensor(invpermute(idx)) # [7, 5, 6, 8, 9, 0, 4, 1, 2, 3]  
    # 使其contiguous()在内存中连续
    return src.t().contiguous(), lengths.view(1, -1), invp


def random_subseq(a, rate):
    """
    Dropping some points between a[1:-2] randomly according to rate.

    Input:
    a (array[int])
    rate (float)
    """
    idx = np.random.rand(len(a)) < rate
    idx[0], idx[-1] = True, True
    return a[idx]



