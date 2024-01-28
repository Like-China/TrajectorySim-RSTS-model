
import os
from loader.data_scaner import DataOrderScaner
from loader.data_utils import pad_arrays_keep_invp
from trainer.train import EncoderDecoder
import torch
"""
输入轨迹向量, 利用训练好的模型得到一组代表向量
"""


def t2vec(args, trj_file_path, m0):
    """
    读取trj.t中的轨迹，返回最后一层输出作为向量表示, 函数内部自动加载模型

    :param args: 参数设置
    :param trj_file_path: 需要转换为向量表示的轨迹文件
    :param m0: 若是None则从checkpoint中自动加载
    :return: decoder的最后一层输出，
      为该组轨迹的向量表示 batch*向量维度 格式：列表
    """
    if m0 is None:
        m0 = EncoderDecoder(args)
        # 加载训练模型
        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            m0.load_state_dict(checkpoint["m0"])
            if torch.cuda.is_available():
                m0.cuda()
            # 不启用dropout和BN
            m0.eval()
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))
            return

    vecs = []
    scaner = DataOrderScaner(os.path.join(args.data, trj_file_path), args.t2vec_batch)
    scaner.load()

    while True:
        # src 该组最大轨迹长度*num_seqs(该组轨迹个数)
        src, lengths, invp = scaner.getbatch()
        if src is None: break
        if torch.cuda.is_available():
            src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
        h, _ = m0.encoder(src, lengths)  # 【层数*双向2，该组轨迹个数，隐藏层数】【6，10，128】
        # (num_layers, batch, hidden_size * num_directions) 【3，10，256】
        h = m0.encoder_hn2decoder_h0(h)
        # (batch, num_layers, hidden_size * num_directions) 【10，3，256】
        h = h.transpose(0, 1).contiguous()
        # (batch, *)
        # h = h.view(h.size(0), -1)
        vecs.append(h[invp].cpu().data)
    # (num_seqs, num_layers, hidden_size * num_directions)
    vecs = torch.cat(vecs)  # [10,3,256]
    # (num_layers, num_seqs, hidden_size * num_directions)
    vecs = vecs.transpose(0, 1).contiguous()  # [3,10,256]

    # 返回最后一层作为该批次 轨迹的代表
    return vecs[m0.num_layers - 1].squeeze(0).numpy().tolist()


def t2vec_input(args, trj_vector, m0):
    """
    给定一组轨迹向量T,  返回一批向量表征, 不需要手动加载训练模型
    
    :param args: 参数设置
    :param trj_vector: 一组轨迹向量
    :param m0: 若是None则从checkpoint中自动加载
    :return: 一批向量表征
    """
    if m0 is None:
        m0 = EncoderDecoder(args)
        # 加载训练模型
        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            m0.load_state_dict(checkpoint["m0"], strict=False)
            if torch.cuda.is_available():
                m0.cuda()
            # 不启用dropout和BN
            m0.eval()
        else:
            print("不存在模型")
            return

    vecs = []
    for ii in range(len(trj_vector) // args.t2vec_batch + 1):
        # src 该组最大轨迹长度*num_seqs(该组轨迹个数)
        src, lengths, invp = pad_arrays_keep_invp(trj_vector[ii * args.t2vec_batch:(ii + 1) * args.t2vec_batch])
        if src is None: break
        if torch.cuda.is_available():
            src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
        h, _ = m0.encoder(src, lengths)  # 【层数*双向2，该组轨迹个数，隐藏层数】【6，10，128】
        # (num_layers, batch, hidden_size * num_directions) 【3，10，256】
        h = m0.encoder_hn2decoder_h0(h)
        # (batch, num_layers, hidden_size * num_directions) 【10，3，256】
        h = h.transpose(0, 1).contiguous()
        # (batch, *)
        # h = h.view(h.size(0), -1)
        vecs.append(h[invp].cpu().data)
    # (num_seqs, num_layers, hidden_size * num_directions)

    vecs = torch.cat(vecs)  # [10,3,256]
    # # 存储三层 输出的隐藏层结构，每一层是 batch个256维的向量
    # (num_layers, num_seqs, hidden_size * num_directions)
    vecs = vecs.transpose(0, 1).contiguous()  # [3,10,256]
    return vecs[m0.num_layers - 1].squeeze(0).numpy()

