import argparse, os, torch, warnings, h5py
warnings.filterwarnings("ignore")
os.environ['CUDA_ENABLE_DEVICES'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
torch.backends.cudnn.enabled = False
from torch.backends import cudnn
cudnn.benchmark = True

"Porto原来有轨迹17万条，经过拆分+下采样/噪声偏移，训练集有约4800万轨迹，测试1200万轨迹,读取耗时一小时" \
"Beijing原有1万条轨迹，处理后有540万条训练集，135万测试集"
PAD = 0
BOS = 1
EOS = 2
UNK = 3
# limit the length of trajectories
min_len = 20
max_len = 100
# city_name = "beijing"
city_name = "porto_randTime"
scale = 0.001  # "beijing 0.001-200 Porto 0.001-300"
time_size = 300
hot_freq = 30
data_path = "/home/Like/data/trajectory_similarity/"
h5_path = "/home/Like/data/"
vector_size = 32


def set_args(is_train=True):
    # 从V矩阵获取对应区域划分下的热度词数 （热度词个数+4个特殊占位词，PAD, BOS, EOS, UNK)
    dist_path = os.path.join(data_path, city_name, city_name + str(int(scale * 100000)) + str(time_size), 'dist.h5')
    print(dist_path)
    if is_train:
        with h5py.File(dist_path, "r") as f:
            v = f["V"][...]
        vocal_size = len(v)
        print("The number of vocal: ", vocal_size)
    else:
        vocal_size = 0
    parser = argparse.ArgumentParser(description="train.py")
    '''
    *************************************************************************
    训练数据参数
    *************************************************************************
    '''
    parser.add_argument("-read_train_num", type=int, default=5000000, help='the number of training data')
    parser.add_argument("-read_val_num", type=int, default=1000000, help="the number of validate data")
    parser.add_argument("-iter_num", default=3000000, help="the number of iteration")
    parser.add_argument("-max_length", default=max_len, help="The maximum length of the target sequence")
    parser.add_argument("-bucket_size", default=[(30, 50), (50, 70), (80, 100)], help="Bucket size for training")
    '''
    *************************************************************************
     Region参数
    *************************************************************************
    '''
    parser.add_argument("-city_name", default=city_name, help="city name")
    parser.add_argument("-scale", type=float, default=scale, help="city scale")
    parser.add_argument("-time_size", type=float, default=time_size, help="time span nums")
    parser.add_argument("-data", default=os.path.join(data_path, city_name, city_name + str(int(scale * 100000))
                                                      + str(time_size)), help="训练集和模型存储目录")
    parser.add_argument("-checkpoint", default=os.path.join(data_path, city_name, city_name + str(int(scale * 100000)) +
                                                            str(time_size), 'checkpoint.pt'), help="checkpoint存放目录")
    parser.add_argument("-knn_vocabs", default=dist_path, help="dist VD存放目录")
    parser.add_argument("-hot_freq", type=int, default=hot_freq, help="hot cell frequency, 计数出现这么多次认为是热度词")
    '''
    *************************************************************************
    神经网络层参数
    *************************************************************************
    '''
    parser.add_argument("-timeWeight", default=0.2, help="The weight used to evaluate the temporal similarity")
    parser.add_argument("-vocab_size", default=vocal_size, help="词汇表数目，从V, D的size上获取")
    parser.add_argument("-need_rank_val", default=False, help="是否需要rank验证")
    # 3090-1 不要三元损失 3090-2 要三元损失
    parser.add_argument("-use_discriminative", action="store_true", default=True, help="是否使用三元判别损失")
    parser.add_argument("-discriminative_w", type=float, default=0.5, help="discriminative loss weight")
    parser.add_argument("-dis_freq", type=float, default=10, help="三元损失使用频率")
    parser.add_argument("-prefix", default="exp", help="存储轨迹文件的前缀")
    parser.add_argument("-num_layers", type=int, default=3, help="Number of layers in the RNN cell")
    parser.add_argument("-bidirectional", type=bool, default=True, help="True if use bidirectional rnn in encoder")
    parser.add_argument("-hidden_size", type=int, default=vector_size, help="The hidden state size in the RNN cell")
    parser.add_argument("-embedding_size", type=int, default=vector_size, help="The word (cell) embedding size")
    parser.add_argument("-dropout", type=float, default=0.2, help="The dropout probability")
    parser.add_argument("-max_grad_norm", type=float, default=5.0, help="The maximum gradient norm")
    parser.add_argument("-learning_rate", type=float, default=0.001)
    # beijing batch=120 3090拉满  porto 100拉满
    parser.add_argument("-batch", type=int, default=30, help="The batch size")
    parser.add_argument("-generator_batch", type=int, default=256, help="每次生成多少个词，消耗内存")
    parser.add_argument("-t2vec_batch", type=int, default=256, help="每一批最大编码轨迹数目")
    parser.add_argument("-start_iteration", type=int, default=0)
    parser.add_argument("-epochs", type=int, default=15, help="The number of training epochs")
    parser.add_argument("-print_freq", type=int, default=1000, help="Print frequency")
    parser.add_argument("-save_freq", type=int, default=5000, help="Save frequency")
    parser.add_argument("-cuda", type=bool, default=True, help="True if we use GPU to train the model")
    parser.add_argument("-criterion_name", default="KLDIV",
                        help="NLL (Negative Log Likelihood) or KLDIV (KL Divergence)")
    parser.add_argument("-dist_decay_speed", type=float, default=0.8)
    args = parser.parse_args()
    return args


