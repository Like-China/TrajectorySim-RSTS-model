"""
用于包装训练集和测试集
随机读取训练集轨迹进行训练
"""
from funcy import merge
from loader.data_utils import *
from tqdm import tqdm
import settings


class DataLoader:
    """
    src_file: source file name
    trg_file: target file name
    batch_size: batch_size size
    validate: if validate = True return batch_size orderly otherwise return batch_size randomly
    """
    def __init__(self, src_file, trg_file, mta_file, batch_size, bucket_size, validate=False):
        self.src_file = src_file
        self.trg_file = trg_file
        self.mta_file = mta_file
        self.batch_size = batch_size
        self.validate = validate
        self.bucket_size = bucket_size
        # 记录轨迹信息
        self.src_data = []
        self.trg_data = []
        self.mta_data = []
        self.allocation = []
        self.p = []
        # 用于批次获取测试集
        self.start = 0
        self.size = 0
        # 记录最大最小轨迹位置点id, 共有多少条轨迹
        self.minID = 100
        self.maxID = 0

    # 插入8组不同的轨迹长度范围的轨迹到轨迹中，对于每一条src轨迹和目标轨迹，判断它们的长度并加入到到对应列表中
    def insert(self, s, t, m):
        for i in range(len(self.bucket_size)):
            if len(s) <= self.bucket_size[i][0] and len(t) <= self.bucket_size[i][1]:
                self.src_data[i].append(np.array(s, dtype=np.int32))
                self.trg_data[i].append(np.array(t, dtype=np.int32))
                self.mta_data[i].append(np.array(m, dtype=np.float32))
                return 1
        return 0

    # 加载固定数目的轨迹
    def load(self, max_num_line=0):
        self.src_data = [[] for _ in range(len(self.bucket_size))]
        self.trg_data = [[] for _ in range(len(self.bucket_size))]
        self.mta_data = [[] for _ in range(len(self.bucket_size))]

        src_stream, trg_stream, mta_stream = open(self.src_file, 'r'), open(self.trg_file, 'r'), open(self.mta_file, 'r')
        num_line = 0
        with tqdm(total=max_num_line, desc='Reading Traj', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            for (s, t, m) in zip(src_stream, trg_stream, mta_stream):
                s = [int(x) for x in s.split()]
                t = [settings.BOS] + [int(x) for x in t.split()] + [settings.EOS]
                m = [float(x) for x in m.split()]
                if len(s) > 0:
                    num_line += self.insert(s, t, m)
                    pbar.update(1)
                    self.maxID = max(max(s), self.maxID)
                    self.minID = min(min(s), self.minID)
                if num_line >= max_num_line > 0:
                    break
        # if validate  we merge all buckets into one
        if self.validate:
            self.src_data = np.array(merge(*self.src_data))
            self.trg_data = np.array(merge(*self.trg_data))
            self.mta_data = np.array(merge(*self.mta_data))
            self.start = 0
        else:
            self.src_data = list(map(np.array, self.src_data))
            self.trg_data = list(map(np.array, self.trg_data))
            self.mta_data = list(map(np.array, self.mta_data))
            self.allocation = list(map(len, self.src_data))
            self.p = np.array(self.allocation) / sum(self.allocation)
        self.size = len(self.src_data)
        src_stream.close(), trg_stream.close(), mta_stream.close()

    def get_batch(self):
        """
        加载一批轨迹，验证集有序加载，训练集随机加载
        :return:
        """
        if self.validate:
            src = self.src_data[self.start:self.start+self.batch_size]
            trg = self.trg_data[self.start:self.start+self.batch_size]
            mta = self.mta_data[self.start:self.start+self.batch_size]
            # update `start` for next batch_size
            self.start += self.batch_size
            if self.start >= self.size:
                self.start = 0
            return list(src), list(trg), list(mta)
        else:
            # select bucket
            sample = np.random.multinomial(1, self.p)
            bucket = np.nonzero(sample)[0][0]
            # select data from the bucket
            idx = np.random.choice(len(self.src_data[bucket]), self.batch_size)
            src = self.src_data[bucket][idx]
            trg = self.trg_data[bucket][idx]
            mta = self.mta_data[bucket][idx]
            return list(src), list(trg), list(mta)

    def get_batch_generative(self):
        """
        返回一组batch个数的 TF对象，排序加补位操作

        :return: ['src', 'lengths', 'trg', 'invp']
        """
        src, trg, _ = self.get_batch()
        # src (seq_len1, batch_size), lengths (1, batch_size), trg (seq_len2, batch_size)
        return pad_arrays_pair(src, trg, keep_invp=False)

    def get_apn_cross(self):
        """
        得到三个batch个数的轨迹集，a,p，n
        a中的轨迹中心更接近于p中的轨迹
        :return: 选取的一组a, p, n， 每个均为一个TF对象 ['src', 'lengths', 'trg', 'invp']
        """
        def distance(x, y):
            return np.linalg.norm(x[0:2]-y[0:2])
            # return 0.5*np.linalg.norm(x[:2] - y[:2])+ (x[2:3] - y[2:3])*0.5/24
        a_src, a_trg, a_mta = self.get_batch()
        p_src, p_trg, p_mta = self.get_batch()
        n_src, n_trg, n_mta = self.get_batch()

        #  p_src, p_trg, p_mta = copy.deepcopy(p_src), copy.deepcopy(p_trg), copy.deepcopy(p_mta)
        #  n_src, n_trg, n_mta = copy.deepcopy(n_src), copy.deepcopy(n_trg), copy.deepcopy(n_mta)
        for i in range(len(a_src)):
            # a_mta[i] float32[] [id, t]
            # 如果a,p两个轨迹距离更大，则将p中的轨迹换为n的轨迹
            if distance(a_mta[i], p_mta[i]) > distance(a_mta[i], n_mta[i]):
                p_src[i], n_src[i] = n_src[i], p_src[i]
                p_trg[i], n_trg[i] = n_trg[i], p_trg[i]
                p_mta[i], n_mta[i] = n_mta[i], p_mta[i]

        a = pad_arrays_pair(a_src, a_trg, keep_invp=True)
        p = pad_arrays_pair(p_src, p_trg, keep_invp=True)
        n = pad_arrays_pair(n_src, n_trg, keep_invp=True)
        return a, p, n

    def get_apn_inner(self):
        """
        以一定概率去除一批batch个数轨迹中的点后生成三个轨迹集合a, p，n

        Test Case:
        a, p, n = dataloader.getbatch_discriminative_inner()
        i = 2
        idx_a = torch.nonzero(a[2].t()[a[3]][i])
        idx_p = torch.nonzero(p[2].t()[p[3]][i])
        idx_n = torch.nonzero(n[2].t()[n[3]][i])
        a_t = a[2].t()[a[3]][i][idx_a].view(-1).numpy()
        p_t = p[2].t()[p[3]][i][idx_p].view(-1).numpy()
        n_t = n[2].t()[n[3]][i][idx_n].view(-1).numpy()
        print(len(np.intersect1d(a_t, p_t)))
        print(len(np.intersect1d(a_t, n_t)))
        """
        a_src, a_trg = [], []
        p_src, p_trg = [], []
        n_src, n_trg = [], []

        _, trgs, _ = self.get_batch()
        for i in range(len(trgs)):
            trg = trgs[i][1:-1]
            if len(trg) < 10: continue
            a1, a3, a5 = 0, len(trg)//2, len(trg)
            a2, a4 = (a1 + a3)//2, (a3 + a5)//2
            rate = np.random.choice([0.5, 0.6, 0.8])
            if np.random.rand() > 0.5:
                a_src.append(random_subseq(trg[a1:a4], rate))
                a_trg.append(np.r_[settings.BOS, trg[a1:a4], settings.EOS])
                p_src.append(random_subseq(trg[a2:a5], rate))
                p_trg.append(np.r_[settings.BOS, trg[a2:a5], settings.EOS])
                n_src.append(random_subseq(trg[a3:a5], rate))
                n_trg.append(np.r_[settings.BOS, trg[a3:a5], settings.EOS])
            else:
                a_src.append(random_subseq(trg[a2:a5], rate))
                a_trg.append(np.r_[settings.BOS, trg[a2:a5], settings.EOS])
                p_src.append(random_subseq(trg[a1:a4], rate))
                p_trg.append(np.r_[settings.BOS, trg[a1:a4], settings.EOS])
                n_src.append(random_subseq(trg[a1:a3], rate))
                n_trg.append(np.r_[settings.BOS, trg[a1:a3], settings.EOS])
        a = pad_arrays_pair(a_src, a_trg, keep_invp=True)
        p = pad_arrays_pair(p_src, p_trg, keep_invp=True)
        n = pad_arrays_pair(n_src, n_trg, keep_invp=True)
        return a, p, n

