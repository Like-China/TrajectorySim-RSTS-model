"""
通过对空间和时间进行划分，将每个轨迹点转化为token值
通过读取h5文件中的轨迹，写出
data/city_name/cityname_regionScale_timeScale/train.src
/data/city_name/cityname_regionScale_timeScale/train1.trg
/data/city_name/cityname_regionScale_timeScale/train.mtc
/data/city_name/cityname_regionScale_timeScale/val.src
/data/city_name/cityname_regionScale_timeScale/val.trg
/data/city_name/cityname_regionScale_timeScale/val.mtc

同时写出用于实验表现验证的数据
exp_data_r1 为最多max_exp_trj_num条 [trip, ts, exact_trj] 记录
porto单核需要7小时左右
beijing大概两小时

对测试集，返回其所有的trips和ts用于生成实验用数据
exp_data_r1 下采样的实验用数据 exp_data_r1[0]为一批下采样率为0的原始轨迹，里面每个记录形如[trip, ts, tokens]
                            exp_data_r1[0]为一批下采样率为0.2的原始轨迹，里面每个记录形如[trip, ts, tokens]
                            ...
exp_data_r2 噪声偏移实验用数据 exp_data_r2[0]为一批噪声偏移率为0的原始轨迹，里面每个记录形如[trip, ts, tokens]
                            exp_data_r2[0]为一批噪声偏移率为0.2的原始轨迹，里面每个记录形如[trip, ts, tokens]
                            ...
"""
# -*- coding: utf-8 -*-
import random, h5py, os, warnings, gc, time, argparse
from tqdm import tqdm
import numpy as np
from scipy import spatial
warnings.filterwarnings("ignore")

data_path = "/home/Like/data/trajectory_similarity/"
h5_path = "/home/Like/data/"
min_len = 20
max_len = 100
hot_freq = 30
PAD = 0
BOS = 1
EOS = 2
UNK = 3


def setRegionArgs(city_name, scale, time_size):
    """
    parameter settings of space partition and time partition
    """
    parser = argparse.ArgumentParser(description="Region.py")
    if city_name[0] == 'b':
        lons_range, lats_range = [116.25, 116.55], [39.83, 40.03]
    else:
        lons_range, lats_range = [-8.735, -8.156], [40.953, 41.307]
    # space partition under specific scale
    maxx, maxy = (lons_range[1]-lons_range[0])//scale, (lats_range[1]-lats_range[0])//scale
    parser.add_argument("-lons", default= lons_range, help="range of longitude")
    parser.add_argument("-lats", default= lats_range, help="range of latitude")
    parser.add_argument("-maxx", type=int, default=maxx, help="最大横坐标空间编号")
    parser.add_argument("-maxy", type=int, default=maxy, help="最大纵坐标空间编号")
    parser.add_argument("-numx", type=int, default=maxx, help="空间上横块数")
    parser.add_argument("-numy", type=int, default=maxy, help="空间上纵块数")
    parser.add_argument("-space_cell_size", type=int, default=maxx*maxy, help="空间cell数")
    # time partition under specific number of time slices
    parser.add_argument("-time_span", type = int, default=86400 // time_size, help="每个时间段长度")
    # spatio-temporal cells
    parser.add_argument("-start", type = int, default = UNK+1, help="vocal word编码从4开始编号，0，1，2，3有特殊作用")
    parser.add_argument("-space_nn_nums", type = int, default = 20, help="编码一个时空格子时，空间上的近邻筛选个数， 用于生成V, D")
    parser.add_argument("-time_nn_nums", type = int, default = 10, help="编码一个时空格子时，时间上的近邻筛选个数，也是最终的时空近邻个数, 用于生成V, D")
    parser.add_argument("-map_cell_size", type = int, default = maxx*maxy*time_size, help="时空格子数目（x,y,t)三维")
    args = parser.parse_args()
    return args


class Region:
    """
    划分空间格子和时间格子
    对于每一个原始的（x,y,t)轨迹点，将其转化为对应的时空格子编码
    """

    def __init__(self,city_name, scale, time_size, max_trjs_num= 10000, max_exp_num=200000):
         
        self.args = setRegionArgs(city_name, scale, time_size)
        # 结果数据文件存放路径 ,存储src,trg, mta的文件路径
        self.save_path = os.path.join(data_path, city_name, city_name+str(int(scale*100000))+str(time_size))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # 读取轨迹 h5文件路径
        self.h5path = os.path.join(h5_path, city_name+".h5")
        # 读取的最大轨迹数目
        self.max_trjs_num = max_trjs_num
        # 输出V，Ds,Dt 的 path ./data/porto_dist.h5
        self.VDpath = os.path.join(self.save_path, "dist.h5")
        # 下采样率+产生噪声的概率
        self.dropping_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.distorting_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.max_exp_trj_num = max_exp_num
        # 记录 map_id 到 热度词 的 nn 映射
        self.mapId2nnword = {}
        # 空间top-k邻居选取数， 用于为不属于热度词中的mapID选择一个可以替换的最近热度词邻居
        self.space_nn_topK = 20
        print("参数设置：", self.args)

    def run(self):
        print("创建词汇表, 获得热度词映射")
        self.create_vocal()
        print("使用利用热度词经纬度构建的Kdtree+ 热度词的时间段, 计算所有热度词的经纬度，构建每个热度词在时空上的近邻V,D")
        self.create_dist()
        self.create_data()

    def create_vocal(self):
        '''
        建立词汇表，获得热度词及 热度词到词汇表的相互映射map
        2021/11/20 验证 计数和字典映射转化正确 + 热度词转化为经纬度转化正确
        '''
        # 建立字典对cell_id进行计数
        id_counter = {}
        # 对编码值进行计数
        with h5py.File(self.h5path, 'r') as f:
            read_trj_num = min(self.max_trjs_num, f.attrs['num']-2)
            # for ii in tqdm(range(read_trj_num)):
            for ii in range(read_trj_num):
                if ii % 10000 == 0: print("%d / %d"%(ii, read_trj_num))
                trip = np.array(f.get('trips/' + str(ii + 1)))
                ts = np.array(f.get('timestamps/' + str(ii + 1)))
                for (lon, lat), t in zip(trip, ts):
                    space_id = self.gps2spaceId(lon, lat)
                    t = int(t) // self.args.time_span
                    map_id = int(self.spaceId2mapId(space_id, t))
                    # 对每个map_id进行计数
                    id_counter[map_id] = id_counter.get(map_id, 0) + 1

        # 对计数从大到小排序
        self.sort_cnt = sorted(id_counter.items(), key=lambda count: count[1], reverse=True)
        # 查找 热度词， 即非低频词, 已经按照出现频词从大到小排序
        sort_cnt = self.sort_cnt
        self.hot_ids = [int(sort_cnt[i][0]) for i in range(len(sort_cnt)) if sort_cnt[i][1]>=hot_freq]
        # 建立从 cell_id 到真实的 map_cell_id 的词典映射, map_cell_id从4开始
        self.hotcell2word = {}
        self.word2hotcell = {}
        for ii in range(len(self.hot_ids)):
            self.hotcell2word[self.hot_ids[ii]] = ii+self.args.start
            self.word2hotcell[ii+self.args.start] = self.hot_ids[ii]
        # 有效词+4个特殊词构成词汇表总的大小
        self.vocal_nums = len(self.hot_ids)+4
        # 得到每个hotcell的 经纬度表示 和 所属时间段 （2021/11/20 验证准确性)
        self.hot_lonlats = [] # 用于建立kdtree
        self.hot_ts = [] # 用于在空间邻居的基础上继续查找时间上的邻居
        for map_id in self.hot_ids:
            space_id = self.mapId2spaceId(map_id)
            lon, lat = self.spaceId2gps(space_id)
            t = self.mapId2t(map_id) # 时间段
            self.hot_lonlats.append([round(lon,8),round(lat,8)])
            self.hot_ts.append(t)

        # 根据经纬度点建立Kdtree
        self.X = np.array(self.hot_lonlats)
        # 使用ckdtree快速检索每个空间点空间上最近的K个邻居, 正确性检测完毕，正确
        self.tree = spatial.cKDTree(data=self.X)
        print('词汇表建立完成，词汇表总量（包含4个特殊编码）：{}'.format(self.vocal_nums))

        # 清除回收所有无用变量
        del self.sort_cnt, self.hot_lonlats
        gc.collect()

    def create_dist(self):
        '''
        获得所有 hotcell 即 word 两两之间的距离， 构建 V, Ds, Dt
        运行较快, 2021/11/20 再次验证准确性
        V 每行为一个热度词的top K个时空热度词邻居
        '''
        # 从4开始，每个word间的space距离 第0行对应得是第4个word
        # 获取热度词之间空间上最接近的 space_nn_nums个邻居与它们之间的距离
        # D.shape (热度词数目, 邻居数目)
        D, V = self.tree.query(self.X, self.args.space_nn_nums)
        D, V = D.tolist(), V.tolist()
        # 从 space_nn_nums 个按照空间最近的邻居中， 再选一定数目个时间上最接近的
        all_nn_time_diff = []  # 时间上的距离
        all_nn_space_diff = [] # 空间上的距离
        all_nn_index = []
        # for ii in tqdm(range(len(self.hot_ts)), desc='create VD'):
        for ii in range(len(self.hot_ts)):
            if ii % 10000 == 0: print("%d / %d"%(ii, len(self.hot_ts)))
            # 编号为ii的空间邻居
            space_nn = V[ii]
            # 空间距离
            space_dist = D[ii]
            # 得到这几个space_nn的时间段
            nn_ts = np.array(self.hot_ts)[space_nn]
            # 得到与ii的时间差距
            line = abs(nn_ts - self.hot_ts[ii]).tolist()
            # 对第i个点到其他点的距离进行排序, 得到时间上的top nn
            id_dist_pair = list(enumerate(line))
            line.sort()
            id_dist_pair.sort(key=lambda x: x[1])
            # 获取每一个点的top-k索引编号
            top_k_index = [id_dist_pair[i][0] for i in range(self.args.time_nn_nums)]
            # 获取每一个点的top-k 距离
            top_k_time_diff = [line[i] for i in range(self.args.time_nn_nums)]
            top_k_dist_diff = [space_dist[ii] for ii in top_k_index]
            # 转换到space_nn上,注意所有V需要+4
            top_k = [space_nn[ii] + self.args.start for ii in top_k_index]

            all_nn_time_diff.append(top_k_time_diff)
            all_nn_space_diff.append(top_k_dist_diff)
            all_nn_index.append(top_k)

        # 构建加上0，1，2，3 (PAD, BOS, EOS, UNK) 号词后的距离V,D
        for ii in range(4):
            all_nn_time_diff.insert(ii, [0] * self.args.time_nn_nums)
            all_nn_space_diff.insert(ii, [0] * self.args.time_nn_nums)
            all_nn_index.insert(ii, [ii] * self.args.time_nn_nums)
        # 写入到V,D文件中， 将对应的写入到h5文件中
        f = h5py.File(self.VDpath, "w")
        # 存储时空上的K*time_size近邻, 空间距离以及时间距离
        # V.shape = (热度词个数+4)*时空上的邻居数目
        f["V"] = np.array(all_nn_index)
        f["Ds"] = np.array(all_nn_space_diff)
        f["Dt"] = np.array(all_nn_time_diff)

    def write(self, f,  train_num, val_num, isTrain):
        write_or_add = 'w'
        train_or_val = 'train' if isTrain else 'val'
        start = 0 if isTrain else train_num
        num = train_num if isTrain else val_num

        all_val_trips = []
        all_val_ts = []

        src_writer = open(os.path.join(self.save_path, '{}.src'.format(train_or_val)), write_or_add)
        trg_writer = open(os.path.join(self.save_path, '{}.trg'.format(train_or_val)), write_or_add)
        mta_writer = open(os.path.join(self.save_path, '{}.mta'.format(train_or_val)), write_or_add)

        for i in range(start, start+num):
            trip = np.array(f.get('trips/' + str(i + 1)))  # numpy n*2
            ts = np.array(f.get('timestamps/' + str(i + 1)), dtype=np.int32)  # numpy n*1

            # 用于处理porto数据集部分长度超出限制的情况
            if len(trip) > max_len:
                trips = []
                tss = []
                # 持续随机切割轨迹
                while len(trip) >= max_len:
                    # 将轨迹按min-max轨迹长度生成一个随机长度的轨迹
                    rand_len = np.random.randint(min_len, max_len)
                    trips.append(trip[0:rand_len])
                    trip = trip[rand_len:]

                    tss.append(ts[0:rand_len])
                    ts = ts[rand_len:]
                if len(trip) >= min_len:
                    trips.append(trip)
                    tss.append(ts)
            else:
                trips = [trip]
                tss = [ts]

            for trip, ts in zip(trips, tss):
                # 验证集数据单独分离出来，用于在实验评价中生成实验数据
                if not isTrain:
                    all_val_trips.append(trip.tolist())
                    all_val_ts.append(ts.tolist())
                mta = self.tripmeta(trip, ts)
                exact_trj = self.trip2words(trip, ts)  # 轨迹时空编码序列，全部为热度词，有效减少了token数目
                if len(exact_trj) < min_len or len(exact_trj) > max_len:
                    continue
                # 根据原轨迹, 只对发生下采样和偏移的位置点进行重新更新，增强效率
                noise_trips, noise_ts, noise_trjs = self.add_noise(exact_trj, trip, ts)
                if i % 1000 == 0 and i > 0:
                    print("生成进度：{}/{} ,{}".format(i, train_num+val_num, time.ctime()))
                # 写出编码序列值到txt文本
                for each in noise_trjs:
                    # write src
                    src_seq = ' '.join([str(id) for id in each])
                    src_writer.writelines(src_seq)
                    src_writer.write('\n')
                    # write trg
                    trg_seq = ' '.join([str(id) for id in exact_trj])
                    trg_writer.writelines(trg_seq)
                    trg_writer.write('\n')
                    # write mta
                    mta_writer.writelines(' '.join([str(mta[0])[0:7], str(mta[1])[0:7], str(mta[2])]))
                    mta_writer.write('\n')

        src_writer.close()
        trg_writer.close()
        mta_writer.close()

        return all_val_trips, all_val_ts

    def create_data(self, train_ratio=0.8):
        """
        划分训练集和测试集并写入文件 train.src/ train.trg/ val.trg/ val.src

        :param train_ratio: 训练集轨迹所占比例
        """
        # 划分训练集和测试集
        f = h5py.File(self.h5path, 'r')
        # 写出测试集的下采样+噪声偏移后的轨迹位置+时间戳信息文本
        trj_nums = min(self.max_trjs_num, f.attrs['num']-2)
        train_num = int(train_ratio * trj_nums)
        val_num = trj_nums - train_num
        # 生成训练集和测试集, 对测试集，返回其所有的trips和ts用于生成实验用数据
        self.write(f,  train_num, val_num, True)
        all_val_trips, all_val_ts = self.write(f,  train_num, val_num, False)
        # print("写出有效测试集trip数目为: ", len(all_val_trips))
        f.close()
        # 根据验证集数据，生成实验用数据
        print("生成实验用数据ing")
        self.generate_exp_data(all_val_trips, all_val_ts)

    def generate_exp_data(self, all_val_trips, all_val_ts):
        all_val_trips, all_val_ts = np.array(all_val_trips), np.array(all_val_ts)
        # 得到不同下采样率和噪声率的数据用于实验测量
        self.exp_data = []
        ii = 0
        max_num = min(len(all_val_trips), self.max_exp_trj_num)
        for trip, ts in zip(all_val_trips, all_val_ts):
            exact_trj = self.trip2words(trip, ts)
            self.exp_data.append([trip, ts, exact_trj])
            ii += 1
            if ii >= max_num:
                break

        self.all_res_r1, self.all_res_r2 = self.downsample_and_distort(self.exp_data)
        path1 = os.path.join(self.save_path, "exp_data_r1")
        path2 = os.path.join(self.save_path, "exp_data_r2")
        np.save(path1, self.all_res_r1, allow_pickle=True)
        np.save(path2, self.all_res_r2, allow_pickle=True)

    def downsample_and_distort(self, data):
        """
        输入data为 [trip, ts, exact_trj] 集合， 输出为各种下采样和distort后的数据
        :param data:
        :return:
        """
        mu = 0
        region_sigma = 0.005
        time_sigma = 300

        all_res_r1 = []
        for r1 in self.dropping_rates:
            res_r1 = []
            for (trip, ts, exact_trj) in data:
                trip, ts, exact_trj = np.array(trip),np.array(ts),np.array(exact_trj)
                randx = np.random.rand(len(trip)) > r1
                # 每条轨迹的起点和终点 设置为 不可下采样删除
                randx[0], randx[-1] = True, True
                sampled_trip = trip[randx]
                sampled_t = ts[randx]
                sampled_trj = np.array(exact_trj)[randx]
                res_r1.append([sampled_trip, sampled_t, sampled_trj])
            all_res_r1.append(res_r1)

        all_res_r2 = []
        for r2 in self.distorting_rates:
            res_r2 = []
            for (trip, ts, exact_trj) in data:
                trip, ts, exact_trj = np.array(trip),np.array(ts),np.array(exact_trj)
                randx = np.random.rand(len(trip)) < r2
                randx[0], randx[-1] = False, False
                trip[randx] = trip[randx] + random.gauss(mu, region_sigma)
                ts[randx] = ts[randx] + random.gauss(mu, time_sigma)
                # 只需要对需要产生噪声的位置点 编码进行重新更新
                exact_trj[randx] = self.trip2words(trip[randx], ts[randx])
                res_r2.append([trip, ts, exact_trj])
            all_res_r2.append(res_r2)
        return all_res_r1, all_res_r2

    def add_noise(self, exact_trj, trip, ts):
        """
        下采样+添加噪声， 只对发生下采样和偏移的位置点进行重新计算热度词编码
        2021/11/20 验证准确性完毕

        :param exact_trj: 原轨迹编码序列
        :param trip: 轨迹位置序列
        :param ts: 轨迹时间戳序列
        :return: 带噪声和下采样的轨迹编码序列集合
        """
        mu = 0
        region_sigma = 0.001
        time_sigma = 200

        # 存储位置序列，时间戳序列，编码值序列
        noise_trips = []
        noise_ts = []
        noise_trjs = []

        for dropping_rate in self.dropping_rates:
            randx = np.random.rand(len(trip))>dropping_rate
            # 每条轨迹的起点和终点 设置为 不可下采样删除
            randx[0], randx[-1] = True, True
            sampled_trip = trip[randx]
            sampled_t = ts[randx]
            sampled_trj = np.array(exact_trj)[randx]

            for distorting_rate in self.distorting_rates:
                # 随机选择会发生偏移的位置
                randx = np.random.rand(len(sampled_trip)) < distorting_rate
                # 每条轨迹的起点和终点不可发生噪声偏移
                randx[0], randx[-1] = False, False
                sampled_trip[randx] = sampled_trip[randx] + random.gauss(mu, region_sigma)
                sampled_t[randx] = sampled_t[randx]+random.gauss(mu,time_sigma)
                # 只需要对需要产生噪声的位置点 编码进行重新更新
                sampled_trj[randx] = self.trip2words(sampled_trip[randx], sampled_t[randx])
                noise_trips.append(sampled_trip)
                noise_ts.append(sampled_t)
                noise_trjs.append(sampled_trj)
        return noise_trips, noise_ts, noise_trjs

    def get_snn(self, long, lat, K):
        """
        获取一个 经纬度点 的 空间最近词

        :param long: 经度
        :param lat: 纬度
        :param K: top K
        :return:
        """
        point = np.array([long,lat])
        k_dists, k_indexs = self.tree.query(point, K, p=1)
        return k_dists, k_indexs+4

    def get_a_nn(self, map_id):
        """
        获得一个离 mapId最近的 hot word
        2021/11/20 检查正确性完毕

        :param map_id: 映射到的时空编码值
        :return: 与mapId最近的 hot word，也是一个时空编码值
        """
        # 得到非热度词的经纬度和时间
        space_id = self.mapId2spaceId(map_id)
        lon, lat = self.spaceId2gps(space_id)
        t = self.mapId2t(map_id)
        # 得到在热度词中的空间邻居, 为+4后的编码值
        k_dists, k_indexs = self.get_snn(lon, lat, self.space_nn_topK)
        # k_dists, k_indexs = k_dists.tolist(), k_indexs.tolist()
        # 选择时间最接近的热度词输出
        # 初始化一个最小的时间差距 和 一个最小邻居
        min_hot_t = 1000
        nn = UNK
        # 经验证以下找最近邻比利用array快
        for hot_word in k_indexs:
            # 注意这里需要-4来获取每个空间邻居所处的时间段
            if abs(self.hot_ts[hot_word-self.args.start]-t) < min_hot_t:
                min_hot_t = abs(self.hot_ts[hot_word-self.args.start]-t)
                nn = hot_word
        return nn
    '''
    ****************************************************************************************************************************************************
    一系列的转化函数
    ****************************************************************************************************************************************************
    '''

    def lonlat2xyoffset(self, lon, lat):
        '''经纬度转换为米为单位, 映射到平面图上 (116.3, 40.0)->(4,8)'''
        xoffset = round((lon - self.args.lons[0]) / scale)
        yoffset = round((lat - self.args.lats[0]) / scale)
        return int(xoffset), int(yoffset)

    def xyoffset2lonlat(self, xoffset, yoffset):
        ''' 米单位转换为经纬度  (4,8)-> (116.3, 40.0)'''
        lon = self.args.lons[0]+xoffset*scale
        lat = self.args.lats[0]+yoffset*scale
        return lon,lat

    def offset2spaceId(self, xoffset, yoffset):
        ''' (xoffset,yoffset) -> space_cell_id  (4,8)->116'''
        return int(yoffset * self.args.numx + xoffset)

    def spaceId2offset(self, space_cell_id):
        ''' space_cell_id -->(x,y) 116->(4.8)'''
        yoffset = space_cell_id // self.args.numx
        xoffset = space_cell_id % self.args.numx
        return int(xoffset), int(yoffset)

    def gps2spaceId(self, lon, lat):
        ''' gps--> space_cell_id  116.3,40->116'''
        xoffset, yoffset = self.lonlat2xyoffset(lon, lat)
        space_cell_id = self.offset2spaceId(xoffset, yoffset)
        return int(space_cell_id)

    def spaceId2gps(self, space_cell_id):
        '''space_cell_id -->gps 116->116.3,40'''
        xoffset, yoffset = self.spaceId2offset(space_cell_id)
        lon,lat = self.xyoffset2lonlat(xoffset,yoffset)
        return lon,lat

    def spaceId2mapId(self, space_id, t):
        ''' space_cell_id+t --> map_id  116,10->1796'''
        return int(space_id + t*self.args.space_cell_size)

    def mapId2spaceId(self, map_id):
        ''' map_id -->space_cell_id  1796-> 116'''
        return int(map_id % self.args.space_cell_size)

    def mapId2t(self, map_id):
        ''' map_id -->t 1796-> 10'''
        return int(map_id // self.args.space_cell_size)

    def mapId2word(self, map_id):
        ''' map_id -> vocal_id 若不在热度词中，则用与其较近的词代替 '''
        word = self.hotcell2word.get(map_id, UNK)
        return word if word != UNK else self.get_a_nn(map_id)

    def word2mapId(self, word):
        ''' word -> map_id 不会存在查找不到的情况'''
        return self.word2hotcell.get(word, 0)

    def word2xyt(self, word):
        ''' word --> xoffset,yoffset,t '''
        map_id = self.word2mapId(word)
        t = self.mapId2t(map_id)
        space_id = self.mapId2spaceId(map_id)
        xoffset,yoffset = self.spaceId2offset(space_id)
        return xoffset,yoffset,t

    def xyt2word(self,x,y,t):
        ''' xoffset,yoffset,t -->word '''
        space_id = self.offset2spaceId(x, y);
        map_id = self.spaceId2mapId(space_id, t)
        word = self.mapId2word(map_id)
        return word

    def mapIds2words(self, map_ids):
        ''' map_ids --> words'''
        words = [self.mapId2word(id) for id in map_ids]
        return words

    def words2mapIds(self, words):
        '''words -+--> map_ids'''
        map_ids = [self.word2hotcell.get(id, 0) for id in words]
        return map_ids

    def trip2spaceIds(self, trip):
        ''' trip --> space_cell_ids '''
        space_ids = []
        for lonlat in trip:
            space_id = self.gps2spaceId(lonlat[0], lonlat[1])
            space_ids.append(space_id)
        return space_ids

    def trip2mapIDs(self, trip, ts):
        ''' trip --> space_cell_ids --> map_ids'''
        map_ids = []
        for (lon, lat), t in zip(trip, ts):
            space_id = self.gps2spaceId(lon, lat)
            t = int(t) // self.args.time_span
            map_id = self.spaceId2mapId(space_id, t)
            map_ids.append(map_id)
        return list(map_ids)

    def trip2words(self, trip, ts):
        ''' 减少迭代次数的trip2words '''
        words = []
        for (lon, lat), t in zip(trip, ts):
            space_id = self.gps2spaceId(lon, lat)
            t = int(t) // self.args.time_span
            map_id = self.spaceId2mapId(space_id, t)
            words.append(self.mapId2word(map_id))
        return words
    
    def tripmeta(self, trip, ts):
        ''' 得到一个trip的meta 经纬度重心坐标'''
        long_min = min([p[0] for p in trip])
        long_max = max([p[0] for p in trip])
        lat_min = min([p[1] for p in trip])
        lat_max = max([p[1] for p in trip])
        min_ts = min(ts)
        max_ts = max(ts)
        ts_centroids = min_ts+(max_ts-min_ts)/2
        
        long_centroids = long_min + (long_max - long_min) / 2
        lat_centroids = lat_min + (lat_max - lat_min) / 2
        return round(long_centroids, 8), round(lat_centroids, 8), round(ts_centroids, 4)


if __name__ == "__main__":
    city_name, scale, time_size = "porto", 0.001, 300
    print(city_name, scale, time_size)
    # 默认生成20万条轨迹用于实验评估
    r = Region(city_name, scale, time_size, max_trjs_num=1000000000, max_exp_num=200000)
    r.run()

        
    