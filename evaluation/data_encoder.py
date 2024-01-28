"""
获取实验原始数据， 通过模型将其转换为表征向量
Region.py中对porto生成了超20万条数据，而对于beijing只生成了37472条
"""
import numpy as np
import os
import random
from evaluator.trj2vec import t2vec_input
from settings import set_args


class ExpLoader:

    def __init__(self):
        self.args = set_args()
        self.r1_path = os.path.join(self.args.data, "exp_data_r1.npy")
        self.r2_path = os.path.join(self.args.data, "exp_data_r2.npy")
        print("读取数据...")
        self.r1_data = np.load(self.r1_path, allow_pickle=True)  # 采样率个数*选取的最大轨迹数目 个 [trip, ts, trj_tokens]
        self.r2_data = np.load(self.r2_path, allow_pickle=True)  # 噪声率个数*选取的最大轨迹数目 个 [trip, ts, trj_tokens]
        print("r1 load finished, number of sampling rates: %d  number of trajectories: %d" % (len(self.r1_data),
              len(self.r1_data[0])))
        print("r2 load finished, number of distorting rates: %d  number of trajectories: %d" % (len(self.r2_data),
              len(self.r2_data[0])))
        # 采样率，噪声率
        self.rs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # 限制下采样后的轨迹最小长度
        self.min_len = 10

        # mean rank data
        self.P0_trips, self.P1_trips, self.Q0_trips, self.Q1_trips = [], [], [], []
        self.P0_ts, self.P1_ts, self.Q0_ts, self.Q1_ts = [], [], [], []
        self.P0_tokens, self.P1_tokens, self.Q0_tokens, self.Q1_tokens = [], [], [], []

        # knn data
        self.query0_trips, self.query1_trips, self.db0_trips, self.db1_trips = [], [], [], []
        self.query0_ts, self.query1_ts, self.db0_ts, self.db1_ts = [], [], [], []

    def mean_rank_data(self, q_num, p_num, r, is_drop):
        """
        获取一批 p, q 用于计算 mean_rank
        1. 获取特定采样率/噪声率下的一组[trip, ts, tokens]数据
        2. 随机打乱该数据
        3. 选择特定数目的记录形成p,q
        4. 交叉采样形成p0,  p1, q0, q1
        :param q_num: 轨迹集q的数目
        :param p_num: 轨迹集p的数目
        :param r: 下采样率或噪声偏移率
        :param is_drop: True 进行下采样 False 进行噪声偏移
        :return: 编码后的表征向量
        """
        # 每次生成都需要重新初始化，防止上次生成的叠加
        self.P0_tokens, self.P1_tokens, self.Q0_tokens, self.Q1_tokens = [], [], [], []
        self.P0_ts, self.P1_ts, self.Q0_ts, self.Q1_ts = [], [], [], []
        self.P0_trips, self.P1_trips, self.Q0_trips, self.Q1_trips = [], [], [], []
        # 获取对应采样率下的数据
        ix = self.rs.index(r)
        data = self.r1_data[ix] if is_drop else self.r2_data[ix]

        # 随机打乱数据
        indexs = random.sample(range(len(data)), len(data))
        data = np.array(data)[indexs].tolist()
        while len(data) < q_num + p_num:
            data.extend(data)
        # 获取指定数目的p,q两组不同的轨迹集合
        q = data[0:q_num]
        p = data[q_num:q_num + p_num]

        # 交叉采样q形成q0, q1, 保证Q0-Q1, P0-P1个数相同
        for trip, ts, tokens in q:
            assert len(trip) == len(ts), "Unequal length"
            if trip[0][0]-trip[-1][0] <= 0.0001 and trip[0][1]-trip[-1][1] <= 0.0001:
                continue
            trip1 = [trip[ii] for ii in range(len(trip)) if ii % 2 == 0]
            trip2 = [trip[ii] for ii in range(len(trip)) if ii % 2 == 1]

            ts1 = [ts[ii] for ii in range(len(ts)) if ii % 2 == 0]
            ts2 = [ts[ii] for ii in range(len(ts)) if ii % 2 == 1]

            tokens1 = [tokens[ii] for ii in range(len(tokens)) if ii % 2 == 0]
            tokens2 = [tokens[ii] for ii in range(len(tokens)) if ii % 2 == 1]

            if len(tokens1) > self.min_len:
                self.Q0_tokens.append(tokens1)
                self.Q1_tokens.append(tokens2)
                self.Q0_ts.append(ts1)
                self.Q1_ts.append(ts2)
                self.Q0_trips.append(trip1)
                self.Q1_trips.append(trip2)

        # 交叉采样p形成p0, p1
        for trip, ts, tokens in p:
            assert len(trip) == len(ts) == len(tokens), "Unequal length"
            if trip[0][0]-trip[-1][0] <= 0.0001 and trip[0][1]-trip[-1][1] <= 0.0001:
                continue
            trip1 = [trip[ii] for ii in range(len(trip)) if ii % 2 == 0]
            trip2 = [trip[ii] for ii in range(len(trip)) if ii % 2 == 1]

            ts1 = [ts[ii] for ii in range(len(ts)) if ii % 2 == 0]
            ts2 = [ts[ii] for ii in range(len(ts)) if ii % 2 == 1]

            tokens1 = [tokens[ii] for ii in range(len(tokens)) if ii % 2 == 0]
            tokens2 = [tokens[ii] for ii in range(len(tokens)) if ii % 2 == 1]

            if len(tokens1) > self.min_len:
                self.P0_tokens.append(tokens1)
                self.P1_tokens.append(tokens2)
                self.P0_ts.append(ts1)
                self.P1_ts.append(ts2)
                self.P0_trips.append(trip1)
                self.P1_trips.append(trip2)

        # 将轨迹向量转化为 表征向量
        p0_vecs = t2vec_input(self.args, self.P0_tokens, None)
        p1_vecs = t2vec_input(self.args, self.P1_tokens, None)
        q0_vecs = t2vec_input(self.args, self.Q0_tokens, None)
        q1_vecs = t2vec_input(self.args, self.Q1_tokens, None)
        return q0_vecs, q1_vecs, p0_vecs, p1_vecs

    def knn_data(self, query_num, db_num, r, is_drop):
        """
        获取一批Tb, Tb', Ta, Ta'数据用于计算cross similarity (query_num, db_num 数目相同）

        获取一批 query, db数据 用于计算 knn 准确率 （query_num, db_num 数目不同)

        :param query_num: query轨迹集数目
        :param db_num: db轨迹集数目
        :param r: 下采样率或噪声偏移率
        :param is_drop: True 进行下采样 False 进行噪声偏移
        :return: 编码后的表征向量
        """
        self.query0_trips, self.query1_trips, self.db0_trips, self.db1_trips = [], [], [], []
        self.query0_ts, self.query1_ts, self.db0_ts, self.db1_ts = [], [], [], []
        ix = self.rs.index(r)
        # 获取对应采样率下的数据
        # 原轨迹数据，未下采样，未添加噪声
        data1 = self.r1_data[0] if is_drop else self.r2_data[0]
        # 有下采样或噪声偏移的轨迹
        data2 = self.r1_data[ix] if is_drop else self.r2_data[ix]
        data1 = data1.tolist()
        data2 = data2.tolist()
        while len(data1) < query_num + db_num:
            data1.extend(data1)
            data2.extend(data2)
        # 随机选择一定数目的轨迹
        indexs = random.sample(range(len(data1)), query_num + db_num)
        data1 = np.array(data1)[indexs].tolist()
        data2 = np.array(data2)[indexs].tolist()

        query1_tokens, query2_tokens, db1_tokens, db2_tokens = [], [], [], []
        for ii in range(query_num + db_num):
            trip1 = data1[ii][0]
            trip2 = data2[ii][0]
            if trip1[0][0]-trip1[-1][0] <= 0.0001 and trip1[0][1]-trip1[-1][1] <= 0.0001:
                continue
            if trip2[0][0]-trip2[-1][0] <= 0.0001 and trip2[0][1]-trip2[-1][1] <= 0.0001:
                continue
            if len(trip1) > self.min_len and len(trip2) > self.min_len:
                if ii < query_num:
                    self.query0_trips.append(data1[ii][0])
                    self.query1_trips.append(data2[ii][0])
                    self.query0_ts.append(data1[ii][1])
                    self.query1_ts.append(data2[ii][1])
                    query1_tokens.append(data1[ii][2])
                    query2_tokens.append(data2[ii][2])
                else:
                    self.db0_trips.append(data1[ii][0])
                    self.db1_trips.append(data2[ii][0])
                    self.db0_ts.append(data1[ii][1])
                    self.db1_ts.append(data2[ii][1])
                    db1_tokens.append(data1[ii][2])
                    db2_tokens.append(data2[ii][2])

        # 将轨迹向量转化为 表征向量
        query1_vecs = t2vec_input(self.args, query1_tokens, None)
        query2_vecs = t2vec_input(self.args, query2_tokens, None)
        db1_vecs = t2vec_input(self.args, db1_tokens, None)
        db2_vecs = t2vec_input(self.args, db2_tokens, None)
        return query1_vecs, query2_vecs, db1_vecs, db2_vecs


if __name__ == "__main__":
    # 经验证能生成正确的实验评估用轨迹
    loader = ExpLoader()
    p0, p1, q0, q1 = loader.mean_rank_data(200, 400, 0.3, False)  # distort
    query0, query1, db1, db2 = loader.knn_data(1000, 4000, 0.3, True)  # drop
    # 检验mean_rank实验数据生成是否正确
    print(len(loader.Q0_tokens), len(loader.Q1_tokens), len(loader.P0_tokens), len(loader.P1_tokens))
    print(loader.P0_ts[0], len(loader.P0_ts[0]))
    print(loader.P1_ts[0], len(loader.P1_ts[0]))
    print(loader.Q0_ts[0], len(loader.Q0_ts[0]))
    print(loader.Q1_ts[0], len(loader.Q1_ts[0]))
    print("\n")
    # 检验cs, knn实验数据生成是否正确
    print(len(query0), len(query1), len(db1), len(db2))
    print(loader.P0_trips[0], len(loader.P0_trips[0]))
    print(loader.P1_trips[0], len(loader.P1_trips[0]))

    print(loader.P0_tokens[0], len(loader.P0_tokens[0]))
    print(loader.P1_tokens[0], len(loader.P1_tokens[0]))

