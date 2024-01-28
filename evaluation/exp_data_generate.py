"""
Created on Mon Dec 28 10:54:27 2020
比较实验
@author: likem
生成用时:  826.391074180603
"""
# -*- coding: utf-8 -*-
import numpy as np
import torch
from evaluator.data_encoder import ExpLoader
import os
from settings import set_args
import settings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def data_transfer(trips, ts):
    """
    将trips, ts转换为[lon, lat, t]轨迹文件用于计算EDR,EDwP
    """
    res = []
    for ii in range(len(trips)):
        trj = []
        for lonlat, t in zip(trips[ii], ts[ii]):
            lonlat = lonlat.tolist()
            lonlat.append(t)
            trj.append(lonlat)
        res.append(trj)
    return res


class ExpData:
    
    def __init__(self):
        self.args = set_args()
        self.loader = ExpLoader()
        data_path = "/home/like/data/trajectory_similarity/"
        city_name = settings.city_name
        scale = settings.scale
        time_size = settings.time_size
        self.save_path = os.path.join(data_path, city_name, city_name + str(int(scale * 100000)) + str(time_size), 'valData')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(os.path.join(self.save_path , "knn")):
            os.makedirs(os.path.join(self.save_path, "knn"))
        if not os.path.exists(os.path.join(self.save_path, "meanrank")):
            os.makedirs(os.path.join(self.save_path, "meanrank"))
        self.rs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 采样率，噪声率

        # self.q_num, self.p_num = 100, 100  # 默认的用于mean_rank计算的轨迹集p, q的数目
        # self.query_num, self.db_num = 100, 100  # 默认的用于knn, cs的query, db轨迹集数目
        # self.pnums = [100,200,300,400,500]

        self.q_num, self.p_num = 10000, 20000  # 默认的用于mean_rank计算的轨迹集p, q的数目
        self.query_num, self.db_num = 20000, 20000  # 默认的用于knn, cs的query, db轨迹集数目
        self.pnums = [10000, 20000, 30000, 40000, 50000]

    # 生成mean_rank评估数据
    def meanrank_data(self):

        print("\n生成变化p_num下的mean_rank数据")
        for num in self.pnums:
            q0_vecs, q1_vecs, p0_vecs, p1_vecs = self.loader.mean_rank_data(self.q_num, num, 0, True)
            DD_vecs = np.append(np.array(q1_vecs), np.array(p1_vecs), axis=0)
            # 2. 利用原轨迹 计算EDR的mean_rank
            # 下述变量记录的真实轨迹而非代表向量
            Q0 = data_transfer(self.loader.Q0_trips, self.loader.Q0_ts)
            Q1 = data_transfer(self.loader.Q1_trips, self.loader.Q1_ts)
            P1 = data_transfer(self.loader.P1_trips, self.loader.P1_ts)
            DD_traj = np.append(np.array(Q1), np.array(P1), axis=0).tolist()
            print(len(Q0), len(Q1), len(P1), len(DD_traj))

            # q0向量数据
            writer = open(os.path.join(self.save_path, 'meanrank', 'vec.q0.pnum {}'.format(num)), "w")
            for ii, vec in enumerate(q0_vecs):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()
            # DD向量数据
            writer = open(os.path.join(self.save_path, 'meanrank', 'vec.dd.pnum {}'.format(num)), "w")
            for ii, vec in enumerate(DD_vecs):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()
            # q0轨迹数据
            writer = open(os.path.join(self.save_path, 'meanrank', 'traj.q0.pnum {}'.format(num)), "w")
            for ii, vec in enumerate( Q0):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()
            # DD轨迹数据
            writer = open(os.path.join(self.save_path, 'meanrank', 'traj.dd.pnum {}'.format(num)), "w")
            for ii, vec in enumerate( DD_traj):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()

        print("\n生成变化r1下的mean_rank数据")
        for r in self.rs:
            q0_vecs, q1_vecs, p0_vecs, p1_vecs = self.loader.mean_rank_data(self.q_num, self.p_num, r, True)
            DD_vecs = np.append(np.array(q1_vecs), np.array(p1_vecs), axis=0)
            # 2. 利用原轨迹 计算EDR的mean_rank (2022/4/7检验合格)
            # 下述变量记录的真实轨迹而非代表向量
            Q0 = data_transfer(self.loader.Q0_trips, self.loader.Q0_ts)
            Q1 = data_transfer(self.loader.Q1_trips, self.loader.Q1_ts)
            P1 = data_transfer(self.loader.P1_trips, self.loader.P1_ts)
            DD_traj = np.append(np.array(Q1), np.array(P1), axis=0).tolist()
            print(len(Q0), len(Q1), len(P1), len(DD_traj))
            # q0向量数据
            writer = open(os.path.join(self.save_path, 'meanrank','vec.q0.r1 {}'.format(r)), "w")
            for ii, vec in enumerate( q0_vecs):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()
            # DD向量数据
            writer = open(os.path.join(self.save_path, 'meanrank', 'vec.dd.r1 {}'.format(r)), "w")
            for ii, vec in enumerate( DD_vecs):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()
            # q0轨迹数据
            writer = open(os.path.join(self.save_path, 'meanrank', 'traj.q0.r1 {}'.format(r)), "w")
            for ii, vec in enumerate( Q0):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()
            # DD轨迹数据
            writer = open(os.path.join(self.save_path, 'meanrank', 'traj.dd.r1 {}'.format(r)), "w")
            for ii, vec in enumerate( DD_traj):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()

        print("\n生成变化r2下的mean_rank数据")
        for r in self.rs:
            q0_vecs, q1_vecs, p0_vecs, p1_vecs = self.loader.mean_rank_data(self.q_num, self.p_num, r, False)
            DD_vecs = np.append(np.array(q1_vecs), np.array(p1_vecs), axis=0)
            # 2. 利用原轨迹 计算EDR的mean_rank (2022/4/7检验合格)
            # 下述变量记录的真实轨迹而非代表向量
            Q0 = data_transfer(self.loader.Q0_trips, self.loader.Q0_ts)
            Q1 = data_transfer(self.loader.Q1_trips, self.loader.Q1_ts)
            P1 = data_transfer(self.loader.P1_trips, self.loader.P1_ts)
            DD_traj = np.append(np.array(Q1), np.array(P1), axis=0).tolist()
            print(len(Q0), len(Q1), len(P1), len(DD_traj))
            # q0向量数据
            writer = open(os.path.join(self.save_path, 'meanrank','vec.q0.r2 {}'.format(r)), "w")
            for ii, vec in enumerate(q0_vecs):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()
            # DD向量数据
            writer = open(os.path.join(self.save_path, 'meanrank', 'vec.dd.r2 {}'.format(r)), "w")
            for ii, vec in enumerate(DD_vecs):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()
            # q0轨迹数据
            writer = open(os.path.join(self.save_path, 'meanrank', 'traj.q0.r2 {}'.format(r)), "w")
            for ii, vec in enumerate( Q0):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()
            # DD轨迹数据
            writer = open(os.path.join(self.save_path, 'meanrank', 'traj.dd.r2 {}'.format(r)), "w")
            for ii, vec in enumerate( DD_traj):
                seq = '%d 24/11/2000 11:30:41 [' % ii + (';'.join([str(id) for id in vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                writer.writelines(seq)
                writer.write('\n')
            writer.close()

    # 生成用于knn评估数据
    def knn_data(self):

        print("\n生成变化r1下的knn数据")
        for r in self.rs:
            self.query0, self.query1, self.db0, self.db1 = self.loader.knn_data(self.query_num, self.db_num, r, True)
            q0 = data_transfer(self.loader.query0_trips, self.loader.query0_ts)
            q1 = data_transfer(self.loader.query1_trips, self.loader.query1_ts)
            db0 = data_transfer(self.loader.db0_trips, self.loader.db0_ts)
            db1 = data_transfer(self.loader.db1_trips, self.loader.db1_ts)

            # q0向量数据
            q0_vec_writer = open(os.path.join(self.save_path, 'knn', 'vec.q0.r1 {}'.format(r)), "w")
            q1_vec_writer = open(os.path.join(self.save_path, 'knn', 'vec.q1.r1 {}'.format(r)), "w")
            db0_vec_writer = open(os.path.join(self.save_path, 'knn', 'vec.db0.r1 {}'.format(r)), "w")
            db1_vec_writer = open(os.path.join(self.save_path, 'knn', 'vec.db1.r1 {}'.format(r)), "w")

            q0_traj_writer = open(os.path.join(self.save_path, 'knn', 'traj.q0.r1 {}'.format(r)), "w")
            q1_traj_writer = open(os.path.join(self.save_path, 'knn', 'traj.q1.r1 {}'.format(r)), "w")
            db0_traj_writer = open(os.path.join(self.save_path, 'knn', 'traj.db0.r1 {}'.format(r)), "w")
            db1_traj_writer = open(os.path.join(self.save_path, 'knn', 'traj.db1.r1 {}'.format(r)), "w")

            num = min(len(q0), len(q1), len(db0), len(db1))
            print(num)
            for ii in range(num):
                q0_traj = q0[ii]
                q1_traj = q1[ii]
                db0_traj = db0[ii]
                db1_traj = db1[ii]

                # 防止序列为空的轨迹
                if q0_traj[0][0]-q0_traj[-1][0] <= 0.0001 and q0_traj[0][1]-q0_traj[-1][1] <= 0.0001:
                    continue
                if q1_traj[0][0] - q1_traj[-1][0] <= 0.0001 and q1_traj[0][1] - q1_traj[-1][1] <= 0.0001:
                    continue
                if db0_traj[0][0] - db0_traj[-1][0] <= 0.0001 and db0_traj[0][1] - db0_traj[-1][1] <= 0.0001:
                    continue
                if db1_traj[0][0] - db1_traj[-1][0] <= 0.0001 and db1_traj[0][1] - db1_traj[-1][1] <= 0.0001:
                    continue

                seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(id) for id in q0_traj]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                q0_traj_writer.writelines(seq)
                q0_traj_writer.write('\n')

                seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(id) for id in q1_traj]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                q1_traj_writer.writelines(seq)
                q1_traj_writer.write('\n')

                seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(id) for id in db0_traj]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                db0_traj_writer.writelines(seq)
                db0_traj_writer.write('\n')

                seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(id) for id in db1_traj]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                db1_traj_writer.writelines(seq)
                db1_traj_writer.write('\n')

                q0_vec = self.query0[ii]
                q0_seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(loc) for loc in q0_vec]).replace("[", "(").replace("]", ")").replace(", ",",")) + "]"
                q0_vec_writer.writelines(q0_seq)
                q0_vec_writer.write('\n')

                q1_vec = self.query1[ii]
                q1_seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(loc) for loc in q1_vec]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                q1_vec_writer.writelines(q1_seq)
                q1_vec_writer.write('\n')

                db0_vec = self.db0[ii]
                db0_seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(loc) for loc in db0_vec]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                db0_vec_writer.writelines(db0_seq)
                db0_vec_writer.write('\n')

                db1_vec = self.db1[ii]
                db1_seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(loc) for loc in db1_vec]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                db1_vec_writer.writelines(db1_seq)
                db1_vec_writer.write('\n')


            q0_vec_writer.close()
            q1_vec_writer.close()
            db0_vec_writer.close()
            db1_vec_writer.close()
            q0_traj_writer.close()
            q1_traj_writer.close()
            db0_traj_writer.close()
            db1_traj_writer.close()

        print("\n生成变化r2下的knn数据")
        for r in self.rs:
            self.query0, self.query1, self.db0, self.db1 = self.loader.knn_data(self.query_num, self.db_num, r, False)
            q0 = data_transfer(self.loader.query0_trips, self.loader.query0_ts)
            q1 = data_transfer(self.loader.query1_trips, self.loader.query1_ts)
            db0 = data_transfer(self.loader.db0_trips, self.loader.db0_ts)
            db1 = data_transfer(self.loader.db1_trips, self.loader.db1_ts)

            # q0向量数据
            q0_vec_writer = open(os.path.join(self.save_path, 'knn', 'vec.q0.r2 {}'.format(r)), "w")
            q1_vec_writer = open(os.path.join(self.save_path, 'knn', 'vec.q1.r2 {}'.format(r)), "w")
            db0_vec_writer = open(os.path.join(self.save_path, 'knn', 'vec.db0.r2 {}'.format(r)), "w")
            db1_vec_writer = open(os.path.join(self.save_path, 'knn', 'vec.db1.r2 {}'.format(r)), "w")

            q0_traj_writer = open(os.path.join(self.save_path, 'knn', 'traj.q0.r2 {}'.format(r)), "w")
            q1_traj_writer = open(os.path.join(self.save_path, 'knn', 'traj.q1.r2 {}'.format(r)), "w")
            db0_traj_writer = open(os.path.join(self.save_path, 'knn', 'traj.db0.r2 {}'.format(r)), "w")
            db1_traj_writer = open(os.path.join(self.save_path, 'knn', 'traj.db1.r2 {}'.format(r)), "w")

            num = min(len(q0), len(q1), len(db0), len(db1))
            print(num)
            for ii in range(num):
                q0_traj = q0[ii]
                q1_traj = q1[ii]
                db0_traj = db0[ii]
                db1_traj = db1[ii]

                if q0_traj[0][0] - q0_traj[-1][0] <= 0.0001 and q0_traj[0][1] - q0_traj[-1][1] <= 0.0001:
                    continue
                if q1_traj[0][0] - q1_traj[-1][0] <= 0.0001 and q1_traj[0][1] - q1_traj[-1][1] <= 0.0001:
                    continue
                if db0_traj[0][0] - db0_traj[-1][0] <= 0.0001 and db0_traj[0][1] - db0_traj[-1][1] <= 0.0001:
                    continue
                if db1_traj[0][0] - db1_traj[-1][0] <= 0.0001 and db1_traj[0][1] - db1_traj[-1][1] <= 0.0001:
                    continue

                seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(id) for id in q0_traj]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                q0_traj_writer.writelines(seq)
                q0_traj_writer.write('\n')

                seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(id) for id in q1_traj]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                q1_traj_writer.writelines(seq)
                q1_traj_writer.write('\n')

                seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(id) for id in db0_traj]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                db0_traj_writer.writelines(seq)
                db0_traj_writer.write('\n')

                seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(id) for id in db1_traj]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                db1_traj_writer.writelines(seq)
                db1_traj_writer.write('\n')

                q0_vec = self.query0[ii]
                q0_seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(loc) for loc in q0_vec]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                q0_vec_writer.writelines(q0_seq)
                q0_vec_writer.write('\n')

                q1_vec = self.query1[ii]
                q1_seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(loc) for loc in q1_vec]).replace("[", "(").replace("]", ")").replace(", ", ",")) + "]"
                q1_vec_writer.writelines(q1_seq)
                q1_vec_writer.write('\n')

                db0_vec = self.db0[ii]
                db0_seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(loc) for loc in db0_vec]).replace("[", "(").replace("]", ")").replace(", ",
                                                                                                        ",")) + "]"
                db0_vec_writer.writelines(db0_seq)
                db0_vec_writer.write('\n')

                db1_vec = self.db1[ii]
                db1_seq = '%d 24/11/2000 11:30:41 [' % ii + (
                    ';'.join([str(loc) for loc in db1_vec]).replace("[", "(").replace("]", ")").replace(", ",
                                                                                                        ",")) + "]"
                db1_vec_writer.writelines(db1_seq)
                db1_vec_writer.write('\n')

            q0_vec_writer.close()
            q1_vec_writer.close()
            db0_vec_writer.close()
            db1_vec_writer.close()
            q0_traj_writer.close()
            q1_traj_writer.close()
            db0_traj_writer.close()
            db1_traj_writer.close()


if __name__ == "__main__":
    c = ExpData()
    with torch.no_grad():
        c.knn_data()
        c.meanrank_data()