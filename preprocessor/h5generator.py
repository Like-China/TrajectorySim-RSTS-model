import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import h5py
import warnings
import time
warnings.filterwarnings("ignore")


class Processor:
    """
    读取一定数量taxi文本， 生成满足条件的轨迹文本 h5,porto已经有对应生成的h5文件，主要是对beijing taxi数据集生成对等格式的h5文件
    参照porto.h5, beijing.h5文件中记录的属性:
    trips       trips/0 -> [[longitude, latitude], ...]

    timestamps     timestamps/0 -> [ts, ts,...]

    f.attrs['num']  记录的有效的轨迹总数目
    """
    def __init__(self, read_txt_size, each_read):
        self.longtitude_range = [116.25, 116.55]
        self.latitude_range = [39.83, 40.03]
        self.current_read = 0
        self.each_read = each_read  # 每一批读取多少个txt文件
        self.read_txt_size = read_txt_size
        self.min_length, self.max_length = 20, 100
        self.file_dir = os.path.join("/home/Like/data/taxi_log_2008_by_id")
        self.valid_trip_nums = 0  # 当前写了多少个trips到h5文件中
        self.h5_filepath = os.path.join("/home/Like/data/beijing11.h5")
        self.freq_interval = 10  # 限制过短的采样频率

    def go(self):
        """
        一批一批地读入taxi文本并写出到h5文件
        :return:
        """
        f = h5py.File(self.h5_filepath, 'w')
        for ii in tqdm(range(self.read_txt_size//self.each_read)):
            batch_trjs = self.read_files()
            self.address_data(batch_trjs)
            self.output_src_tgr(f)
            self.current_read += self.each_read
        f.close()
        print("\n共写出轨迹："+str(self.valid_trip_nums))

    def read_files(self):
        """
        从一定数目的taxi txt轨迹文本中读取一批轨迹信息, 存储到 raw_trjs 中并返回
        :return: trjs_raw 为固定数目的一批txt文件，在经纬度范围内，每条记录为同一天之内的轨迹，且满足轨迹最小长度限制
        """
        trjs_raw = []
        all_file_list = os.listdir(self.file_dir)
        all_file_list.sort(key=lambda x: int(x[:-4]))
        all_file_list = all_file_list[self.current_read:self.current_read + self.each_read]
        all_data = pd.DataFrame()
        for file in all_file_list:
            single_data = pd.read_csv(os.path.join(self.file_dir, file), names=['id', 'times', 'longitude', 'latitude'],
                                      header=None)
            if (len(single_data)) < 100: continue  # 如果该 taxi文本记录的轨迹信息太少，则不记录该轨迹信息
            all_data = all_data.append(single_data)
        # 过滤不在经纬度区域范围内的数据
        all_data = all_data[self.longtitude_range[0] <= all_data.longitude]
        all_data = all_data[all_data.longitude <= self.longtitude_range[1]]
        all_data = all_data[self.latitude_range[0] <= all_data.latitude]
        all_data = all_data[all_data.latitude <= self.latitude_range[1]]

        str_times = list(all_data['times'])
        longitudes = list(all_data['longitude'])
        latitudes = list(all_data['latitude'])
        location = list(zip(longitudes, latitudes))

        i = 0
        while i < len(location):
            cur_date = str_times[i]
            single_taj_with_time = []
            while i < len(location) and cur_date[:10] == str_times[i][:10]:
                single_taj_with_time.append((str_times[i][11:], location[i]))
                i += 1
            if len(single_taj_with_time) >= self.min_length:
                trjs_raw.append(single_taj_with_time)
        return trjs_raw

    def address_data(self, batch_trjs):
        """
        处理一批轨迹，如果轨迹太长，则划分轨迹
        """
        # [经纬度坐标，时间]
        self.tajs_with_time = []
        # [经纬度坐标, 转化的时间戳]
        self.tajs_with_ts = []
        # taj是一条轨迹 [('15:36:08', (116.51172, 39.92123)), ('15:46:08', (116.51135, 39.938829999999996))]
        for taj in batch_trjs:
            # 单条轨迹不能超过最大长度 max_length，如果大于需要划分
            # taj = [('13:33:52', (116.36421999999999, 39.887809999999995))]
            tajs = []
            while len(taj) >= self.max_length:
                # 将轨迹按min-max轨迹长度生成一个随机长度的轨迹
                rand_len = np.random.randint(20, 100)
                tajs.append(taj[0:rand_len])
                taj = taj[rand_len:]
            if len(taj) >= self.min_length:
                tajs.append(taj)

            for ii in range(len(tajs)):
                # t = [('13:33:52', (116.36421999999999, 39.887809999999995))]
                t = tajs[ii]
                single_taj_with_time = []
                single_taj_with_ts = []
                # 记录前一个时间戳，防止记录过于频繁的轨迹
                last_timestamp = 0
                flag = True
                for jj in range(len(t)):
                    time = t[jj][0]
                    # 转化为时间戳
                    h, m, s = [int(i) for i in time.split(":")]
                    timestamp = h * 3600 + m * 60 + s
                    # 过滤记录频繁的轨迹
                    if timestamp - last_timestamp <= self.freq_interval:
                        flag = False
                        break
                    longtitude, latitude = t[jj][1][0], t[jj][1][1]
                    single_taj_with_ts.append([longtitude, latitude, timestamp])
                    single_taj_with_time.append([longtitude, latitude, time])
                # 如果轨迹中没有较为频繁的采样间隔，则记录
                if flag:
                    self.tajs_with_ts.append(single_taj_with_ts)
                    self.tajs_with_time.append(single_taj_with_time)

    def output_src_tgr(self, f):
        """
        写出一批轨迹
        :param f: 目标h5文件
        """
        for each_taj in self.tajs_with_ts:
            # 再一次控制轨迹长度
            if self.min_length <= len(each_taj) <= self.max_length:
                self.valid_trip_nums += 1
                # 转换为array形式
                each_taj = np.array(each_taj)
                locations = each_taj[:, 0:2]
                times = [each[0] for each in each_taj[:, 2:3]]
                f["trips/"+str(self.valid_trip_nums)] = locations
                f["timestamps/"+str(self.valid_trip_nums)] = times
        f.attrs['num'] = self.valid_trip_nums


class ProcessorTester:
    """
    检测生成的h5文件是否合理，是否存在错误
    """
    def __init__(self):
        # 存储h5文件的路径
        self.h5_filepath = os.path.join("/home/Like/data/porto.h5")
        f = h5py.File(self.h5_filepath, 'r')
        self.checked_nums = min(100000, f.attrs['num'])

    def observe(self):
        with h5py.File(self.h5_filepath, 'r') as f:
            for ii in tqdm(range(self.checked_nums)):
                trip = np.array(f.get('trips/' + str(ii + 1)))
                ts = np.array(f.get('timestamps/' + str(ii + 1)))
                print(trip)
                print(ts)

    def check_lens(self):
        """
        检测轨迹长度是否合理
        """
        with h5py.File(self.h5_filepath, 'r') as f:
            # print(f.keys())
            lens = []
            for ii in tqdm(range(self.checked_nums)):
                trip = np.array(f.get('trips/' + str(ii + 1)))
                ts = np.array(f.get('timestamps/' + str(ii + 1)))
                lens.append(len(ts))
                assert 20 <= len(ts) <= 100, "长度超出范围限制"
                assert len(trip) == len(ts), "locations长度和时间长度不一致!!"


if __name__ == "__main__":

    # t1 = time.time()
    # P = Processor(10200, 50)
    # P.go()
    # print("获取h5文件用时：", time.time()-t1)

    Ptest = ProcessorTester()
    Ptest.observe()
