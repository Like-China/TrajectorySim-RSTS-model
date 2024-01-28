
import java.util.ArrayList;
import java.util.Collections;

public class meanrankRSTS {

    int expNum = Settings.expNum;
    int qNum = Settings.qNum;
    int dbNum = Settings.dbNum;
    String data_path = Settings.path + "meanrank/";
    final String[] rates = Settings.rates;
    final String[] pnums = Settings.pnums;

    public String varyR1() {
        System.out.println("变化r1下的mean rank 变化");
        long timeSum = 0;
        ArrayList<Float> meanRanks = new ArrayList<>();
        for (String r1 : rates) {
            String query_path = data_path + "vec.q0.r1 " + r1;
            String db_path = data_path + "vec.dd.r1 " + r1;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q = q_stream.read();
            ArrayList<ArrayList<Float>> db = db_stream.read();
            // 计算mean rank
            long time1 = System.currentTimeMillis();
            ArrayList<Integer> ranks = meanrank(q, db);
            float rankSum = 0f;
            for (int rank : ranks) {
                rankSum += rank;
            }
            float meanrank = rankSum / ranks.size();
            meanRanks.add(meanrank);
            long time2 = System.currentTimeMillis();
            timeSum += (time2 - time1);
        }
        System.out.println(meanRanks);
        return "vary r1->mean rank " + meanRanks + " CPU time: " + timeSum;
    }

    public String varyR2() {
        System.out.println("变化r1下的mean rank 变化");
        long timeSum = 0;
        ArrayList<Float> meanRanks = new ArrayList<>();
        for (String r1 : rates) {
            String query_path = data_path + "vec.q0.r2 " + r1;
            String db_path = data_path + "vec.dd.r2 " + r1;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q = q_stream.read();
            ArrayList<ArrayList<Float>> db = db_stream.read();
            // 计算mean rank
            long time1 = System.currentTimeMillis();
            ArrayList<Integer> ranks = meanrank(q, db);
            float rankSum = 0f;
            for (int rank : ranks) {
                rankSum += rank;
            }
            float meanrank = rankSum / ranks.size();
            meanRanks.add(meanrank);
            long time2 = System.currentTimeMillis();
            timeSum += (time2 - time1);
        }
        System.out.println(meanRanks);
        return "vary r2->mean rank " + meanRanks + " CPU time: " + timeSum;
    }

    public String varyNum() {
        System.out.println("变化P Size的mean rank 变化");
        long timeSum = 0;
        ArrayList<Float> meanRanks = new ArrayList<>();
        for (String pnum : pnums) {
            String query_path = data_path + "vec.q0.pnum " + pnum;
            String db_path = data_path + "vec.dd.pnum " + pnum;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q = q_stream.read();
            ArrayList<ArrayList<Float>> db = db_stream.read();
            // 计算mean rank
            long time1 = System.currentTimeMillis();
            ArrayList<Integer> ranks = meanrank(q, db);
            float rankSum = 0f;
            for (int rank : ranks) {
                rankSum += rank;
            }
            float meanrank = rankSum / ranks.size();
            meanRanks.add(meanrank);
            long time2 = System.currentTimeMillis();
            timeSum += (time2 - time1);
        }
        System.out.println(meanRanks);
        return "vary num->mean rank " + meanRanks;
    }

    // 计算一组查询轨迹集合到db集合的mean rank
    public ArrayList<Integer> meanrank(ArrayList<ArrayList<Float>> q, ArrayList<ArrayList<Float>> db) {
        ArrayList<Integer> ranks = new ArrayList<>();
        // 计算两组轨迹集合两两之间的距离并排序
        for (int i = 0; i < q.size(); i++) {
            ArrayList<Float> query_traj = q.get(i);
            ArrayList<Float> dists = new ArrayList<>();
            for (int j = 0; j < db.size(); j++) {
                ArrayList<Float> db_traj = db.get(j);
                float edr_dist = Distance.edulidean(query_traj, db_traj);
                dists.add(edr_dist);
            }
            // 每当一个query到所有db距离计算完毕，对其进行排序
            // 获取与twin的距离
            float twin_dist = dists.get(i);
            // 对距离排序
            Collections.sort(dists);
            // 记录twin的排名
            int rank = dists.indexOf(twin_dist) + 1;
            ranks.add(rank);
        }
        return ranks;
    }
}
