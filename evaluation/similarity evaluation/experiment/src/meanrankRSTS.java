
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
        System.out.println("vary r1: meanrank");
        long timeSum = 0;
        ArrayList<Float> meanRanks = new ArrayList<>();
        for (String r1 : rates) {
            String query_path = data_path + "vec.q0.r1 " + r1;
            String db_path = data_path + "vec.dd.r1 " + r1;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q = q_stream.read();
            ArrayList<ArrayList<Float>> db = db_stream.read();
            // calculate mean rank
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
        System.out.println("vary r2: meanrank");
        long timeSum = 0;
        ArrayList<Float> meanRanks = new ArrayList<>();
        for (String r1 : rates) {
            String query_path = data_path + "vec.q0.r2 " + r1;
            String db_path = data_path + "vec.dd.r2 " + r1;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q = q_stream.read();
            ArrayList<ArrayList<Float>> db = db_stream.read();
            // º∆À„mean rank
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
        System.out.println("vary the size of P: meanrank");
        long timeSum = 0;
        ArrayList<Float> meanRanks = new ArrayList<>();
        for (String pnum : pnums) {
            String query_path = data_path + "vec.q0.pnum " + pnum;
            String db_path = data_path + "vec.dd.pnum " + pnum;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q = q_stream.read();
            ArrayList<ArrayList<Float>> db = db_stream.read();
            // º∆À„mean rank
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
        return "vary num->mean rank " + meanRanks + " CPU time: " + timeSum;
    }

    public ArrayList<Integer> meanrank(ArrayList<ArrayList<Float>> q, ArrayList<ArrayList<Float>> db) {
        ArrayList<Integer> ranks = new ArrayList<>();
        // Calculate the distance between two sets of trajectories and sort them
        for (int i = 0; i < q.size(); i++) {
            ArrayList<Float> query_traj = q.get(i);
            ArrayList<Float> dists = new ArrayList<>();
            for (int j = 0; j < db.size(); j++) {
                ArrayList<Float> db_traj = db.get(j);
                float edr_dist = Distance.edulidean(query_traj, db_traj);
                dists.add(edr_dist);
            }
            // Sort a query every time the distance to all db's is calculated, get its twin
            // distance
            float twin_dist = dists.get(i);
            Collections.sort(dists);
            // get the ranking of its twin
            int rank = dists.indexOf(twin_dist) + 1;
            ranks.add(rank);
        }
        return ranks;
    }
}
