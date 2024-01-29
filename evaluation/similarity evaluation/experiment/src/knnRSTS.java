
import java.util.ArrayList;
import java.util.Collections;


public class knnRSTS {

    int expNum = Settings.expNum;
    int qNum = Settings.knn_qNum;
    int dbNum = Settings.knn_dbNum;
    String data_path = Settings.path+"knn/";
    final String[] rates = Settings.rates;
    final String[] pnums = Settings.pnums;
    // knn k
    public int k = Settings.k;

    public String varyR1()
    {   
        System.out.println("vary r1: kNN precision");
        long timeSum = 0;
        ArrayList<Float> precisions = new ArrayList<>();
        for(String r1: rates)
        {   
            // the trajectory representation vectors of raw trajectories
            String query_path = data_path+"vec.q0.r1 "+r1;
            String db_path = data_path+"vec.db0.r1 "+r1;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q0 = q_stream.read();
            ArrayList<ArrayList<Float>> db0 = db_stream.read();
            // representation vectors of processed trajectories
            query_path = data_path+"vec.q1.r1 "+r1;
            db_path = data_path+"vec.db1.r1 "+r1;
            q_stream = new TrajectoryVecLoader(query_path, qNum);
            db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q1 = q_stream.read();
            ArrayList<ArrayList<Float>> db1 = db_stream.read();
            // knn calculation
            long time1 = System.currentTimeMillis();
            float precision = knnPrecision(q0,db0,q1,db1,k);
            precisions.add(precision);
            long time2 = System.currentTimeMillis();
            timeSum += (time2-time1);
        }
        System.out.println(precisions);
        return  "vary r1->knn "+precisions+ " CPU time: "+timeSum;
    }

    public String varyR2()
    {   
        System.out.println("�仯r2��KNN׼ȷ�ʱ仯");
        long timeSum = 0;
        ArrayList<Float> precisions = new ArrayList<>();
        for(String r: rates)
        {   
            // the trajectory representation vectors of raw trajectories
            String query_path = data_path+"vec.q0.r2 "+r;
            String db_path = data_path+"vec.db0.r2 "+r;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q0 = q_stream.read();
            ArrayList<ArrayList<Float>> db0 = db_stream.read();
            query_path = data_path+"vec.q1.r2 "+r;
            db_path = data_path+"vec.db1.r2 "+r;
            q_stream = new TrajectoryVecLoader(query_path, qNum);
            db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q1 = q_stream.read();
            ArrayList<ArrayList<Float>> db1 = db_stream.read();
            long time1 = System.currentTimeMillis();
            float precision = knnPrecision(q0,db0,q1,db1,k);
            precisions.add(precision);
            long time2 = System.currentTimeMillis();
            timeSum += (time2-time1);
        }
        System.out.println(precisions);
        return  "vary r2->knn "+precisions+ " CPU time: "+timeSum;
    }
       

    /**
     * calculate the kNN neighbors of query trajectory in database trajectory
     * @param q query
     * @param db database
     * @param k kNN parameter
     * @return the kNN neighbors of q in db
     */
    public int[][] knn(ArrayList<ArrayList<Float>> q, ArrayList<ArrayList<Float>> db, int k)
    {   
        int[][] knns = new int[q.size()][k];
        for(int i=0;i<q.size();i++)
        {   
            ArrayList<Float> query_traj = q.get(i);
            ArrayList<Float> dists = new ArrayList<>();
            for(int j=0;j<db.size();j++)
            {
                ArrayList<Float> db_traj = db.get(j);
                float edr_dist = Distance.edulidean(query_traj, db_traj);
                dists.add(edr_dist);
            }
            ArrayList<Float> rawDists = new ArrayList<>(dists);
            Collections.sort(dists);
            // get kNN neighbors of the query
            for(int nn = 0;nn<k;nn++)
            {
                knns[i][nn] = rawDists.indexOf(dists.get(nn));
            }
        }
        return knns;
    }

    public float knnPrecision(ArrayList<ArrayList<Float>> q0, ArrayList<ArrayList<Float>> db0, ArrayList<ArrayList<Float>> q1, ArrayList<ArrayList<Float>> db1, int k)
    {   
        // calculate the intersection of two kNN lists to get kNN precision
        int[][] knns1 = knn(q0, db0, k);
        int[][] knns2 = knn(q1, db1, k);
        float intersection_size = 0;
        for(int row = 0;row<knns1.length;row++)
        {
            ArrayList<Integer> knn1 = new ArrayList<>();
            for(int nn: knns1[row])
            {
                knn1.add(nn);
            }
            ArrayList<Integer> knn2 = new ArrayList<>();
            for(int nn: knns2[row])
            {
                knn2.add(nn);
            }
            knn2.retainAll(knn1);
            intersection_size += knn2.size();
        }
        float precision = intersection_size/(knns1.length*k);
        return precision;
    }

}
