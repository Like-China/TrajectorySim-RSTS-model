
import java.util.ArrayList;
import java.util.Collections;


public class knnRSTS {

    int expNum = Settings.expNum;
    int qNum = Settings.knn_qNum;
    int dbNum = Settings.knn_dbNum;
    String data_path = Settings.path+"knn/";
    final String[] rates = Settings.rates;
    final String[] pnums = Settings.pnums;
    // knn k����Ŀ, ���ݼ��켣��ĿӦ�ô���K
    public int k = Settings.k;

    public String varyR1()
    {   
        System.out.println("�仯r1��KNN׼ȷ�ʱ仯");
        long timeSum = 0;
        ArrayList<Float> precisions = new ArrayList<>();
        for(String r1: rates)
        {   
            // ��ȡԭʼ�켣
            String query_path = data_path+"vec.q0.r1 "+r1;
            String db_path = data_path+"vec.db0.r1 "+r1;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q0 = q_stream.read();
            ArrayList<ArrayList<Float>> db0 = db_stream.read();
            // ��ȡ�����켣
            query_path = data_path+"vec.q1.r1 "+r1;
            db_path = data_path+"vec.db1.r1 "+r1;
            q_stream = new TrajectoryVecLoader(query_path, qNum);
            db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q1 = q_stream.read();
            ArrayList<ArrayList<Float>> db1 = db_stream.read();
            // ����knn
            long time1 = System.currentTimeMillis();
            float precision = knnEDR(q0,db0,q1,db1,k);
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
            // ��ȡԭʼ�켣
            String query_path = data_path+"vec.q0.r2 "+r;
            String db_path = data_path+"vec.db0.r2 "+r;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q0 = q_stream.read();
            ArrayList<ArrayList<Float>> db0 = db_stream.read();
            // ��ȡ�����켣
            query_path = data_path+"vec.q1.r2 "+r;
            db_path = data_path+"vec.db1.r2 "+r;
            q_stream = new TrajectoryVecLoader(query_path, qNum);
            db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q1 = q_stream.read();
            ArrayList<ArrayList<Float>> db1 = db_stream.read();
            // ����knn
            long time1 = System.currentTimeMillis();
            float precision = knnEDR(q0,db0,q1,db1,k);
            precisions.add(precision);
            long time2 = System.currentTimeMillis();
            timeSum += (time2-time1);
        }
        System.out.println(precisions);
        return  "vary r2->knn "+precisions+ " CPU time: "+timeSum;
    }
    

    

    // ����query��db��knn�ھӣ�query��ĿX�ھ���Ŀ
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
            // �Ծ�������
            Collections.sort(dists);
            // ��ȡknn�ھ�
            for(int nn = 0;nn<k;nn++)
            {
                knns[i][nn] = rawDists.indexOf(dists.get(nn));
            }
        }
        return knns;
    }

    public float knnEDR(ArrayList<ArrayList<Float>> q0, ArrayList<ArrayList<Float>> db0, ArrayList<ArrayList<Float>> q1, ArrayList<ArrayList<Float>> db1, int k)
    {   
        // ����knns�Ľ�����Ŀ
        int[][] knns1 = knn(q0, db0, k);
        int[][] knns2 = knn(q1, db1, k);
        // ÿ�ж��������ж��ٽ���
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
