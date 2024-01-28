
import java.util.ArrayList;

public class csdRSTS {
    int expNum = Settings.expNum;
    int qNum = Settings.qNum;
    int dbNum = Settings.dbNum;
    String data_path = Settings.path+"knn/";
    final String[] rates = Settings.rates;
    final String[] pnums = Settings.pnums;


    public String varyR1()
    {   
        System.out.println("变化r1下的csd变化");
        long timeSum = 0;
        ArrayList<Float> csds = new ArrayList<>();
        for(String r1: rates)
        {   
            float csd = 0f;
            // 获取原始轨迹
            String query_path = data_path+"vec.q0.r1 "+r1;
            String db_path = data_path+"vec.db0.r1 "+r1;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q0 = q_stream.read();
            ArrayList<ArrayList<Float>> db0 = db_stream.read();
            // 获取处理后轨迹
            query_path = data_path+"vec.q1.r1 "+r1;
            db_path = data_path+"vec.db1.r1 "+r1;
            q_stream = new TrajectoryVecLoader(query_path, qNum);
            db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q1 = q_stream.read();
            ArrayList<ArrayList<Float>> db1 = db_stream.read();
            // 计算csd
            long time1 = System.currentTimeMillis();
            int min_size = q0.size()>db0.size()?db0.size():q0.size();
            for(int i=0;i<min_size;i++)
            {
                float dist0 = Distance.edulidean(q0.get(i), db0.get(i));
                float dist1 = Distance.edulidean(q1.get(i), db1.get(i));
                if(dist0 !=0)
                {
                    csd += (float)Math.abs(dist0-dist1)/dist0;
                }
            }
            csds.add(csd/min_size);
            long time2 = System.currentTimeMillis();
            timeSum += (time2-time1);
        }
        System.out.println(csds);
        return  "vary r1->csd "+csds+ " CPU time: "+timeSum;
    }
    
    public String varyR2()
    {   
        System.out.println("变化下r2下的csd变化");
        long timeSum = 0;
        ArrayList<Float> csds = new ArrayList<>();
        for(String r2: rates)
        {   
            float csd = 0f;
            // 获取原始轨迹
            String query_path = data_path+"vec.q0.r2 "+r2;
            String db_path = data_path+"vec.db0.r2 "+r2;
            TrajectoryVecLoader q_stream = new TrajectoryVecLoader(query_path, qNum);
            TrajectoryVecLoader db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q0 = q_stream.read();
            ArrayList<ArrayList<Float>> db0 = db_stream.read();
            // 获取处理后轨迹
            query_path = data_path+"vec.q1.r2 "+r2;
            db_path = data_path+"vec.db1.r2 "+r2;
            q_stream = new TrajectoryVecLoader(query_path, qNum);
            db_stream = new TrajectoryVecLoader(db_path, dbNum);
            ArrayList<ArrayList<Float>> q1 = q_stream.read();
            ArrayList<ArrayList<Float>> db1 = db_stream.read();
            // 计算csd
            long time1 = System.currentTimeMillis();
            int min_size = q0.size()>db0.size()?db0.size():q0.size();
            for(int i=0;i<min_size;i++)
            {
                float dist0 = Distance.edulidean(q0.get(i), db0.get(i));
                float dist1 = Distance.edulidean(q1.get(i), db1.get(i));
                if(dist0 !=0)
                {
                    csd += (float)Math.abs(dist0-dist1)/dist0;
                }
            }
            csds.add(csd/min_size);
            long time2 = System.currentTimeMillis();
            timeSum += (time2-time1);
        }
        System.out.println(csds);
        return  "vary r2->csd "+csds+ " CPU time: "+timeSum;
    }
    
}
