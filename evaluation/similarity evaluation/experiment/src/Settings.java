
public class Settings {
    public static String path = "/home/like/data/trajectory_similarity/beijing/beijing100200/valData/";
    public static int expNum = 1;
    public static int qNum = 10000;
    public static int dbNum = 10000;
    public static String data_path = null;
    public static final String[] rates = "0,0.1,0.2,0.3,0.4,0.5".split(",");
    public static final String[] pnums = "10000,20000,30000,40000,50000".split(",");
    // EDR setting
    public static float xyDiff = 0.005f;
    public static float timeDiff = 50f;
    // Knn setting
    public static int k = 200;
    public static int knn_qNum = 5000;
    public static int knn_dbNum = 10000;
}
