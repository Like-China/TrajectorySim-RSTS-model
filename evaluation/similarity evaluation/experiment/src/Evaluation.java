public class Evaluation {

    public static void main(String[] args) {
        long time1 = System.currentTimeMillis();
        String info = null;
        // get meanrank of RSTS when varying dropping rates, distorting rates and the trajectory size
        meanrankRSTS m = new meanrankRSTS();
        info = "RSTS:  " + m.varyR1();
        Loger.writeFile(info);
        info = "RSTS:  " + m.varyR2();
        Loger.writeFile(info);
        info = "RSTS:  " + m.varyNum();
        Loger.writeFile(info);
        // get csd
        csdRSTS c = new csdRSTS();
        info = "RSTS:  " + c.varyR1();
        Loger.writeFile(info);
        info = "RSTS:  " + c.varyR2();
        Loger.writeFile(info);
        // get knn precision
        knnRSTS k = new knnRSTS();
        info = "RSTS:  " + k.varyR1();
        Loger.writeFile(info);
        info = "RSTS:  " + k.varyR2();
        Loger.writeFile(info);
        long time2 = System.currentTimeMillis();
        System.out.println("total time: " + (time2 - time1));
    }
}
