public class Evaluation {

    public static void main(String[] args) {
        long time1 = System.currentTimeMillis();
        String info = null;
        meanrankRSTS m = new meanrankRSTS();
        info = "RSTS:  " + m.varyR1();
        Loger.writeFile(info);
        info = "RSTS:  " + m.varyR2();
        Loger.writeFile(info);
        info = "RSTS:  " + m.varyNum();
        Loger.writeFile(info);
        csdRSTS c = new csdRSTS();
        info = "RSTS:  " + c.varyR1();
        Loger.writeFile(info);
        info = "RSTS:  " + c.varyR2();
        Loger.writeFile(info);
        knnRSTS k = new knnRSTS();
        info = "RSTS:  " + k.varyR1();
        Loger.writeFile(info);
        info = "RSTS:  " + k.varyR2();
        Loger.writeFile(info);
        long time2 = System.currentTimeMillis();
        System.out.println(" total time: " + (time2 - time1));
    }
}
