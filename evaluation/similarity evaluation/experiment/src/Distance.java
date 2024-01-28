

import java.util.ArrayList;

public class Distance {

    public static float edulidean(ArrayList<Float> a, ArrayList<Float> b)
    {
        double distance = 0;
		
		if (a.size() == b.size()) {
			for (int i = 0; i < a.size(); i++) {
				double temp = Math.pow(a.get(i)-b.get(i), 2);
				distance += temp;
			}
			distance = Math.sqrt(distance);
		}else{
            System.out.println("Unequal length");
        }
		return (float)distance;
    }
}
