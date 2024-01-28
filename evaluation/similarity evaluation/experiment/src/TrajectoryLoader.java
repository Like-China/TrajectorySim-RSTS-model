import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

/*
Given a trajectory txt file, each line is in the form of "id date time trjLocationSequence"
We focus on the timestamped trajectory location sequence [(lon1,lat1,ts1),(lon2,lat2,ts2),...]
Return a batch of trajectories in the form of ArrayList<float[][]>, each float[][] is represented by [lon,lat,ts]
Used for EDwP_t and EDR_t evaluation
 */

public class TrajectoryLoader {
	String txt_path;
	BufferedReader reader;
	int readTrjMaxNum;

	public TrajectoryLoader(String txt_path, int num) {
		this.readTrjMaxNum = num;
		this.txt_path = txt_path;
		// TODO Auto-generated constructor stub
		try {
			File file = new File(txt_path);
			reader = new BufferedReader(new FileReader(file));
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}

	/**
	 * load trajectory batch
	 */
	public ArrayList<float[][]> read() {
		int count = 0;
		ArrayList<float[][]> batch = new ArrayList<>();
		try {
			String lineString;
			while ((lineString = reader.readLine()) != null) {
				String seq = lineString.split(" ")[3];
				System.out.println(lineString.split(" ")[2]);
				seq = seq.replace("[", "").replace("]", "").replace("(", "").replace(")", "");
				String[] locations = seq.split(";");
				float[][] trajectory = new float[locations.length][3];
				for (int i = 0; i < locations.length; i++) {
					String[] lon_lat_time = locations[i].split(",");
					trajectory[i][0] = Float.parseFloat(lon_lat_time[0]);
					trajectory[i][1] = Float.parseFloat(lon_lat_time[1]);
					trajectory[i][2] = Float.parseFloat(lon_lat_time[2]);
				}
				batch.add(trajectory);
				count++;
				if (count >= readTrjMaxNum) {
					break;
				}
			}
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return batch;
	}

	// test
	public static void main(String[] args) {
		String txt_path = "/home/like/data/trajectory_similarity/beijing/beijing100200/valData/meanrank/traj.dd.pnum 10000";
		int num = 10000;
		TrajectoryLoader s = new TrajectoryLoader(txt_path, num);
		ArrayList<float[][]> batch = s.read();
		System.out.println("The number of loading trajectories: " + batch.size());
		System.out.println("The first trajectory: ");
		System.out.println(Arrays.deepToString(batch.get(0)));
	}

}
