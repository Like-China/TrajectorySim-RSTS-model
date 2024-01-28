
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;

/*
Given a trajectory txt file, each line is in the form of "id date time trjVectorSequence"
We focus on the timestamped trajectory location sequence [(lon1,lat1,ts1),(lon2,lat2,ts2),...]
Return a batch of trajectory vectors in the form of ArrayList<float[]>, each float[][] is represented a traj vector
 */

public class TrajectoryVecLoader {
	String txt_path;
	BufferedReader reader;
	int readTrjMaxNum;

	public TrajectoryVecLoader(String txt_path, int num) {
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

	public ArrayList<ArrayList<Float>> read() {
		int count = 0;
		ArrayList<ArrayList<Float>> batch = new ArrayList<>();
		try {
			String lineString;
			while ((lineString = reader.readLine()) != null) {
				ArrayList<Float> traj = new ArrayList<>();
				String seq = lineString.split(" ")[3];
				seq = seq.replace("[", "").replace("]", "").replace("(", "").replace(")", "");
				String[] vecs = seq.split(";");
				for (String vec : vecs) {
					float a = Float.parseFloat(vec);
					traj.add(a);
				}
				count++;
				if (count > readTrjMaxNum) {
					break;
				}
				batch.add(traj);
			}
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return batch;
	}

	public static void main(String[] args) {
		String data_path = "/home/like/data/trajectory_similarity/porto/porto100300/valData/knn/";
		String query_path = data_path + "vec.q0.r1 " + 0;
		TrajectoryVecLoader v = new TrajectoryVecLoader(query_path, 1);
		ArrayList<ArrayList<Float>> batch = v.read();
		System.out.println(batch);
	}

}
