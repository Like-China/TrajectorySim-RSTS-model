

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Loger {
    // Write results log
	public static void writeFile(String info){
		try {
			File writeName = new File("./out.txt");
			writeName.createNewFile();
			try (FileWriter writer = new FileWriter(writeName, true);
				  BufferedWriter out = new BufferedWriter(writer)){
				out.write(info);
				out.newLine();
				out.flush();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
