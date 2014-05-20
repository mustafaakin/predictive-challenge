import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.google.gson.Gson;

public class Test {
	static String readFile(String path) throws IOException {
		Charset encoding = StandardCharsets.UTF_8;
		byte[] encoded = Files.readAllBytes(Paths.get(path));
		return new String(encoded, encoding);
	}

	final static String path = "/home/mustafa/machine/train/";

	public static void main(String[] args) throws Exception {
		Gson gson = new Gson();
		for(int i = 0; i < 100; i++){
			Host h = gson.fromJson(readFile(path + i + ".json"), Host.class);
			for(Page p : h.pages){
				Series s = p.series_1h;
				for(int x : s.visits){
					System.out.print(x + " ");
				}
				System.out.println();
			}		
			
		}
		// Series p = gson.fromJson(readFile(path + "test.json"), Series.class);
		
	}
}
