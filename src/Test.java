import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.AdditiveRegression;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.MultiScheme;
import weka.classifiers.meta.RegressionByDiscretization;
import weka.classifiers.meta.Stacking;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import com.google.gson.Gson;

public class Test {

	static String readFile(String path) throws IOException {
		Charset encoding = StandardCharsets.UTF_8;
		byte[] encoded = Files.readAllBytes(Paths.get(path));
		return new String(encoded, encoding);
	}

	final static String path = "/home/mustafa/machine/train/";

	synchronized static void log(String msg) {
		System.out.println(msg);
	}

	static class Data {
		ArrayList<Attribute> attrs = new ArrayList<>();

		Instances train;
		Instances test;

		int resVisit, resFacebook, resTwitter;
		int resActive;

		boolean cumulative;

		public Data(boolean cumulative, int resVisit, int resFacebook, int resTwitter, int resActive, ArrayList<Page> pages, double trainPercentage) {
			this.resActive = resActive;
			this.resFacebook = resFacebook;
			this.resTwitter = resTwitter;
			this.resVisit = resVisit;
			this.cumulative = cumulative;

			int size = resVisit + resFacebook + resTwitter + resActive;
			for (int i = 0; i < size; i++) {
				attrs.add(new Attribute("data" + i));
			}
			attrs.add(new Attribute("totalvisits"));

			train = new Instances("train", attrs, (int) (pages.size() * (0.1 + trainPercentage)));
			train.setClassIndex(size);
			test = new Instances("test", attrs, (int) (pages.size() * (1.1 - trainPercentage)));
			test.setClassIndex(size);

			for (int i = 0; i < pages.size(); i++) {
				Page p = pages.get(i);
				Series s = p.series_1h;
				Instance instance = getData(s, p.sum_visits_48h);
				if (i < pages.size() * trainPercentage) {
					train.add(instance);
				} else {
					test.add(instance);
				}
			}
		}

		public int[] compressData(boolean cumulative, int resolution, int[] data) {
			int[] result = new int[resolution];
			if (resolution == 0)
				return result;
			// Compress data
			for (int i = 0; i < 12; i++) {
				int place = i / (12 / resolution);
				result[place] += data[i];
			}
			// If cumulative, sum data
			if (cumulative) {
				for (int i = 0; i < result.length - 1; i++) {
					result[i + 1] = result[i + 1] + result[i];
				}
			}
			return result;
		}

		public Instance getData(Series s, int visitCount) {
			int[] facebookData = compressData(cumulative, resFacebook, s.facebook);
			int[] twitterData = compressData(cumulative, resTwitter, s.twitter);
			int[] visitData = compressData(cumulative, resVisit, s.visits);
			int[] avgVisit;
			if (resActive == 0) {
				avgVisit = new int[0];
			} else {
				avgVisit = s.average_active_time;
			}
			int size = facebookData.length + twitterData.length + visitData.length + avgVisit.length;

			int[] all = new int[size];
			// So ugly
			System.arraycopy(facebookData, 0, all, 0, facebookData.length);
			System.arraycopy(twitterData, 0, all, facebookData.length, twitterData.length);
			System.arraycopy(visitData, 0, all, facebookData.length + twitterData.length, visitData.length);
			System.arraycopy(avgVisit, 0, all, facebookData.length + twitterData.length + visitData.length, avgVisit.length);

			Instance instance = new DenseInstance(size + 1);
			for (int i = 0; i < size; i++) {
				instance.setValue(i, all[i]);
			}
			instance.setValue(size, visitCount);

			return instance;
		}
	}

	public static void main(String[] args) throws Exception {
		Gson gson = new Gson();
		final int HOST_COUNT = 10;

		/*
		 * int[] data = new int[]{1,2,3,4,5,6,7,8,9,10,11,12}; int[] aq =
		 * Data.compressData(true, 2, data);
		 * 
		 * for(int i : aq){ System.out.print(i + " "); } System.exit(1);
		 */

		log("Reading JSON data");
		final ArrayList<Page> pages = new ArrayList<Page>();
		for (int i = 0; i < HOST_COUNT; i++) {
			Host h = gson.fromJson(readFile(path + i + ".json"), Host.class);
			for (Page p : h.pages) {
				if (p.sum_visits_48h < 150)
					pages.add(p);
			}
		}
		log("Read JSON");

		Collections.shuffle(pages);

		int[] values = new int[] { 6, 12 };
		int[] visits = new int[] { 0, 12 };
		boolean[] cumulatives = new boolean[] { true, false };
		int cores = Runtime.getRuntime().availableProcessors();
		ExecutorService executor = Executors.newFixedThreadPool(cores);

		final Class[] types = { ZeroR.class, RegressionByDiscretization.class, Bagging.class, AdditiveRegression.class, MultilayerPerceptron.class,
				SimpleLinearRegression.class, LinearRegression.class, RandomForest.class, RandomTree.class, SMOreg.class, KStar.class};

		for (final int v1 : values) {
			for (final int v2 : values) {
				for (final int v3 : values) {
					for (final int v4 : visits) {
						for (final boolean b : cumulatives) {
							for (final Class clazz : types) {
								Runnable r = new Runnable() {
									public void run() {
										long start = System.currentTimeMillis();
										Data d = new Data(b, v1, v2, v3, v4, pages, 0.95);
										try {
											Classifier model = (Classifier) clazz.getConstructors()[0].newInstance();
											model.buildClassifier(d.train);
											Evaluation eTest = new Evaluation(d.train);
											eTest.evaluateModel(model, d.test);

											long end = System.currentTimeMillis();

											int time = (int) (end - start);
											log(String.format("%30s %5b %2d %2d %2d %2d %7.3f %5d", model.getClass().getSimpleName(), b, v1, v2, v3, v4,
													eTest.rootRelativeSquaredError(), time));
										} catch (Exception e) {

											// Like i give a fuck
											// e.printStackTrace();
										}
									};
								};
								executor.execute(r);
							}
						}
					}
				}
			}
		}

		executor.shutdown();

		/*
		 * for (Instance i : d.test) { double guessed =
		 * model.classifyInstance(i); double actual = i.classValue(); for(int j
		 * = 0; j < i.numAttributes(); j++){ double dx = i.value(j);
		 * //System.out.print((int)dx + " "); } // System.out.println();
		 * log(String.format("Guess: %5.0f Actual: %5.0f Diff: %5.0f", guessed,
		 * actual, Math.abs(guessed - actual))); }
		 */
	}
}
