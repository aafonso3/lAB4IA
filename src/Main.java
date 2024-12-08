import java.io.IOException;
import java.util.List;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws IOException, ClassNotFoundException {

        List<double[]> dataset = DataLoader.loadDataset("dataset/dataset.csv");
        int[] labels = DataLoader.loadLabels("dataset/labels.csv");

        double[][] normalizedData = DataLoader.normalizeDataset(dataset);
        Map<String, Object> split = DataLoader.splitDataset(normalizedData, labels, 0.8);
        double[][] trainData = (double[][]) split.get("trainData");
        int[] trainLabels = (int[]) split.get("trainLabels");
        double[][] testData = (double[][]) split.get("testData");
        int[] testLabels = (int[]) split.get("testLabels");


        Network network = new Network(400, 5, 0.05);
        network.train(trainData, trainLabels, testData, testLabels,  50000,  1e-4,0.5);


        int[] predictions = network.test(testData);

        calculateAccuracy(predictions, testLabels);

     
        NetworkUtils.saveNetwork(network, "src/network.ser");

    }

    public static void calculateAccuracy(int[] predictions, int[] actualLabels) {
        int TP = 0, TN = 0, FP = 0, FN = 0;
    
        for (int i = 0; i < predictions.length; i++) {
            if (actualLabels[i] == 1 && predictions[i] == 1) {
                TP++;
            } else if (actualLabels[i] == 0 && predictions[i] == 0) {
                TN++;
            } else if (actualLabels[i] == 0 && predictions[i] == 1) {
                FP++;
            } else if (actualLabels[i] == 1 && predictions[i] == 0) {
                FN++;
            }
        }
    
        double accuracy = (double) (TP + TN) / (double)(TP + TN + FP + FN);
        System.out.println("TP: " + TP + ", TN: " + TN + ", FP: " + FP + ", FN: " + FN);
        System.out.println("Accuracy: " + accuracy);
    }
}
