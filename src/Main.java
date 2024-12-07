import java.io.IOException;
import java.util.List;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws IOException {
  
        List<double[]> dataset = DataLoader.loadDataset("dataset/dataset.csv");
        int[] labels = DataLoader.loadLabels("dataset/labels.csv");

   
        double[][] normalizedData = DataLoader.normalizeDataset(dataset);

    
        Map<String, Object> split = DataLoader.splitDataset(normalizedData, labels, 0.8);
        double[][] trainData = (double[][]) split.get("trainData");
        int[] trainLabels = (int[]) split.get("trainLabels");
        double[][] testData = (double[][]) split.get("testData");
        int[] testLabels = (int[]) split.get("testLabels");

  
        Network network = new Network(400, 1, 0.9); 

 
        network.train(trainData, trainLabels, testData, testLabels, 1000, 0.01);

        int[] output = network.test(testData);

      
        for (int i = 0; i < output.length; i++) {
            System.out.println("Expected: " + testLabels[i] + ", Predicted: " + output[i]);
        }
    }
}
