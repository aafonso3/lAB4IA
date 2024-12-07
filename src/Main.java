import java.io.IOException;
import java.util.List;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        // Carregar dataset
        List<double[]> dataset = DataLoader.loadDataset("Lab4/dataset/dataset.csv");
        int[] labels = DataLoader.loadLabels("Lab4/dataset/labels.csv");

        // Normalizar e dividir o dataset
        double[][] normalizedData = DataLoader.normalizeDataset(dataset);
        Map<String, Object> split = DataLoader.splitDataset(normalizedData, labels, 0.8);
        double[][] trainData = (double[][]) split.get("trainData");
        int[] trainLabels = (int[]) split.get("trainLabels");
        double[][] testData = (double[][]) split.get("testData");
        int[] testLabels = (int[]) split.get("testLabels");

        // Inicializar e treinar a rede
        Network network = new Network(400, 200, 0.2);
        network.train(trainData, trainLabels, testData, testLabels, 500, 0.01);

        // Salvar a rede treinada
        NetworkUtils.saveNetwork(network, "Lab4/src/network.ser");

        System.out.println("Rede neural treinada e salva em 'network.ser'");
    }
}
