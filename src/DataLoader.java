import java.io.*;
import java.util.*;

public class DataLoader {


    public static List<double[]> loadDataset(String filePath) throws IOException {
        List<double[]> dataset = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] data = Arrays.stream(values).mapToDouble(Double::parseDouble).toArray();
                dataset.add(data);
            }
        }
    
        return dataset;
    }

    public static int[] loadLabels(String filePath) throws IOException {
        List<Integer> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(Integer.parseInt(line));
            }
        }
        return labels.stream().mapToInt(i -> i).toArray();
    }


    public static double[][] normalizeDataset(List<double[]> dataset) {
        double[][] normalizedData = new double[dataset.size()][];
        for (int i = 0; i < dataset.size(); i++) {
            double[] row = dataset.get(i);
            normalizedData[i] = Arrays.stream(row).map(x -> x / 255.0).toArray();
        }
        return normalizedData;
    }


    public static Map<String, Object> splitDataset(double[][] data, int[] labels, double trainRatio) {
        int trainSize = (int) (data.length * trainRatio);
        double[][] trainData = Arrays.copyOfRange(data, 0, trainSize);
        int[] trainLabels = Arrays.copyOfRange(labels, 0, trainSize);
        double[][] testData = Arrays.copyOfRange(data, trainSize, data.length);
        int[] testLabels = Arrays.copyOfRange(labels, trainSize, labels.length);

        Map<String, Object> split = new HashMap<>();
        split.put("trainData", trainData);
        split.put("trainLabels", trainLabels);
        split.put("testData", testData);
        split.put("testLabels", testLabels);

        return split;
    }


        // Método para carregar um único input
        public static double[] loadSingleInput(String filePath) throws IOException {
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line = br.readLine();
                if (line == null || line.isEmpty()) {
                    throw new IOException("Input file is empty or invalid");
                }
                String[] values = line.split(",");
                return Arrays.stream(values).mapToDouble(Double::parseDouble).toArray();
            }
        }
    
        // Método para normalizar um único input
        public static double[] normalizeInput(double[] inputData) {
            return Arrays.stream(inputData).map(x -> x / 255.0).toArray();
        }
}