import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

public class MainPredict {
    public static void main(String[] args) throws IOException, ClassNotFoundException {

        String networkPath = "Lab4/src/network.ser";

    
        Scanner scanner = new Scanner(System.in);
        String line = scanner.nextLine();
        scanner.close();

 
        String[] values = line.split(",");
        double[] inputData = Arrays.stream(values).mapToDouble(Double::parseDouble).toArray();

  
        double[] normalizedInput = DataLoader.normalizeInput(inputData);
        System.out.println("tamanho do input "+normalizedInput.length);

        Network network = NetworkUtils.loadNetwork(networkPath);

    
        int predictedLabel = network.predict(normalizedInput);

     
        System.out.println(predictedLabel);
    }
}
