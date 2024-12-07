import java.io.*;
import java.util.Random;

public class Network implements Serializable {
    private static final long serialVersionUID = 1L; // Identificador para serialização

    Neuron[] hiddenLayer; // Camada oculta com múltiplos neurônios
    Neuron outputNeuron; // Neurônio da camada de saída
    private final double learningRate; // Taxa de aprendizado

    public Network(int numInputs, int numHiddenNeurons, double learningRate) {
        this.learningRate = learningRate;
        Random rd = new Random();
    
        // Inicializar os neurônios da camada oculta com Xavier initialization
        hiddenLayer = new Neuron[numHiddenNeurons];
        for (int i = 0; i < numHiddenNeurons; i++) {
            double[] weights = new double[numInputs];
            for (int j = 0; j < numInputs; j++) {
                // Xavier initialization para pesos
                weights[j] = rd.nextGaussian() * Math.sqrt(1.0 / numInputs);
            }
            double bias = rd.nextGaussian() * Math.sqrt(1.0 / numInputs); // Xavier initialization para bias
            hiddenLayer[i] = new Neuron(weights, bias);
        }
    
        // Inicializar o neurônio de saída com Xavier initialization
        double[] outputWeights = new double[numHiddenNeurons];
        for (int j = 0; j < numHiddenNeurons; j++) {
            // Xavier initialization para pesos
            outputWeights[j] = rd.nextGaussian() * Math.sqrt(1.0 / numHiddenNeurons);
        }
        double outputBias = rd.nextGaussian() * Math.sqrt(1.0 / numHiddenNeurons); // Xavier initialization para bias
        outputNeuron = new Neuron(outputWeights, outputBias);
    }

    // Método para treinar a rede neural
    public void train(double[][] trainData, int[] trainLabels, double[][] testData, int[] testLabels, int maxIterations, double minError) {
        for (int iter = 0; iter < maxIterations; iter++) {
            double totalError = 0;

            for (int i = 0; i < trainData.length; i++) {
        
                double[] hiddenOutputs = new double[hiddenLayer.length];
                for (int j = 0; j < hiddenLayer.length; j++) {
                    hiddenOutputs[j] = hiddenLayer[j].activation(trainData[i]);
                }
                double output = outputNeuron.activation(hiddenOutputs);

            
                double error = trainLabels[i] - output;
                totalError += error * error;

                // Backpropagation
                double outputDelta = error * output * (1 - output);
                double[] hiddenDeltas = new double[hiddenLayer.length];
                for (int j = 0; j < hiddenLayer.length; j++) {
                    hiddenDeltas[j] = outputDelta * outputNeuron.weights[j] * hiddenOutputs[j] * (1 - hiddenOutputs[j]);
                }

      
                for (int j = 0; j < hiddenLayer.length; j++) {
                    outputNeuron.weights[j] += learningRate * outputDelta * hiddenOutputs[j];
                }
                outputNeuron.bias += learningRate * outputDelta;

             
                for (int j = 0; j < hiddenLayer.length; j++) {
                    for (int k = 0; k < hiddenLayer[j].weights.length; k++) {
                        hiddenLayer[j].weights[k] += learningRate * hiddenDeltas[j] * trainData[i][k];
                    }
                    hiddenLayer[j].bias += learningRate * hiddenDeltas[j];
                }
            }

            double trainMSE = totalError / trainData.length;
            double testMSE = MSE(testData, testLabels);
    
            // Exibir progresso do treinamento
            System.out.println("Iteration: " + iter + ", Train MSE: " + trainMSE + ", Test MSE: " + testMSE);
    

            // Parar se o erro médio quadrático for menor que o limite
            if (totalError / trainData.length < minError) break;

            // Opcional: Mostrar progresso do erro
            System.out.println("Iteration: " + iter + ", Error: " + (totalError / trainData.length));
        }
    }

    // Método para testar a rede neural
    public int[] test(double[][] testData) {
        int[] predictions = new int[testData.length];

        for (int i = 0; i < testData.length; i++) {
            double[] hiddenOutputs = new double[hiddenLayer.length];
            for (int j = 0; j < hiddenLayer.length; j++) {
                hiddenOutputs[j] = hiddenLayer[j].activation(testData[i]);
            }
            double output = outputNeuron.activation(hiddenOutputs);
            predictions[i] = output >= 0.5 ? 1 : 0;
        }

        return predictions;
    }

    // Método para fazer uma predição com um único input
    public int predict(double[] inputData) {
        double[] hiddenOutputs = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenOutputs[i] = hiddenLayer[i].activation(inputData);
        }
        double output = outputNeuron.activation(hiddenOutputs);
        return output >= 0.5 ? 1 : 0; // Classificação binária
    }

    // Método para salvar a rede neural em um arquivo
    public void saveNetwork(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
        }
    }

    // Método para carregar a rede neural de um arquivo
    public static Network loadNetwork(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (Network) ois.readObject();
        }
    }

    private double MSE(double[][] data, int[] labels) {
        double totalError = 0.0;
        for (int i = 0; i < data.length; i++) {
     
            double[] hiddenOutputs = new double[hiddenLayer.length];
            for (int j = 0; j < hiddenLayer.length; j++) {
                hiddenOutputs[j] = hiddenLayer[j].activation(data[i]);
            }
            double output = outputNeuron.activation(hiddenOutputs);
    
      
            double error = labels[i] - output; 
            totalError += error * error;
        }
        return totalError / data.length; 
    }
    
    
}
