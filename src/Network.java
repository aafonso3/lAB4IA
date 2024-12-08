import java.io.*;
import java.util.Random;

public class Network implements Serializable {
    private static final long serialVersionUID = 1L; 

    Neuron[] hiddenLayer; 
    Neuron outputNeuron; 
    private final double learningRate; 

    public Network(int numInputs, int numHiddenNeurons, double learningRate) {
        this.learningRate = learningRate;
        Random rd = new Random();
    
   
        hiddenLayer = new Neuron[numHiddenNeurons];
        for (int i = 0; i < numHiddenNeurons; i++) {
            double[] weights = new double[numInputs];
            for (int j = 0; j < numInputs; j++) {
               
                weights[j] = rd.nextGaussian() * Math.sqrt(1.0 / numInputs);
            }
            double bias = rd.nextGaussian() * Math.sqrt(1.0 / numInputs); 
            hiddenLayer[i] = new Neuron(weights, bias);
        }
    
       
        double[] outputWeights = new double[numHiddenNeurons];
        for (int j = 0; j < numHiddenNeurons; j++) {
            
            outputWeights[j] = rd.nextGaussian() * Math.sqrt(1.0 / numHiddenNeurons);
        }
        double outputBias = rd.nextGaussian() * Math.sqrt(1.0 / numHiddenNeurons); 
        outputNeuron = new Neuron(outputWeights, outputBias);
    }


    public void train(double[][] trainData, int[] trainLabels, double[][] testData, int[] testLabels, int maxIterations, double minError, double momentum) {

        double[][] hiddenMomentum = new double[hiddenLayer.length][trainData[0].length];
        double[] outputMomentum = new double[hiddenLayer.length];
    
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
    
   
                double outputDelta = error * output * (1 - output);
             
                double[] hiddenDeltas = new double[hiddenLayer.length];
                for (int j = 0; j < hiddenLayer.length; j++) {
                    hiddenDeltas[j] = outputDelta * outputNeuron.weights[j] * hiddenOutputs[j] * (1 - hiddenOutputs[j]);
                }
    
      
                for (int j = 0; j < hiddenLayer.length; j++) {
                    double delta = learningRate * outputDelta * hiddenOutputs[j];
                    outputMomentum[j] = momentum * outputMomentum[j] + delta;
                    outputNeuron.weights[j] += outputMomentum[j];
                }
                outputNeuron.bias += learningRate * outputDelta;
    
      
                for (int j = 0; j < hiddenLayer.length; j++) {
                    for (int k = 0; k < hiddenLayer[j].weights.length; k++) {
                        double delta = learningRate * hiddenDeltas[j] * trainData[i][k];
                        hiddenMomentum[j][k] = momentum * hiddenMomentum[j][k] + delta;
                        hiddenLayer[j].weights[k] += hiddenMomentum[j][k];
                       
                    }
                    hiddenLayer[j].bias += learningRate * hiddenDeltas[j];
                }
            }
    
            double trainMSE = totalError / trainData.length;
            double testMSE = MSE(testData, testLabels);
    
            System.out.println("Iteration: " + iter + ", Train MSE: " + trainMSE + ", Test MSE: " + testMSE);
    
            if (trainMSE < minError) break;
        }
    }
    

    public int[] test(double[][] testData) {
        int[] predictions = new int[testData.length];
    
        for (int i = 0; i < testData.length; i++) {
            double[] hiddenOutputs = new double[hiddenLayer.length];
            for (int j = 0; j < hiddenLayer.length; j++) {
                hiddenOutputs[j] = hiddenLayer[j].activation(testData[i]);
            }
            double output = outputNeuron.activation(hiddenOutputs);
            System.out.println("output test " + output);
     
            predictions[i] = (int) Math.round(output);
        }
    
        return predictions;
    }
    


    public int predict(double[] inputData) {
        double[] hiddenOutputs = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenOutputs[i] = hiddenLayer[i].activation(inputData);
        }
        double output = outputNeuron.activation(hiddenOutputs);
        System.out.println(output);
    
        return output >= 0.5 ? 1 : 0;
    }
    

    public void saveNetwork(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
        }
    }


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
