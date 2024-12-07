import java.util.Random;

public class Network {
    Neuron[] hiddenLayer;
    Neuron outputNeuron;
    private final double learningRate;

    public Network(int numInputs, int numHiddenNeurons, double learningRate) {
        this.learningRate = learningRate;
        Random rd = new Random();

        hiddenLayer = new Neuron[numHiddenNeurons];
        for (int i = 0; i < numHiddenNeurons; i++) {
            hiddenLayer[i] = new Neuron(new double[numInputs], rd.nextDouble());
        }

 
        outputNeuron = new Neuron(new double[numHiddenNeurons], rd.nextDouble());
    }

    public void train(double[][] trainData, int[] trainLabels, double[][] testData, int[] testLabels, int maxIterations, double minError) {
        for (int iter = 0; iter < maxIterations; iter++) {
            double totalError = 0;

            for (int i = 0; i < trainData.length; i++) {
      
                double[] hiddenOutputs = new double[hiddenLayer.length];
                for (int j = 0; j < hiddenLayer.length; j++) {
                    hiddenOutputs[j] = hiddenLayer[j].activation(trainData[i], 0);
                }
                double output = outputNeuron.activation(hiddenOutputs, 0);

                double error = trainLabels[i] - output;
                totalError += error * error;

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

     
            if (totalError / trainData.length < minError) break;

    
            System.out.println("Iteration: " + iter + ", Error: " + (totalError / trainData.length));
        }
    }

    public int[] test(double[][] testData) {
        int[] predictions = new int[testData.length];

        for (int i = 0; i < testData.length; i++) {
            double[] hiddenOutputs = new double[hiddenLayer.length];
            for (int j = 0; j < hiddenLayer.length; j++) {
                hiddenOutputs[j] = hiddenLayer[j].activation(testData[i], 0);
            }
            double output = outputNeuron.activation(hiddenOutputs, 0);
            predictions[i] = output >= 0.5 ? 1 : 0;
        }

        return predictions;
    }
}
