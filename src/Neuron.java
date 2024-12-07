import java.io.Serializable;

public class Neuron implements Serializable {
    private static final long serialVersionUID = 1L; 

    double[] weights;
    double bias;

    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    private double weightSum(double[] inputs) {
        double result = bias;
        for (int i = 0; i < inputs.length; i++) {
            result += inputs[i] * weights[i];
        }
        return result;
    }

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public double activation(double[] inputs) {
        return sigmoid(weightSum(inputs));
    }
}
