//====================JASON ZONE DO NOT TOUCH==========================
//The DenseLayer is used to make it easier to code the architecutre of each neural network, because it automatically updates all of the neurons' weights and biases for you, so you don't have to use as many loops to do this.
public class DenseLayer {
  private Neuron[] dense;
  private double[] a;
  private String activation;
  private int previousLayerSize;
  private double[] previousActivations;

  public DenseLayer(int neurons, String activation, int previousLayerSize) {
    dense = new Neuron[neurons];
    for (int i = 0; i < dense.length; i++) {
      dense[i] = new Neuron(previousLayerSize);
    }
    a = new double[dense.length];
    this.activation = activation;
    this.previousLayerSize = previousLayerSize;
  }

  public void updateActivations(double[] x) { //updates the activations in each neuron of the layer, for the first layer
    previousActivations = x;
    for (int i = 0; i < dense.length; i++) {
      if (this.activation.equals("ReLU")) {
        a[i] = dense[i].ReLU_Activation(x);
      } else if (this.activation.equals("Sigmoid")) {
        a[i] = dense[i].Sigmoid_Activation(x);
      }
    }
  }

  public void updateActivations(DenseLayer previousLayer) { //same as above, but for the second layer. I had to do this in two seperate methods because each layer will take in a different data type.
    previousActivations = previousLayer.getActivations();
    for (int i = 0; i < dense.length; i++) {
      if (this.activation.equals("ReLU")) {
        a[i] = dense[i].ReLU_Activation(previousActivations);
      } else if (this.activation.equals("Sigmoid")) {
        a[i] = dense[i].Sigmoid_Activation(previousActivations);
      }
    }
  }

  public double[] getActivations() {
    return this.a;
  }

  public Neuron[] getDense() {
    return this.dense;
  }

  public String getActivation() {
    return this.activation;
  }

  public double[] getPreviousActivations() {
    return this.previousActivations;
  }

  public int getPreviousLayerSize() {
    return this.previousLayerSize;
  }
}
// ====================END OF JASON ZONE==========================