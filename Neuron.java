
//====================JASON ZONE DO NOT TOUCH==========================
import java.util.ArrayList;
//The Neuron class holds the weights and biases of each neuron in each layer of each neural network.
//These weights and biases are used to predict if you will get rescinded or not.
public class Neuron {
  private ArrayList<Double> w = new ArrayList<Double>();
  private double b;
  private double z;

  public Neuron(int previousLayerSize) {
    for (int i = 0; i < previousLayerSize; i++) {
      w.add(Math.random() * 3); 
    }
    this.b = Math.random() * 3; 
  }

  public double ReLU_Activation(double[] x) { //we have two activations, ReLU and Sigmoid. ReLU is used in the first layer, Sigmoid in the second, to predict whether or not you will be rescinded. 
    z = 0.0;
    double dot_product = 0.0;
    for (int i = 0; i < x.length; i++) {
      dot_product += x[i] * w.get(i);
    }
    z = dot_product + b;
    double a = 0.0;
    if (z > 0.0) {
      a = z;
    }
    return a;
  }

  public double Sigmoid_Activation(double[] x) {
    z = 0.0;
    double dot_product = 0.0;
    for (int i = 0; i < x.length; i++) {
      dot_product += x[i] * w.get(i);
    }
    z = dot_product + b;
    double a = 1 / (1 + Math.exp(-1 * z));
    return a;
  }

  public void set_w(ArrayList<Double> w) {
    this.w = w;
  }

  public void set_b(double b) {
    this.b = b;
  }

  public ArrayList<Double> get_w() {
    return this.w;
  }

  public double get_b() {
    return this.b;
  }

  public double get_z() {
    return this.z;
  }
}
// ====================END OF JASON ZONE==========================