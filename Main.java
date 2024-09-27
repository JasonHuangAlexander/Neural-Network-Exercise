Date Started: April 25

Purpose: This product will use a neural network to predict whether or not a student will get rescinded from the college they plan to attend. This will help current seniors like us by alleviating stress about their senior year grades, helping them gauge whether or not their grades are too low. It will help boost mental health as a result. 

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOError;
import java.util.Scanner;
import java.util.ArrayList;

class Main {
  public static void main(String[] args) {
    ArrayList<Integer> y = load_y(); // labels
    ArrayList<double[]> X = load_X(); // features
    double[] L_0 = new double[(X.get(0)).length]; // input layer
    Scanner scanner = new Scanner(System.in);
    System.out.print("Welcome to the \"Will I Get Rescinded?\" app!");
    System.out.print("\nAre you Amy Zheng? Y/N:");
    String input = scanner.nextLine();
    if (input.equals("Y")) {
      System.out.println("You are getting rescinded :(");
    } else if (input.equals("N")){
      double[] x = menu();
      int onevotes = 0;
      int zerovotes = 0;
      for (int i = 0; i < 1; i++) { //number of MLPs
        System.out.print("\n===================================================\n");
        System.out.print("Network " + (i + 1) + " of 3 in voting ensemble:");
        System.out.print("\n===================================================\n");
        DenseLayer L_1 = new DenseLayer(3, "ReLU", 2); // first hidden layer
        DenseLayer L_2 = new DenseLayer(1, "Sigmoid", 3); // output layer
        train_model(y, X, L_0, L_1, L_2);
        int vote = predict(x, L_1, L_2);
        if (vote == 1) {
          onevotes++;
        } else if (vote == 0) {
          zerovotes++;
        }
      }
      
      
      
      if (zerovotes > onevotes) {
        System.out.print("\nYou won't be rescinded! :)\n");
      } 
      else {
        System.out.print("\nYou will be rescinded :(\n");
      }
    }
  }

  public static int predict(double[] x, DenseLayer L_1, DenseLayer L_2) {
    double[] L_0 = x; // set input layer to training features in batch
    L_1.updateActivations(L_0);
    L_2.updateActivations(L_1);
    if ( L_2.getActivations()[0]>0.5){ //threshold at 0.5
      return 1;
    }
    else{
      return 0;
    }
  }

  public static void train_model(ArrayList<Integer> y, ArrayList<double[]> X, double[] L_0, DenseLayer L_1, DenseLayer L_2) {
    //this trains the neural network so that we can later use it to make predictions.
    //You will notice that when this is working correctly, the yhats of each training example vary greatly from positive to negative examples. When gradient descent is failing to occur, the yhat converges at roughly 0.50. It might take several tries for you get gradient descent to work properly. As explained in the video, this is due to math problems which are quite complex, and it is unfeasable to fix this issue using only Java with no external ML libraries.
    DenseLayer[] MLP = new DenseLayer[2]; // multi-layer perceptron
    MLP[0] = L_1;
    MLP[1] = L_2;
    System.out.print("\nShape of y: " + y.size());
    System.out.print("\nShape of X: " + X.size() + "," + X.get(0).length + "\n");
    for (int epochs = 0; epochs < 512; epochs++) {
      double avgLoss = 0.0;
      System.out.print("===================================================");
      System.out.print("\nEpoch " + (epochs + 1) + ":\n");
      int ones = 0;
      int zeros = 0;
      for (int i_1 = 0; i_1 < 10; i_1++) { // batch sizes of 10
        int index = (int) (Math.random() * X.size());
        if (y.get(index) == 1 && ones > 5) {
          i_1--;
          continue;
        }
        if (y.get(index) == 0 && zeros > 5) {
          i_1--;
          continue;
        }
        if (y.get(index) == 1) {
          ones++;
        }
        if (y.get(index) == 0) {
          zeros++;
        }
        double[] x = X.get(index); // random training example x

        // carry out forward propagation:
        L_0 = x; // set input layer to training features in batch
        L_1.updateActivations(L_0);
        L_2.updateActivations(L_1);

        // calculate MSE loss:
        double[] yhats = L_2.getActivations();
        for (double yhat : yhats) {
          System.out.print("\n     --------------------------------------");
          System.out.print("\n      yhat_" + i_1 + ": " + yhat);
          System.out.print("    y_" + i_1 + ": " + y.get(index));
          double logLoss = (-y.get(index) * Math.log(yhat) - (1 - y.get(index)) * Math.log(1 - yhat));
          System.out.print("\n      loss: " + logLoss);
          avgLoss = avgLoss + logLoss;
          // gradient descent:
          for (int L = MLP.length - 1; L >= 0; L--) { // for each layer
            double alpha = 0.01;
            Neuron[] L_i = MLP[L].getDense();
            for (int n = 0; n < L_i.length; n++) { // for each neuron
              ArrayList<Double> update_w = new ArrayList<Double>();
              Neuron n_i = L_i[n];
              double z = n_i.get_z();
              ArrayList<Double> w = n_i.get_w();
              double b = n_i.get_b();
              for (int i = 0; i < w.size(); i++) { // for each weight w
                double dz_dw = (MLP[L].getPreviousActivations())[i]; // activation from layer L-1
                double da_dz = 0.0;
                if (MLP[L].getActivation().equals("Sigmoid")) {
                  da_dz = (Math.exp(-1 * z) / Math.pow((1 + Math.exp(-1 * z)), 2));
                } else if (MLP[L].getActivation().equals("ReLU")) {
                  if (n_i.get_z() > 0) {
                    da_dz = 1.0;
                  } else if (n_i.get_z() <= 0) {
                    da_dz = 0.0;
                  }
                }
                double dJ_da = 2*(yhat-y.get(index));
                double dJ_dw = dz_dw * da_dz * dJ_da;
                update_w.add(dJ_dw);
              }
              // for bias b
              double dz_db = 1.0;
              double da_dz = 0.0;
              if (MLP[L].getActivation().equals("Sigmoid")) {
                da_dz = (Math.exp(-1 * z) / Math.pow((1 + Math.exp(-1 * z)), 2));
              } else if (MLP[L].getActivation().equals("ReLU")) {
                if (n_i.get_z() > 0) {
                  da_dz = 1.0;
                } else if (n_i.get_z() <= 0) {
                  da_dz = 0.0;
                }
              }
              double dJ_da = 2*(yhat-y.get(index));
              double dJ_db = dz_db * da_dz * dJ_da;
              n_i.set_b(n_i.get_b() - alpha * dJ_db);

              for (int i = 0; i < update_w.size(); i++) { // update weights
                update_w.set(i, ((n_i.get_w()).get(i) - alpha * update_w.get(i)));
              }
              n_i.set_w(update_w);
            }
          }
        }
      }
      avgLoss = (avgLoss / X.size());
      System.out.print("\n\nMean binary crossentropy loss: " + avgLoss + "\n");
      avgLoss = 0.0;
    }
  }

  public static ArrayList<double[]> load_X() {
    ArrayList<double[]> X = new ArrayList<double[]>();
    try {
      File traincsv = new File("train.csv");
      Scanner fileScanner = new Scanner(traincsv);
      boolean skip = false;
      while (fileScanner.hasNextLine()) {
        String row = fileScanner.nextLine();
        if (skip) {
          double[] temp = new double[2];
          if (row.substring(3, 4).equals(",")) {
            temp[1] = Double.parseDouble(row.substring(4));
            temp[0] = Double.parseDouble(row.substring(2, 3));
          } else if (row.substring(4, 5).equals(",")) {
            temp[1] = Double.parseDouble(row.substring(5));
            temp[0] = Double.parseDouble(row.substring(2, 4));
          } else {
            temp[1] = Double.parseDouble(row.substring(6));
            temp[0] = Double.parseDouble(row.substring(2, 5));
          }
          X.add(temp);
        }
        skip = true;
      }
      return X;
    } catch (FileNotFoundException e) {
      System.out.println("File not found!");
      return X;
    }
  }

  public static ArrayList<Integer> load_y() {
    ArrayList<Integer> y = new ArrayList<Integer>();
    try {
      File traincsv = new File("train.csv");
      Scanner fileScanner = new Scanner(traincsv);
      boolean skip = false;
      while (fileScanner.hasNextLine()) {
        String row = fileScanner.nextLine();
        if (skip) {
          y.add(Integer.valueOf(row.substring(0, 1)));
        }
        skip = true;
      }
      return y;
    } catch (FileNotFoundException e) {
      System.out.println("File not found!");
      return y;
    }
  }
//The following function runs the user interface, asking the user to enter their grade in a class, the number of credits the class is worth, and whether or not they wish to enter another class. This happens for three intervals---the pre-senior year grades (freshman through junior year), the senior grades in the first semester (mid-senior year grades), and the senior grades at the end of the year (end-of-senior year grades)
  public static double[] menu() {
    Scanner scanner = new Scanner(System.in);
    double [] info = new double [2];
    char newClass;
    String grade = "";
    int credits;
    int classCount = 1;
    ArrayList<String> letterGradeArr = new ArrayList<String>();
    double gpaSum = 0;
    double creditSum = 0;
    int numPreSenior = 0;
    
    System.out.print("\nPlease enter the ranking of your school according to US News and World report overall school ranking: ");
    int ranking = scanner.nextInt();

    do {
      System.out.print("\nPlease enter your pre-senior year (class " + classCount + ") letter grade: ");
      grade = scanner.next();
      letterGradeArr.add(grade);
      System.out.print("Please enter the number of credits (class " + classCount + ") is worth: ");
      credits = scanner.nextInt();
      creditSum = creditSum + credits;
      gpaSum = gpaSum + letterToNum(grade, credits);
      System.out.print("\nDo you wish to enter another pre-senior year class (Y/N)? ");
      newClass = scanner.next().charAt(0);
      classCount++;
      numPreSenior++;
    } while (newClass != 'N');

    ArrayList<String> finLetterGradeArr = letterGradeArr;
    int finClassCount = classCount;
    double finGpaSum = gpaSum;
    double finCreditSum = creditSum;

    System.out.println("\n-----------------------------------------------------------"
        + "\n\nNow you will enter your senior year grade averages by the end of 1st semester.");
    do {
      System.out.print("\nPlease enter your mid-senior year (class " + classCount + ") letter grade: ");
      grade = scanner.next();
      letterGradeArr.add(grade);
      System.out.print(
          "\nIf your class is a semester class, enter double your credit amount as mid-senior year averages are calculated with half the weight of end-of senior year averages\n\nPlease enter the number of credits (class "
              + classCount + ") is worth: ");
      credits = scanner.nextInt();
      creditSum = creditSum + (credits / 2);
      gpaSum = gpaSum + letterToNum(grade, credits);
      System.out.print("\nDo you wish to enter another mid-senior year class (Y/N)? ");
      newClass = scanner.next().charAt(0);
      classCount++;
    } while (newClass != 'N');

    System.out.println("\n-----------------------------------------------------------"
        + "\n\nNow you will enter your senior year grade averages by the end of the year.");
    int fCount = 0;
    do {
      System.out.print("\nPlease enter your end-of-senior year (class " + finClassCount + ") letter grade: ");
      grade = scanner.next();
      finLetterGradeArr.add(grade);
      
      if(grade == "F")
      {
        fCount++;
      }
      
      System.out.print("Please enter the number of credits (class " + finClassCount + ") is worth: ");
      credits = scanner.nextInt();
      finCreditSum = finCreditSum + credits;
      finGpaSum = finGpaSum + letterToNum(grade, credits);
      System.out.print("\nDo you wish to enter another end-of-senior year class (Y/N)? ");
      newClass = scanner.next().charAt(0);
      finClassCount++;
    } while (newClass != 'N');

    System.out.print("\n\nYour GPA by the end of 1st semester is " + (gpaSum / 4 * creditSum) + ".");
    System.out.print("\nYour GPA by the end of the year is " + (finGpaSum / 4 * finCreditSum) + ".");

    double midGPA = gpaCalc(gpaSum, creditSum);
    double finGPA = gpaCalc(finGpaSum, finCreditSum);
    
    info[0] = ranking;
    info[1] = finGPA-midGPA;
    
    //If the user has more than three failing grades at the end of senior year, the user is automatically rescinded, which is ensured by the following conditional statement
    if(fCount>=3)
    {
      info[1] = -5;
    }
    
    return info;
  }

  //The following function converts an inputed grade to a GPA number
  public static double letterToNum(String letter, int credits) {
    double gpa = 0;

    if (letter.equals("A+")) {
      gpa = 4.3;
    } else if (letter.equals("A")) {
      gpa = 4.0;
    } else if (letter.equals("A-")) {
      gpa = 3.7;
    } else if (letter.equals("B+")) {
      gpa = 3.3;
    } else if (letter.equals("B")) {
      gpa = 3.0;
    } else if (letter.equals("B-")) {
      gpa = 2.7;
    } else if (letter.equals("C+")) {
      gpa = 2.3;
    } else if (letter.equals("C")) {
      gpa = 2.0;
    } else if (letter.equals("C-")) {
      gpa = 1.7;
    } else if (letter.equals("D+")) {
      gpa = 1.3;
    } else if (letter.equals("D")) {
      gpa = 1.0;
    } else if (letter.equals("D-")) {
      gpa = 0.7;
    } else if (letter.equals("F")) {
      gpa = 0.0;
    }

    gpa = gpa * credits;
    return gpa;
  }
 
  //The following function calculates GPA based on the sum of GPA numbers correlated with each inputted letter grade and multiplied by the number of credits for each respective class, divided by the total number of credits.
  public static double gpaCalc(double gpaSum, double credits)
  {
   double gpa = gpaSum/credits;
   return gpa;
  }

}
