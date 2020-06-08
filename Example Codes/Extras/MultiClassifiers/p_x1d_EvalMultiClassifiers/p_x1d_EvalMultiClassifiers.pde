//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

ArrayList<Attribute>[] attributes = new ArrayList[1];
Instances[] instances = new Instances[1];
Classifier[] classifiers = new Classifier[3];

void setup() {
  size(500, 500);             //set a canvas
  instances[0] = loadTrainARFFToInstances(dataset="mouseTrain.arff");
  attributes[0] = loadAttributesFromInstances(instances[0]);
  classifiers[0] = loadModelToClassifier(model="RBFSVC.model"); //load a pretrained model.
  classifiers[1] = loadModelToClassifier(model="LinearSVC.model"); //load a pretrained model.
  classifiers[2] = loadModelToClassifier(model="KNN.model"); //load a pretrained model.
  loadTestARFF(dataset="mouseTrain.arff");//load a ARFF dataset
  evaluateTestSet(classifiers[0],test,isRegression = false, showEvalDetails=true);  //5-fold cross validation
  evaluateTestSet(classifiers[1],test,isRegression = false, showEvalDetails=true);  //5-fold cross validation
  evaluateTestSet(classifiers[2],test,isRegression = false, showEvalDetails=true);  //5-fold cross validation
}
void draw() {
  background(255);
  float[] X = {mouseX, mouseY};
  String[] Y = new String[classifiers.length];
  for(int i = 0 ; i < classifiers.length ; i++){
    Y[i] = getPrediction(X, classifiers[i], attributes[0],instances[0]);
    drawPrediction(X, Y[i], colors[i]); //draw the prediction
  }
}
