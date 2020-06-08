//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

ArrayList<Attribute>[] attributes = new ArrayList[2];
Instances[] instances = new Instances[2];
Classifier[] classifiers = new Classifier[2];

void setup() {
  size(500, 500);             //set a canvas
  instances[0] = loadTrainARFFToInstances(dataset="mouseTrainNum.arff");
  instances[1] = loadTrainARFFToInstances(dataset="mouseTrain.arff");
  attributes[0] = loadAttributesFromInstances(instances[0]);
  attributes[1] = loadAttributesFromInstances(instances[1]);
  classifiers[0] = loadModelToClassifier(model="LSVR.model");
  classifiers[1] = loadModelToClassifier(model="LinearSVC.model");
}
void draw() {
  background(255);
  float[] X = {mouseX, mouseY};
  double Y0 = getPredictionIndex(X, classifiers[0], attributes[0]);
  String Y1 = getPrediction(X, classifiers[1], attributes[1],instances[1]);
  drawPrediction(X, Y0, colors[0]);
  drawPrediction(X, Y1, colors[1]);
}
