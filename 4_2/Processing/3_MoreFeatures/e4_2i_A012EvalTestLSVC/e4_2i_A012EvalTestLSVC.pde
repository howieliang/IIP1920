//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

void setup() {
  size(500, 500, P2D);
  loadTrainARFF(dataset="A012GestTrain.arff");//load a ARFF dataset
  loadTestARFF(dataset="A012GestTest.arff");//load a ARFF dataset
  loadModel(model="LinearSVC.model"); //load a pretrained model.
  evaluateTestSet(isRegression = false, showEvalDetails=true);  //5-fold cross validation
}

void draw() {
  background(255);
}
