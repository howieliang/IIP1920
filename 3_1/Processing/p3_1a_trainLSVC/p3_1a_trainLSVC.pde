//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrain.arff"); //load a ARFF dataset
  trainLinearSVC(C=1);             //train a SV classifier
  setModelDrawing(unit=2);         //set the model visualization (for 2D features)
  evaluateTrainSet(fold=5, isRegression=false, showEvalDetails=true);  //5-fold cross validation
  saveModel(model="LinearSVC.model"); //save the model
}

void draw() {
  drawModel(0, 0); //draw the model visualization (for 2D features)
  drawDataPoints(train); //draw the datapoints
  float[] X = {mouseX, mouseY}; 
  String Y = getPrediction(X);
  drawPrediction(X, Y); //draw the prediction
}
