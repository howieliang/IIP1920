//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="testData.arff"); //load a ARFF dataset
  println(train);
  trainLinearSVC(C=64);               //train a KNN classifier
  setModelDrawing(unit=2);         //set the model visualization (for 2D features)
  evaluateTrainSet(fold=5, showEvalDetails=true);  //5-fold cross validation
  saveSVC(model="LinearSVC.model"); //save the model
}

void draw() {
  drawModel(0, 0); //draw the model visualization (for 2D features)
  drawDataPoints(); //draw the datapoints
  float[] X = {mouseX, mouseY}; 
  String Y = getPrediction(X);
  drawPrediction(X, Y); //draw the prediction
}
