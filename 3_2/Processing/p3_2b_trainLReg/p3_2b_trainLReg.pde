//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrainNum.arff"); //load a ARFF dataset
  trainLinearRegression();    //train a linear regressor
  setModelDrawing(unit=2);    //set the model visualization (for 2D features)
  evaluateTrainSet(fold=5, isRegression=true, showEvalDetails=true);  //5-fold cross validation
  saveModel(model="LinearReg.model"); //save the model
}

void draw() {
  drawModel(0, 0); //draw the model visualization (for 2D features)
  drawDataPoints(train); //draw the datapoints
  float[] X = {mouseX, mouseY}; 
  //String Y = getPrediction(X);
  double Y = getPredictionIndex(X);
  drawPrediction(X, Y); //draw the prediction
}
