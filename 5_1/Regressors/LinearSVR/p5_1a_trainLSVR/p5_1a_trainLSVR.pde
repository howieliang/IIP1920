//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrainNum.arff"); //load a ARFF dataset
  trainLinearSVR(epsilon=0.1);             //train a SV classifier
  setModelDrawing(unit=2);         //set the model visualization (for 2D features)
  evaluateTrainSet(fold=5, isRegression=true, showEvalDetails=true);  //5-fold cross validation
  saveModel(model="LSVR.model"); //save the model
}

void draw() {
  drawModel(0, 0); //draw the model visualization (for 2D features)
  drawDataPoints(train); //draw the datapoints
  float[] X = {mouseX, mouseY}; 
  double Y = getPredictionIndex(X);
  drawPrediction(X, Y); //draw the prediction
}
