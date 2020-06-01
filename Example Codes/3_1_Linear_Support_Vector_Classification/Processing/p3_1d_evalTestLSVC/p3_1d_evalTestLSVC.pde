void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrain.arff");//load a ARFF dataset
  loadTestARFF(dataset="mouseTest.arff");//load a ARFF dataset
  loadModel(model="LinearSVC.model"); //load a pretrained model.
  setModelDrawing(unit=2);          //set the model visualization (for 2D features)
  evaluateTestSet(isRegression = false, showEvalDetails=true);  //5-fold cross validation
  
}
void draw() {
  drawModel(0, 0); //draw the model visualization (for 2D features)
  drawDataPoints(test); //draw the datapoints
  float[] X = {mouseX, mouseY}; 
  String Y = getPrediction(X);
  drawPrediction(X, Y); //draw the prediction
}
