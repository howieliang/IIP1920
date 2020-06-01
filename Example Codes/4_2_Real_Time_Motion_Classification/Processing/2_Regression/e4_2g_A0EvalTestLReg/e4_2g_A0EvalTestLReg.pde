//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

void setup() {
  size(500, 500, P2D);
  loadTrainARFF(dataset="A0GestTrainNum.arff"); //load a ARFF dataset
  loadTestARFF(dataset="A0GestTestNum.arff");//load a ARFF dataset
  loadModel(model="LinearReg.model"); //load a pretrained model.
  setModelDrawing(unit=2);         //set the model visualization (for 2D features)
  evaluateTestSet(isRegression = true, showEvalDetails=true);  //5-fold cross validation
}

void draw() {
  drawModel(0, 0); //draw the model visualization (for 2D features)
  drawDataPoints(test); //draw the datapoints
}
