void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="A012GestTrain.arff"); //load a ARFF dataset
  trainMLP(hiddenLayers = "9", trainingTime = 100);
  //setModelDrawing(unit=2);         //set the model visualization (for 2D features)
  evaluateTrainSet(fold=5, isRegression=false, showEvalDetails=true);  //5-fold cross validationevaluateTrainSet(fold=5, isRegression=false, showEvalDetails=true);  //5-fold cross validation
  saveModel(model="MLP.model"); //save the model
}

void draw() {
  //drawModel(0, 0); //draw the model visualization (for 2D features)
  //drawDataPoints(train); //draw the datapoints
  //float[] X = {mouseX, mouseY}; 
  //String Y = getPrediction(X);
  //double Y = getPredictionIndex(X);
  //drawPrediction(X, Y); //draw the prediction
}
