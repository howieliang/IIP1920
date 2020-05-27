//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrainNum.arff"); //load a ARFF dataset
  loadModel(model="LSVR.model"); //load a pretrained model.
  setModelDrawing(unit=2);         //set the model visualization (for 2D features)
}

void draw() {
  drawModel(0, 0); //draw the model visualization (for 2D features)
  //drawDataPoints(train); //draw the datapoints
  float[] X = {mouseX, mouseY}; 
  double Y = getPredictionIndex(X);
  drawPrediction(X, Y); //draw the prediction
}
