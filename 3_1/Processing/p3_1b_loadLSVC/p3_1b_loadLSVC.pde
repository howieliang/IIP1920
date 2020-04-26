//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrain.arff");//load a ARFF dataset
  loadModel(model="LinearSVC.model"); //load a pretrained model.
  setModelDrawing(unit=2);          //set the model visualization (for 2D features)
}
void draw() {
  drawModel(0, 0);  //draw the model visualization (for 2D features)
  float[] X = {mouseX, mouseY}; 
  String Y = getPrediction(X);
  drawPrediction(X, Y); //draw the prediction
}
