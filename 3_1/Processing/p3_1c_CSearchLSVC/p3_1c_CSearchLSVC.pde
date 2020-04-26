//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
double[] CArray = {1, 2, 4, 8, 16, 32, 64, 128, 256};
boolean showModelOnly = false;

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrain.arff"); //load a ARFF dataset
  CSearchLinear(CArray); //train a model with every C in CArray
}

void draw() {
  drawCSearchModels(0, 0, width, height); //draw the model visualization (for 2D features)
  if (!showModelOnly) drawCSearchResults(0, 0, width, height); //draw the statistics
}

void keyPressed() {
  if (key == ENTER || key == ENTER) {
    showModelOnly = (showModelOnly? false : true);
  }
}
