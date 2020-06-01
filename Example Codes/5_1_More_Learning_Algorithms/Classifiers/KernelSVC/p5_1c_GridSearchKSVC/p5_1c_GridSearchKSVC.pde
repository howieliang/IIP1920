//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
double[] CArray = {1, 8, 64};
double[] gammaArray = {1, 8, 64};
boolean showModelOnly = false;

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrain.arff"); //load a ARFF dataset
  gridSearchSVC_RBF(CArray, gammaArray);
}

void draw() {
  drawGridSearchModels(0, 0, width, height); //draw the model visualization (for 2D features)
  if (!showModelOnly) drawGridSearchResults(0, 0, width, height); //draw the statistics
}

void keyPressed() {
  if (key == ENTER || key == ENTER) {
    showModelOnly = (showModelOnly? false : true);
  }
}
