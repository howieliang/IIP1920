//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
double[] EpsArray = {0.1, 0.1/4., 0.1/16.};
double[] gammaArray = {1, 8, 64};
boolean showModelOnly = false;

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrainNum.arff"); //load a ARFF dataset
  gridSearchSVR_RBF(EpsArray, gammaArray);
}

void draw() {
  drawGridSearchModels_SVR(0, 0, width, height); //draw the model visualization (for 2D features)
  if (!showModelOnly) drawGridSearchResults_SVR(0, 0, width, height); //draw the statistics
}

void keyPressed() {
  if (key == ENTER || key == ENTER) {
    showModelOnly = (showModelOnly? false : true);
  }
}
