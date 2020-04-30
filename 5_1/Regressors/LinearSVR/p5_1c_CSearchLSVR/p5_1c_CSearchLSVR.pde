//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
double[] EpsArray = {1, 1/2., 1/4., 1/8., 1/16., 1/32., 1/64., 1/128., 1/256.};
boolean showModelOnly = false;

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrainNum.arff"); //load a ARFF dataset
  EpsSearchLSVR(EpsArray);
}

void draw() {
  drawEpsSearchModels(0, 0, width, height);
  if (!showModelOnly) drawEpsSearchResults(0, 0, width, height);
}

void keyPressed() {
  if (key == ENTER || key == ENTER) {
    showModelOnly = (showModelOnly? false : true);
  }
}
