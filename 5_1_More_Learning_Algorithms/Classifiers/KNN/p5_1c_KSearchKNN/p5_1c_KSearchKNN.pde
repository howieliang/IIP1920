//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
int[] KArray = {1, 3, 5, 7, 9, 11, 13, 15, 17};
boolean showModelOnly = false;

void setup() {
  size(500, 500);             //set a canvas
  loadTrainARFF(dataset="mouseTrain.arff"); //load a ARFF dataset
  //CSearchLinear(CArray); //train a model with every C in CArray
  KSearch(KArray);
}

void draw() {
  drawKSearchModels(0, 0, width, height); //draw the model visualization (for 2D features)
  if (!showModelOnly) drawKSearchResults(0, 0, width, height); //draw the statistics
}

void keyPressed() {
  if (key == ENTER || key == ENTER) {
    showModelOnly = (showModelOnly? false : true);
  }
}
