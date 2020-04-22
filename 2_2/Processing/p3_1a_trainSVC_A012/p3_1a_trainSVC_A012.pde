//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

import processing.serial.*;
Serial port; 

int sensorNum = 3;
int[] rawData = new int[sensorNum];
boolean dataUpdated = false;

void setup() {
  size(500, 500);             //set a canvas

  //Initialize the serial port
  for (int i = 0; i < Serial.list().length; i++) println("[", i, "]:", Serial.list()[i]);
  String portName = Serial.list()[Serial.list().length-1];//MAC: check the printed list
  //String portName = Serial.list()[9];//WINDOWS: check the printed list
  port = new Serial(this, portName, 115200);
  port.bufferUntil('\n'); // arduino ends each data packet with a carriage return 
  port.clear();           // flush the Serial buffer

  loadTrainARFF(dataset="accData.arff"); //load a ARFF dataset
  println(train);
  trainLinearSVC(C=64);               //train a KNN classifier
  //setModelDrawing(unit=2); //set the model visualization (for 2D features)
  evaluateTrainSet(fold=5, showEvalDetails=true);  //5-fold cross validation
  saveSVC(model="LinearSVC.model"); //save the model

  background(52);
}

void draw() {
  //drawModel(0, 0); //draw the model visualization (for 2D features)
  //drawDataPoints(); //draw the datapoints
  if (dataUpdated) {
    background(52);
    fill(255);
    float[] X = {rawData[0], rawData[1], rawData[2]}; 
    String Y = getPrediction(X);
    textSize(32);
    textAlign(CENTER,CENTER);
    String text = "Prediction: "+Y+
                  "\n X="+rawData[0]+
                  "\n Y="+rawData[1]+
                  "\n Z="+rawData[2];
    text(text, width/2, height/2);
    switch(Y){
      case "A": port.write('a'); break;
      case "B": port.write('b'); break;
      default: break;
    }
    dataUpdated = false;
    println(rawData[0], rawData[1], rawData[2], Y);
  }
}

void serialEvent(Serial port) {   
  String inData = port.readStringUntil('\n');  // read the serial string until seeing a carriage return
  if (!dataUpdated) 
  {
    if (inData.charAt(0) == 'A') {
      rawData[0] = int(trim(inData.substring(1)));
    }
    if (inData.charAt(0) == 'B') {
      rawData[1] = int(trim(inData.substring(1)));
    }
    if (inData.charAt(0) == 'C') {
      rawData[2] = int(trim(inData.substring(1)));
      dataUpdated = true;
    }
  }
  return;
}
