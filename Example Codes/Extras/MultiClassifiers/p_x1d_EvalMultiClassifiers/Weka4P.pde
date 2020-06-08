import weka.core.converters.ConverterUtils.DataSource; //https://weka.sourceforge.io/doc.stable-3-8/weka/core/converters/ConverterUtils.DataSource.html
import weka.core.Attribute; //https://weka.sourceforge.io/doc.dev/weka/core/Attribute.html
import weka.core.Instances; //https://weka.sourceforge.io/doc.dev/weka/core/Instances.html
import weka.core.DenseInstance; //https://weka.sourceforge.io/doc.dev/weka/core/DenseInstance.html
import weka.classifiers.Classifier; //https://weka.sourceforge.io/doc.stable-3-8/weka/classifiers/Classifier.html
import weka.classifiers.Evaluation; //https://weka.sourceforge.io/doc.dev/weka/classifiers/Evaluation.html
import weka.classifiers.AbstractClassifier; //https://weka.sourceforge.io/doc.dev/weka/classifiers/AbstractClassifier.html
import weka.filters.Filter; //https://weka.sourceforge.io/doc.dev/weka/filters/Filter.html
import weka.filters.unsupervised.attribute.NumericToNominal; //https://weka.sourceforge.io/doc.dev/weka/filters/unsupervised/attribute/NumericToNominal.html
import java.util.Random; //https://docs.oracle.com/javase/8/docs/api/java/util/Random.html
import weka.classifiers.functions.SMO; //https://weka.sourceforge.io/doc.stable/weka/classifiers/functions/SMO.htm
import weka.classifiers.functions.supportVector.RBFKernel; //https://weka.sourceforge.io/doc.dev/weka/classifiers/functions/supportVector/RBFKernel.html
import weka.classifiers.functions.supportVector.PolyKernel; //https://weka.sourceforge.io/doc.dev/weka/classifiers/functions/supportVector/PolyKernel.html
import weka.classifiers.functions.LinearRegression; //https://weka.sourceforge.io/doc.dev/weka/classifiers/functions/LinearRegression.html
import weka.classifiers.evaluation.RegressionAnalysis; //https://weka.sourceforge.io/doc.dev/weka/classifiers/evaluation/RegressionAnalysis.html
import weka.classifiers.functions.supportVector.RegSMOImproved; //https://weka.sourceforge.io/doc.dev/weka/classifiers/functions/supportVector/RegSMOImproved.html
import weka.classifiers.meta.AttributeSelectedClassifier; //https://weka.sourceforge.io/doc.dev/weka/classifiers/meta/AttributeSelectedClassifier.html
import weka.attributeSelection.InfoGainAttributeEval; //https://weka.sourceforge.io/doc.dev/weka/attributeSelection/InfoGainAttributeEval.html
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.Ranker; //https://weka.sourceforge.io/doc.dev/weka/attributeSelection/Ranker.html
import weka.classifiers.functions.MultilayerPerceptron; //https://weka.sourceforge.io/doc.dev/weka/classifiers/functions/MultilayerPerceptron.html

DataSource source;
Instances train;
Instances test;
ArrayList<Attribute> attributesTrain;
ArrayList<Attribute> attributesTest;
Evaluation eval;

PGraphics pg;
Classifier cls;

AttributeSelectedClassifier attrSelCls;
CorrelationAttributeEval corrEval;
Ranker ranker;
InfoGainAttributeEval IGEval;


int nClassesTrain;
int nAttributesTrain;
int nInstancesTrain;
double accuracyTrain, weightedPrecisionTrain, weightedRecallTrain; 
double weightedFprTrain, weightedFnrTrain, weightedFTrain; 
double weightedMccTrain, weightedRocTrain, weightedPrcTrain;
double[] precisionTrain, recallTrain, tprTrain, fprTrain, fnrTrain, fTrain, mccTrain, rocTrain, prcTrain;
double[][] confusionMatrixTrain;
double maeTrain = 0;
double rmseTrain = 0;
double raeTrain = 0;
double rrseTrain = 0;

int nClassesTest;
int nAttributesTest;
int nInstancesTest;
double accuracyTest, weightedPrecisionTest, weightedRecallTest; 
double weightedFprTest, weightedFnrTest, weightedFTest; 
double weightedMccTest, weightedRocTest, weightedPrcTest;
double[] precisionTest, recallTest, tprTest, fprTest, fnrTest, fTest, mccTest, rocTest, prcTest;
double[][] confusionMatrixTest;
double maeTest = 0;
double rmseTest = 0;
double raeTest = 0;
double rrseTest = 0;

double slope = 0;
double intercept = 0;
double corrCoef = 0;

double ssr = 0;
double rSquared = 0;

String dataset = "";
String model = "";
double C = 64;
double gamma = 64;
double epsilon = 64;
double corrThld = 0.5;
double learningRate = 0.3;
int K = 1;
String hiddenLayers = "10,20,10";
int trainingTime = 500;
int fold = 5;
int unit = 2;
long timeStamp = millis();
long timeLapse = 0;

PImage[][] modelImageGrid;// = new double[numOfC][numOfGamma];
double[][] accuracyGrid;// = new double[numOfC][numOfGamma];
//double[][] timeLapseGrid;// = new double[numOfC][numOfGamma];
boolean showEvalDetails = true;
boolean isRegression = false;
boolean drawModels = true;

double[] CList;
double[] gammaList;
double[] EpsList;
int[] KList;

color colors[] = {
  color(155, 89, 182), color(63, 195, 128), color(214, 69, 65), 
  color(82, 179, 217), color(244, 208, 63), color(242, 121, 53), 
  color(0, 121, 53), color(128, 128, 0), color(52, 0, 128), 
  color(128, 52, 0), color(52, 128, 0), color(128, 52, 0)
};


Instances loadTrainARFFToInstances(String filename) {
  Instances insts;
  try {
    source = new DataSource(dataPath(filename));
    insts = source.getDataSet();
    insts.setClassIndex(insts.numAttributes()-1);
    nClassesTrain = insts.numClasses();
    nAttributesTrain = insts.numAttributes();
    nInstancesTrain = insts.numInstances();
    //attrs = new ArrayList<Attribute>();
    //for (int i = 0; i < nAttributesTrain; i++) {
    //  attrs.add(insts.attribute(i));
    //}
    println("===");
    println("Data set: " + filename);
    println("Attributes: " + insts.numAttributes());
    println("Instances: " + insts.numInstances());
    println("Classes: " + insts.numClasses());
    println("Name: " + insts.classAttribute().toString());
    return insts;
  }
  catch(java.lang.Exception e) {
    println(e);
    return null;
  }
}

ArrayList<Attribute> loadAttributesFromInstances(Instances _insts) {
  ArrayList<Attribute> attrs;
  try {
    attrs = new ArrayList<Attribute>();
    for (int i = 0; i < nAttributesTrain; i++) {
      attrs.add(_insts.attribute(i));
    }
    return attrs;
  }
  catch(java.lang.Exception e) {
    println(e);
    return null;
  }
}




void loadTrainARFF(String filename) {
  try {
    source = new DataSource(dataPath(filename));
    train = source.getDataSet();
    train.setClassIndex(train.numAttributes()-1);
    nClassesTrain = train.numClasses();
    nAttributesTrain = train.numAttributes();
    nInstancesTrain = train.numInstances();

    attributesTrain = new ArrayList<Attribute>();
    for (int i = 0; i < nAttributesTrain; i++) {
      attributesTrain.add(train.attribute(i));
    }
    println("===");
    println("Train set: " + filename);
    println("Attributes: " + nAttributesTrain);
    println("Instances: " + nInstancesTrain);
    println("Classes: " + nClassesTrain);
    println("Name: " + train.classAttribute().toString());
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void loadTestARFF(String filename) {
  try {
    source = new DataSource(dataPath(filename));
    test = source.getDataSet();
    test.setClassIndex(test.numAttributes()-1);
    nClassesTest = test.numClasses();
    nAttributesTest = test.numAttributes();
    nInstancesTest = test.numInstances();

    attributesTest = new ArrayList<Attribute>();
    for (int i = 0; i < nAttributesTest; i++) {
      attributesTest.add(test.attribute(i));
    }
    println("===");
    println("Test set: " + filename);
    println("Attributes: " + nAttributesTest);
    println("Instances: " + nInstancesTest);
    println("Classes: " + nClassesTest);
    println("Name: " + test.classAttribute().toString());
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void loadCSV(String _filename) {  
  try {
    readCSVNominal(_filename);
    nClassesTrain = train.numClasses();
    nAttributesTrain = train.numAttributes();
    nInstancesTrain = train.numInstances();
  }
  catch (Exception e) {
    e.printStackTrace();
  }
}

void loadCSVNumeric(String _filename) { 
  Table _csvData = loadTable(_filename, "header");
  String[] attrStr = _csvData.getColumnTitles();
  int attrNum = attrStr.length;
  attributesTrain = new ArrayList<Attribute>();
  for (int i = 0; i < attrStr.length; i++) {
    attributesTrain.add(new Attribute(attrStr[i]));
  }
  // Make an empty training set
  train = new Instances("Train Data", attributesTrain, _csvData.getRowCount());
  // The last element is the "class"?
  train.setClassIndex(attrNum-1);

  for (int i = 0; i < _csvData.getRowCount(); i++) {
    // Add training data
    Instance inst = new DenseInstance(attrNum);
    TableRow row = _csvData.getRow(i);
    for (int j = 0; j< attrNum; j++) {
      inst.setValue(attributesTrain.get(j), row.getFloat(attrStr[j]));
    }
    train.add(inst);
  }
}

void saveModel(String _filename) {
  saveClassifier(cls, _filename);
}

void saveClassifier(Classifier _cls, String _filename) {
  try {
    weka.core.SerializationHelper.write(dataPath(_filename), _cls);
  } 
  catch (Exception e) {
    e.printStackTrace();
  }
}



double[] getProbability(float[] _features, ArrayList<Attribute> _attributes) {
  double[] prob = new double[nClassesTrain];
  try {
    Instances test = new Instances("Test Data", _attributes, 0);
    test.setClassIndex(_attributes.size()-1);
    Instance instance = new DenseInstance(_attributes.size());
    for (int i = 0; i<_features.length; i++) {
      instance.setValue(_attributes.get(i), _features[i]);
    }
    instance.setDataset(test);
    prob = cls.distributionForInstance(instance);
    double maxP = 0;
    for (int i = 0; i < prob.length; i++) {
      maxP = (prob[i]> maxP? prob[i] : maxP);
    }
    for (int i = 0; i < prob.length; i++) {
      prob[i] = prob[i] /= maxP;
    }
  }
  catch (Exception ex) {
    ex.printStackTrace();
  }
  return prob;
}

double getPredictionIndex(float[] _features) {
  double _pred = -1;
  try {
    Instances test = new Instances("Test Data", attributesTrain, 0);
    test.setClassIndex(attributesTrain.size()-1);
    Instance instance = new DenseInstance(attributesTrain.size());
    for (int i = 0; i<_features.length; i++) {
      instance.setValue(attributesTrain.get(i), _features[i]);
    }
    instance.setDataset(test);
    _pred = cls.classifyInstance(instance);
  }
  catch (Exception ex) {
    ex.printStackTrace();
  }
  return _pred;
}

double getPredictionIndex(float[] _features, Classifier _cls, ArrayList<Attribute> _attrs) {
  double _pred = -1;
  try {
    Instances test = new Instances("Test Data", _attrs, 0);
    test.setClassIndex(_attrs.size()-1);
    Instance instance = new DenseInstance(_attrs.size());
    for (int i = 0; i<_features.length; i++) {
      instance.setValue(_attrs.get(i), _features[i]);
    }
    instance.setDataset(test);
    _pred = _cls.classifyInstance(instance);
  }
  catch (Exception ex) {
    ex.printStackTrace();
  }
  return _pred;
}

double getPredictionIndex(float[] _features, Classifier _cls) {
  double _pred = -1;
  try {
    Instances test = new Instances("Test Data", attributesTrain, 0);
    test.setClassIndex(attributesTrain.size()-1);
    Instance instance = new DenseInstance(attributesTrain.size());
    for (int i = 0; i<_features.length; i++) {
      instance.setValue(attributesTrain.get(i), _features[i]);
    }
    instance.setDataset(test);
    _pred = _cls.classifyInstance(instance);
  }
  catch (Exception ex) {
    ex.printStackTrace();
  }
  return _pred;
}

String getPrediction(float[] _features, Classifier _cls, ArrayList<Attribute> _attrs, Instances _insts) {
  String label = "";
  try {
    Instances test = new Instances("Test Data", _attrs, 0);
    test.setClassIndex(_attrs.size()-1);
    Instance instance = new DenseInstance(_attrs.size());
    for (int i = 0; i<_features.length; i++) {
      instance.setValue(_attrs.get(i), _features[i]);
    }
    instance.setDataset(test);
    int _pred = (int) _cls.classifyInstance(instance);
    label = _insts.classAttribute().value(_pred);
  }
  catch (Exception ex) {
    ex.printStackTrace();
  }
  return label;
}

String getPrediction(float[] _features, Classifier _cls) {
  String label = "";
  try {
    Instances test = new Instances("Test Data", attributesTrain, 0);
    test.setClassIndex(attributesTrain.size()-1);
    Instance instance = new DenseInstance(attributesTrain.size());
    for (int i = 0; i<_features.length; i++) {
      instance.setValue(attributesTrain.get(i), _features[i]);
    }
    instance.setDataset(test);
    int _pred = (int) _cls.classifyInstance(instance);
    label = train.classAttribute().value(_pred);
  }
  catch (Exception ex) {
    ex.printStackTrace();
  }
  return label;
}

String getPrediction(float[] _features) {
  String label = "";
  try {
    Instances test = new Instances("Test Data", attributesTrain, 0);
    test.setClassIndex(attributesTrain.size()-1);
    Instance instance = new DenseInstance(attributesTrain.size());
    for (int i = 0; i<_features.length; i++) {
      instance.setValue(attributesTrain.get(i), _features[i]);
    }
    instance.setDataset(test);
    int _pred = (int) cls.classifyInstance(instance);
    label = train.classAttribute().value(_pred);
  }
  catch (Exception ex) {
    ex.printStackTrace();
  }
  return label;
}

void loadModel(String fileName) {
  try {
    cls = (Classifier) weka.core.SerializationHelper.read(dataPath(fileName));
  } 
  catch (Exception e) {
    e.printStackTrace();
  }
}

Classifier loadModelToClassifier(String fileName) {
  Classifier _cls;
  try {
    _cls = (Classifier) weka.core.SerializationHelper.read(dataPath(fileName));
    return _cls;
  } 
  catch (Exception e) {
    e.printStackTrace();
    return null;
  }
  
}

void evaluateTestSet(Classifier _cls, Instances _insts, boolean _isRegression, boolean _showEvalDetails) {
  showEvalDetails = _showEvalDetails;
  try {
    eval = new Evaluation(_insts);
    eval.evaluateModel(_cls, _insts);
    if (_isRegression) {
      corrCoef = eval.correlationCoefficient();
      maeTest = eval.meanAbsoluteError();
      rmseTest = eval.rootMeanSquaredError();
      raeTest = eval.relativeAbsoluteError();
      rrseTest = eval.rootRelativeSquaredError();

      if (showEvalDetails) {
        println(_cls);
        println(eval.toSummaryString("\nResults\n======\n", false));
      }
    } else if (!_isRegression) {
      if (showEvalDetails) {
        println(_cls);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toClassDetailsString());
      }
      accuracyTest = eval.pctCorrect();
      maeTest = eval.meanAbsoluteError();
      rmseTest = eval.rootMeanSquaredError();
      raeTest = eval.relativeAbsoluteError();
      rrseTest = eval.rootRelativeSquaredError();
      weightedPrecisionTest = eval.weightedPrecision();
      weightedRecallTest = eval.weightedRecall();
      weightedFprTest = eval.weightedFalsePositiveRate();
      weightedFnrTest = eval.weightedFalseNegativeRate();
      weightedFTest = eval.weightedFMeasure();
      weightedMccTest = eval. weightedMatthewsCorrelation();
      weightedRocTest = eval.weightedAreaUnderROC();
      weightedPrcTest = eval.weightedAreaUnderPRC();

      confusionMatrixTest = eval.confusionMatrix();

      precisionTest = new double[nClassesTest];
      recallTest = new double[nClassesTest];
      tprTest = new double[nClassesTest];
      fprTest = new double[nClassesTest];
      fnrTest = new double[nClassesTest];
      fTest = new double[nClassesTest];
      rocTest = new double[nClassesTest];
      prcTest = new double[nClassesTest];
      mccTest = new double[nClassesTest];
      for (int i = 0; i < nClassesTest; i++) {
        precisionTest[i] = eval.precision(i);
        recallTest[i] = eval.recall(i);
        fnrTest[i] = eval.falseNegativeRate(i);
        fprTest[i] = eval.falsePositiveRate(i);
        tprTest[i] = eval.truePositiveRate(i);
        fTest[i] = eval.fMeasure(i);
        rocTest[i] = eval.areaUnderROC(i);
        prcTest[i] = eval.areaUnderPRC(i);
        mccTest[i] = eval.matthewsCorrelationCoefficient(i);
      }
    }
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}


void evaluateTestSet(boolean _isRegression, boolean _showEvalDetails) {
  showEvalDetails = _showEvalDetails;
  try {
    eval = new Evaluation(test);
    eval.evaluateModel(cls, test);
    if (_isRegression) {
      corrCoef = eval.correlationCoefficient();
      maeTest = eval.meanAbsoluteError();
      rmseTest = eval.rootMeanSquaredError();
      raeTest = eval.relativeAbsoluteError();
      rrseTest = eval.rootRelativeSquaredError();

      LinearRegression lReg= new LinearRegression();
      lReg.buildClassifier(train);
      slope = lReg.coefficients()[0];
      intercept = lReg.coefficients()[lReg.coefficients().length-1];
      ssr = RegressionAnalysis.calculateSSR(train, attributesTrain.get(0), slope, intercept);
      rSquared = RegressionAnalysis.calculateRSquared(train, ssr);
      if (showEvalDetails) {
        println(eval.toSummaryString("\nResults\n======\n", false));
        println(lReg);
        for (int n=0; n<train.numAttributes()-1; n++) {
          println("slope ("+train.attribute(n).name()+"):", lReg.coefficients()[n]);
        }
        println("intercept:", lReg.coefficients()[lReg.coefficients().length-1]);
        println("ssr:", ssr);
        println("r-Square:", rSquared);
      }
    } else if (!_isRegression) {
      if (showEvalDetails) {
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toClassDetailsString());
      }
      accuracyTest = eval.pctCorrect();
      maeTest = eval.meanAbsoluteError();
      rmseTest = eval.rootMeanSquaredError();
      raeTest = eval.relativeAbsoluteError();
      rrseTest = eval.rootRelativeSquaredError();
      weightedPrecisionTest = eval.weightedPrecision();
      weightedRecallTest = eval.weightedRecall();
      weightedFprTest = eval.weightedFalsePositiveRate();
      weightedFnrTest = eval.weightedFalseNegativeRate();
      weightedFTest = eval.weightedFMeasure();
      weightedMccTest = eval. weightedMatthewsCorrelation();
      weightedRocTest = eval.weightedAreaUnderROC();
      weightedPrcTest = eval.weightedAreaUnderPRC();

      confusionMatrixTest = eval.confusionMatrix();

      precisionTest = new double[nClassesTest];
      recallTest = new double[nClassesTest];
      tprTest = new double[nClassesTest];
      fprTest = new double[nClassesTest];
      fnrTest = new double[nClassesTest];
      fTest = new double[nClassesTest];
      rocTest = new double[nClassesTest];
      prcTest = new double[nClassesTest];
      mccTest = new double[nClassesTest];
      for (int i = 0; i < nClassesTest; i++) {
        precisionTest[i] = eval.precision(i);
        recallTest[i] = eval.recall(i);
        fnrTest[i] = eval.falseNegativeRate(i);
        fprTest[i] = eval.falsePositiveRate(i);
        tprTest[i] = eval.truePositiveRate(i);
        fTest[i] = eval.fMeasure(i);
        rocTest[i] = eval.areaUnderROC(i);
        prcTest[i] = eval.areaUnderPRC(i);
        mccTest[i] = eval.matthewsCorrelationCoefficient(i);
      }
    }
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void evaluateTrainSet(int _fold, boolean _isRegression, boolean _showEvalDetails) {
  showEvalDetails = _showEvalDetails;
  try {
    eval = new Evaluation(train);
    eval.crossValidateModel(cls, train, _fold, new Random(1)); //10-fold cross validation
    if (_isRegression) {
      corrCoef = eval.correlationCoefficient();
      maeTrain = eval.meanAbsoluteError();
      rmseTrain = eval.rootMeanSquaredError();
      raeTrain = eval.relativeAbsoluteError();
      rrseTrain = eval.rootRelativeSquaredError();

      LinearRegression lReg= new LinearRegression();
      lReg.buildClassifier(train);
      slope = lReg.coefficients()[0];
      intercept = lReg.coefficients()[lReg.coefficients().length-1];
      ssr = RegressionAnalysis.calculateSSR(train, attributesTrain.get(0), slope, intercept);
      rSquared = RegressionAnalysis.calculateRSquared(train, ssr);
      if (showEvalDetails) {
        println(eval.toSummaryString("\nResults\n======\n", false));
        println(lReg);
        for (int n=0; n<train.numAttributes()-1; n++) {
          println("slope ("+train.attribute(n).name()+"):", lReg.coefficients()[n]);
        }
        println("intercept:", lReg.coefficients()[lReg.coefficients().length-1]);
        println("ssr:", ssr);
        println("r-Square:", rSquared);
      }
    } else if (!_isRegression) {
      if (showEvalDetails) {
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toClassDetailsString());
      }
      accuracyTrain = eval.pctCorrect();
      maeTrain = eval.meanAbsoluteError();
      rmseTrain = eval.rootMeanSquaredError();
      raeTrain = eval.relativeAbsoluteError();
      rrseTrain = eval.rootRelativeSquaredError();
      
      weightedPrecisionTrain = eval.weightedPrecision();
      weightedRecallTrain = eval.weightedRecall();
      weightedFprTrain = eval.weightedFalsePositiveRate();
      weightedFnrTrain = eval.weightedFalseNegativeRate();
      weightedFTrain = eval.weightedFMeasure();
      weightedMccTrain = eval. weightedMatthewsCorrelation();
      weightedRocTrain = eval.weightedAreaUnderROC();
      weightedPrcTrain = eval.weightedAreaUnderPRC();

      confusionMatrixTrain = eval.confusionMatrix();

      precisionTrain = new double[nClassesTrain];
      recallTrain = new double[nClassesTrain];
      tprTrain = new double[nClassesTrain];
      fprTrain = new double[nClassesTrain];
      fnrTrain = new double[nClassesTrain];
      fTrain = new double[nClassesTrain];
      rocTrain = new double[nClassesTrain];
      prcTrain = new double[nClassesTrain];
      mccTrain = new double[nClassesTrain];
      for (int i = 0; i < nClassesTrain; i++) {
        precisionTrain[i] = eval.precision(i);
        recallTrain[i] = eval.recall(i);
        fnrTrain[i] = eval.falseNegativeRate(i);
        fprTrain[i] = eval.falsePositiveRate(i);
        tprTrain[i] = eval.truePositiveRate(i);
        fTrain[i] = eval.fMeasure(i);
        rocTrain[i] = eval.areaUnderROC(i);
        prcTrain[i] = eval.areaUnderPRC(i);
        mccTrain[i] = eval.matthewsCorrelationCoefficient(i);
      }
    }
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void rankAttrLSVC(double C) {
  attrSelCls = new AttributeSelectedClassifier();
  corrEval = new CorrelationAttributeEval();
  ranker = new Ranker();
  try {
    cls = new SMO();
    ((SMO)cls).setC(C);
    attrSelCls.setClassifier(cls);
    attrSelCls.setSearch(ranker);
    attrSelCls.setEvaluator(corrEval);
    attrSelCls.buildClassifier(train);
    double[][] ra = ranker.rankedAttributes();
    println("Rank\tIndex\tAttrName\tValue");
    for (int i = 0; i < ra.length; i++) {
      int index = (int)ra[i][0];
      print(i+1);
      print('\t');
      print(index);
      print('\t');
      print(train.attribute(index).name());
      print('\t');
      print(ra[i][1]);
      println();
    }
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void trainMLP(String _hiddenLayers, int _trainingTime, double _learningRate) {
  try {
    cls = new MultilayerPerceptron();
    ((MultilayerPerceptron)cls).setGUI(false); //visualization network  
    ((MultilayerPerceptron)cls).setHiddenLayers(_hiddenLayers); //network structure
    ((MultilayerPerceptron)cls).setTrainingTime(_trainingTime); //network structure
    ((MultilayerPerceptron)cls).setLearningRate(_learningRate);
    ((MultilayerPerceptron)cls).setSeed(millis());
    timeStamp = millis();
    println("\n=== Training: Multi-layer Perceptron [", _hiddenLayers, "], training time =", _trainingTime);
    cls.buildClassifier(train);
    timeLapse = millis()-timeStamp;
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}


void trainLinearRegression() {
  try {
    cls = new LinearRegression();
    println("\n=== Training: Linear Regression");
    timeStamp = millis();
    cls.buildClassifier(train);
    timeLapse = millis()-timeStamp;
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void trainKNN(int K) {
  PolyKernel poly;
  try {
    cls = new IBk(K); //IBk(int k): kNN classifier.
    println("\n=== Training: KNN ( K =", K, ")");
    timeStamp = millis();
    cls.buildClassifier(train);
    timeLapse = millis()-timeStamp;
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void KSearch(int[] _KList) {
  KList = _KList;
  accuracyGrid = new double[_KList.length][1];
  modelImageGrid = new PImage[_KList.length][1];
  for (int c = 0; c < _KList.length; c++) {
    trainKNN(K=_KList[c]);
    evaluateTrainSet(fold=5, isRegression=false, showEvalDetails=false);        //5-fold cross validation
    setModelDrawing(unit=ceil(sqrt(_KList.length))*2);         //set the model visualization (for 2D features)
    modelImageGrid[c][0] = pg.get();
    accuracyGrid[c][0] = accuracyTrain;
    println(fold+"-fold CV Accuracy:", nf((float)accuracyTrain, 0, 2), "%\n");
  }
}

void drawKSearchModels(float x, float y, float w, float h) {
  pushMatrix();
  pushStyle();
  translate(x, y);
  float N = ceil(sqrt((float)KList.length));
  float W = w/N;
  for (int c = 0; c < KList.length; c++) {
    float X = c%N*W;
    float Y = floor(c/N)*W;
    PImage p = modelImageGrid[c][0];
    p.resize((int)W, (int)W);
    image(p, X, Y);
  }
  popStyle();
  popMatrix();
}

void drawKSearchResults(float x, float y, float w, float h) {
  pushMatrix();
  pushStyle();
  translate(x, y);
  textSize(32);
  float N = ceil(sqrt((float)KList.length));
  float W = w/N;
  for (int c = 0; c < KList.length; c++) {
    float X = c%N*W;
    float Y = floor(c/N)*W;
    String s = "K="+KList[c]+"\n"+nf((float)accuracyGrid[c][0], 0, 2)+"%";
    fill(255);
    text(s, X+10, Y+32);
  }
  popStyle();
  popMatrix();
}

void trainLinearSVR(double epsilon) {
  RegSMOImproved rsi;
  try {
    cls = new SMOreg();
    rsi = new RegSMOImproved();
    rsi.setTolerance(epsilon);
    ((SMOreg)cls).setC(C);
    ((SMOreg)cls).setRegOptimizer(rsi);
    timeStamp = millis();
    println("\n=== Training: Linear SVR ( C =", C, ", epsilon =", epsilon, ")");
    cls.buildClassifier(train);
    timeLapse = millis()-timeStamp;
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void trainRBFSVR(double epsilon, double gamma) {
  RBFKernel rbf;
  RegSMOImproved rsi;
  try {
    cls = new SMOreg();
    rbf = new RBFKernel();
    rbf.setGamma(gamma);
    rsi = new RegSMOImproved();
    rsi.setTolerance(epsilon);
    ((SMOreg)cls).setC(C);
    ((SMOreg)cls).setKernel(rbf);
    ((SMOreg)cls).setRegOptimizer(rsi);
    timeStamp = millis();
    println("\n=== Training: RBF SVR ( C =", C, ", gamma =", gamma, ", epsilon =", epsilon, ")");
    cls.buildClassifier(train);
    timeLapse = millis()-timeStamp;
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void trainLinearSVC(double C) {
  PolyKernel poly;
  try {
    cls = new SMO();
    poly = new PolyKernel();
    poly.setExponent(1);
    ((SMO)cls).setC(C);
    ((SMO)cls).setKernel(poly);
    println("\n=== Training: Linear SVM ( C =", C, ")");
    timeStamp = millis();
    cls.buildClassifier(train);
    timeLapse = millis()-timeStamp;
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void trainPolySVC(int exp, double C) {
  PolyKernel poly;
  try {
    cls = new SMO();
    poly = new PolyKernel();
    poly.setExponent(exp);
    ((SMO)cls).setC(C);
    ((SMO)cls).setKernel(poly);
    timeStamp = millis();
    println("\n=== Training: Polynomial SVM ( C =", C, ", Exponent=", exp, ")");
    cls.buildClassifier(train);
    timeLapse = millis()-timeStamp;
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void trainRBFSVC(double gamma, double C) {
  RBFKernel rbf;
  try {
    cls = new SMO();
    rbf = new RBFKernel();
    rbf.setGamma(gamma);
    ((SMO)cls).setC(C);
    ((SMO)cls).setKernel(rbf);
    timeStamp = millis();
    println("\n=== Training: RBF SVM ( C =", C, ", gamma =", gamma, ")");
    cls.buildClassifier(train);
    timeLapse = millis()-timeStamp;
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void saveSVM(String fileName) {
  try {
    weka.core.SerializationHelper.write(dataPath(fileName), cls);
  }
  catch(java.lang.Exception e) {
    println(e);
  }
}

void printEvalResults(Instances ins, Classifier cls, int n_fold) {
  try {
    eval = new Evaluation(ins);
    eval.crossValidateModel(cls, ins, n_fold, new Random(1)); //10-fold cross validation
    if (showEvalDetails) {
      System.out.println(eval.toSummaryString("\nResults\n======\n", false));
      System.out.println(eval.toMatrixString());
      System.out.println(eval.toClassDetailsString());
    }
    accuracyTrain = eval.pctCorrect();
    weightedPrecisionTrain = eval.weightedPrecision();
    weightedRecallTrain = eval.weightedRecall();
    weightedFprTrain = eval.weightedFalsePositiveRate();
    weightedFnrTrain = eval.weightedFalseNegativeRate();
    weightedFTrain = eval.weightedFMeasure();
    weightedMccTrain = eval. weightedMatthewsCorrelation();
    weightedRocTrain = eval.weightedAreaUnderROC();
    weightedPrcTrain = eval.weightedAreaUnderPRC();

    confusionMatrixTrain = eval.confusionMatrix();

    precisionTrain = new double[nClassesTrain];
    recallTrain = new double[nClassesTrain];
    tprTrain = new double[nClassesTrain];
    fprTrain = new double[nClassesTrain];
    fnrTrain = new double[nClassesTrain];
    fTrain = new double[nClassesTrain];
    rocTrain = new double[nClassesTrain];
    prcTrain = new double[nClassesTrain];
    mccTrain = new double[nClassesTrain];
    for (int i = 0; i < nClassesTrain; i++) {
      precisionTrain[i] = eval.precision(i);
      recallTrain[i] = eval.recall(i);
      fnrTrain[i] = eval.falseNegativeRate(i);
      fprTrain[i] = eval.falsePositiveRate(i);
      tprTrain[i] = eval.truePositiveRate(i);
      fTrain[i] = eval.fMeasure(i);
      rocTrain[i] = eval.areaUnderROC(i);
      prcTrain[i] = eval.areaUnderPRC(i);
      mccTrain[i] = eval.matthewsCorrelationCoefficient(i);
    }
  } 
  catch (Exception e) {
    e.printStackTrace();
  }
}

void setModelDrawing(int pixelSize) {
  if (nAttributesTrain == 3) pg = getModelImage(pg, cls, train, pixelSize, pixelSize); 
  else pg = createGraphics(width, height); // cannot show the KNN model image for now
}

void drawModel(int x, int y) {
  pushMatrix();
  translate(x, y);
  if (pg!=null) image(pg, 0, 0);
  popMatrix();
}

void drawPrediction(float[] X, double Y) {
  pushStyle();
  textSize(12);
  textAlign(LEFT, CENTER);
  noStroke();
  fill(255);
  ellipse(X[0], X[1], 15, 15);

  noStroke();
  if (Y >= 0) {
    color c = lerpColor(colors[floor((float)Y)%colors.length], colors[ceil((float)Y)%colors.length], (float)Y-floor((float)Y));
    fill(c);
  }
  if (Y < 0) {
    color c = lerpColor(colors[(floor((float)Y)%colors.length+colors.length)%colors.length], colors[(ceil((float)Y)%colors.length+colors.length)%colors.length], (float)Y-floor((float)Y));
    fill(c);
  }
  ellipse(X[0], X[1], 10, 10);
  fill(0);
  String label = "X = ["+X[0]+","+X[1]+"]\nY = "+nf((float)Y, 0, 3);
  text(label, X[0]+10, X[1]);
  popStyle();
}

void drawPrediction(float[] X, double Y, color c) {
  pushStyle();
  textSize(12);
  textAlign(LEFT, CENTER);
  noStroke();
  fill(255);
  ellipse(X[0], X[1], 15, 15);

  noStroke();
  fill(c);
  ellipse(X[0], X[1], 10, 10);
  fill(0);
  String label = "X = ["+X[0]+","+X[1]+"]\nY = "+nf((float)Y, 0, 3);
  text(label, X[0]+10, X[1]);
  popStyle();
}

void drawPrediction(float[] X, String Y, color c) {
  pushStyle();
  textSize(12);
  textAlign(LEFT, CENTER);
  noStroke();
  fill(255);
  ellipse(X[0], X[1], 15, 15);
  fill(c);
  ellipse(X[0], X[1], 10, 10);
  String label = "X = ["+X[0]+","+X[1]+"]\nY = "+Y;
  text(label, X[0]+10, X[1]);
  popStyle();
}

void drawPrediction(float[] X, String Y) {
  pushStyle();
  textSize(12);
  textAlign(LEFT, CENTER);
  noStroke();
  fill(255);
  ellipse(X[0], X[1], 15, 15);
  fill(colors[train.classAttribute().indexOfValue(Y)]);
  ellipse(X[0], X[1], 10, 10);
  fill(0);
  String label = "X = ["+X[0]+","+X[1]+"]\nY = "+Y;
  text(label, X[0]+10, X[1]);
  popStyle();
}

void drawDataPoints(Instances db) {
  pushStyle();
  textSize(12);
  textAlign(CENTER, CENTER);

  for (int i = 0; i < db.numInstances(); i ++) {
    Instance ins = db.instance(i);
    float[] X = new float[ins.numAttributes()-1]; 
    for (int j = 0; j < X.length; j++) {
      X[j] = (float)ins.value(j);
    }

    pushStyle();
    stroke(255);
    double Y = ins.value(ins.numAttributes()-1);
    if (Y >= 0) {
      color c = lerpColor(colors[floor((float)Y)%colors.length], colors[ceil((float)Y)%colors.length], (float)Y-floor((float)Y));
      fill(c);
    }
    if (Y < 0) {
      color c = lerpColor(colors[(floor((float)Y)%colors.length+colors.length)%colors.length], colors[(ceil((float)Y)%colors.length+colors.length)%colors.length], (float)Y-floor((float)Y));
      fill(c);
    }

    ellipse(X[0], X[1], 10, 10);
    fill(255);
    if (attributesTrain.get(ins.numAttributes()-1).type() == 0) {
      String strY = ""+ins.value(ins.numAttributes()-1);
      text(strY.charAt(0), X[0], X[1]);
    } else {
      String strY = ins.stringValue(ins.numAttributes()-1);
      text(strY.charAt(0), X[0], X[1]);
    }
    popStyle();
  }
}

void drawDataPoints() {
  pushStyle();
  textSize(12);
  textAlign(CENTER, CENTER);

  for (int i = 0; i < train.numInstances(); i ++) {
    Instance ins = train.instance(i);
    float[] X = new float[ins.numAttributes()-1]; 
    for (int j = 0; j < X.length; j++) {
      X[j] = (float)ins.value(j);
    }

    pushStyle();
    stroke(255);
    double Y = ins.value(ins.numAttributes()-1);
    if (Y >= 0) {
      color c = lerpColor(colors[floor((float)Y)%colors.length], colors[ceil((float)Y)%colors.length], (float)Y-floor((float)Y));
      fill(c);
    }
    if (Y < 0) {
      color c = lerpColor(colors[(floor((float)Y)%colors.length+colors.length)%colors.length], colors[(ceil((float)Y)%colors.length+colors.length)%colors.length], (float)Y-floor((float)Y));
      fill(c);
    }
    ellipse(X[0], X[1], 10, 10);
    fill(255);
    if (attributesTrain.get(ins.numAttributes()-1).type() == 0) {
      String strY = ""+ins.value(ins.numAttributes()-1);
      text(strY.charAt(0), X[0], X[1]);
    } else {
      String strY = ins.stringValue(ins.numAttributes()-1);
      text(strY.charAt(0), X[0], X[1]);
    }
    popStyle();
  }
}


PGraphics getModelImage(PGraphics pg, Classifier cls, Instances training, int w, int h) {
  //drawModelImage
  pg = createGraphics(width, height);
  pg.beginDraw();
  //pg.rectMode(CORNER);
  pg.background(255);
  for (int x = 0; x <= pg.width; x+=w) {
    for (int y = 0; y <= pg.height; y+=h) {
      Instance inst = new DenseInstance(3);     
      inst.setValue(training.attribute(0), (float)x+w/2); 
      inst.setValue(training.attribute(1), (float)y+h/2); 

      // "instance" has to be associated with "Instances"
      Instances testData = new Instances("Test Data", attributesTrain, 0);
      testData.add(inst);
      testData.setClassIndex(2);        

      float classification = -1;
      try {
        // have to get the data out of Instances
        classification = (float) cls.classifyInstance(testData.firstInstance());
      } 
      catch (Exception e) {
        e.printStackTrace();
      }
      pg.noStroke();
      if (classification >= 0) {
        color c = lerpColor(colors[floor(classification)%colors.length], colors[ceil(classification)%colors.length], classification-floor((float)classification));
        pg.fill(c);
      }
      if (classification < 0) {
        color c = lerpColor(colors[(floor(classification)%colors.length+colors.length)%colors.length], colors[(ceil(classification)%colors.length+colors.length)%colors.length], classification-floor((float)classification));
        pg.fill(c);
      }
      pg.rect(x, y, w, h);
    }
  }
  pg.endDraw();
  return pg;
}

void CSearchLSVC(double[] _CList) {
  CList = _CList;
  accuracyGrid = new double[_CList.length][1];
  modelImageGrid = new PImage[_CList.length][1];
  for (int c = 0; c < _CList.length; c++) {
    trainLinearSVC(C=_CList[c]);
    evaluateTrainSet(fold=5, isRegression=false, showEvalDetails=false);        //5-fold cross validation
    setModelDrawing(unit=ceil(sqrt(_CList.length))*2);         //set the model visualization (for 2D features)
    modelImageGrid[c][0] = pg.get();
    accuracyGrid[c][0] = accuracyTrain;
    println(fold+"-fold CV Accuracy:", nf((float)accuracyTrain, 0, 2), "%\n");
  }
}

void EpsSearchLSVR(double[] _EpsList) {
  EpsList = _EpsList;
  accuracyGrid = new double[_EpsList.length][1];
  modelImageGrid = new PImage[_EpsList.length][1];
  for (int c = 0; c < _EpsList.length; c++) {
    trainLinearSVR(epsilon=_EpsList[c]);
    evaluateTrainSet(fold=5, isRegression=true, showEvalDetails=false);        //5-fold cross validation
    setModelDrawing(unit=ceil(sqrt(_EpsList.length))*2);         //set the model visualization (for 2D features)
    modelImageGrid[c][0] = pg.get();
    accuracyGrid[c][0] = rmseTrain;
    println(fold+"-fold CV Accuracy:", nf((float)accuracyTrain, 0, 2), "%\n");
  }
}

void drawCSearchModels(float x, float y, float w, float h) {
  pushMatrix();
  pushStyle();
  translate(x, y);
  float N = ceil(sqrt((float)CList.length));
  float W = w/N;
  for (int c = 0; c < CList.length; c++) {
    float X = c%N*W;
    float Y = floor(c/N)*W;
    PImage p = modelImageGrid[c][0];
    p.resize((int)W, (int)W);
    image(p, X, Y);
  }
  popStyle();
  popMatrix();
}

void drawCSearchResults(float x, float y, float w, float h) {
  pushMatrix();
  pushStyle();
  translate(x, y);
  textSize(32);
  float N = ceil(sqrt((float)CList.length));
  float W = w/N;
  for (int c = 0; c < CList.length; c++) {
    float X = c%N*W;
    float Y = floor(c/N)*W;
    String s = "C="+CList[c]+"\n"+nf((float)accuracyGrid[c][0], 0, 2)+"%";
    fill(255);
    text(s, X+10, Y+32);
  }
  popStyle();
  popMatrix();
}

void drawEpsSearchModels(float x, float y, float w, float h) {
  pushMatrix();
  pushStyle();
  translate(x, y);
  float N = ceil(sqrt((float)EpsList.length));
  float W = w/N;
  for (int c = 0; c < EpsList.length; c++) {
    float X = c%N*W;
    float Y = floor(c/N)*W;
    PImage p = modelImageGrid[c][0];
    p.resize((int)W, (int)W);
    image(p, X, Y);
  }
  popStyle();
  popMatrix();
}

void drawEpsSearchResults(float x, float y, float w, float h) {
  pushMatrix();
  pushStyle();
  translate(x, y);
  textSize(32);
  float N = ceil(sqrt((float)EpsList.length));
  float W = w/N;
  for (int c = 0; c < EpsList.length; c++) {
    float X = c%N*W;
    float Y = floor(c/N)*W;
    String s = "e="+nf((float)EpsList[c],0,4)+"\n"+nf((float)accuracyGrid[c][0], 0, 2);
    fill(255);
    text(s, X+10, Y+32);
  }
  popStyle();
  popMatrix();
}

void gridSearchSVR_RBF(double[] _EpsList, double[] _gammaList) {
  EpsList = _EpsList;
  gammaList = _gammaList;
  accuracyGrid = new double[_EpsList.length][_gammaList.length];
  modelImageGrid = new PImage[_EpsList.length][_gammaList.length];
  for (int g = 0; g < _gammaList.length; g++) {
    for (int c = 0; c < _EpsList.length; c++) {
      trainRBFSVR(epsilon=_EpsList[c], gamma=_gammaList[g]);
      evaluateTrainSet(fold=5, isRegression=true, showEvalDetails=false);        //5-fold cross validation
      setModelDrawing(unit=_gammaList.length*2);         //set the model visualization (for 2D features)
      modelImageGrid[c][g] = pg.get();
      accuracyGrid[c][g] = rmseTrain;
      println(fold+"-fold CV Accuracy:", nf((float)accuracyTrain, 0, 2), "%\n");
    }
  }
}

void gridSearchSVC_RBF(double[] _CList, double[] _gammaList) {
  CList = _CList;
  gammaList = _gammaList;
  accuracyGrid = new double[_CList.length][_gammaList.length];
  modelImageGrid = new PImage[_CList.length][_gammaList.length];
  for (int g = 0; g < _gammaList.length; g++) {
    for (int c = 0; c < _CList.length; c++) {
      trainRBFSVC(gamma=_gammaList[g], C=_CList[c]);
      evaluateTrainSet(fold=5, isRegression=false, showEvalDetails=false);        //5-fold cross validation
      setModelDrawing(unit=_gammaList.length*2);         //set the model visualization (for 2D features)
      modelImageGrid[c][g] = pg.get();
      accuracyGrid[c][g] = accuracyTrain;
      println(fold+"-fold CV Accuracy:", nf((float)accuracyTrain, 0, 2), "%\n");
    }
  }
}

void drawGridSearchModels(float x, float y, float w, float h) {
  pushMatrix();
  pushStyle();
  translate(x, y);
  float W = w/(float)CList.length;
  float H = h/(float)gammaList.length;
  for (int g = 0; g < gammaList.length; g++) {
    for (int c = 0; c < CList.length; c++) {
      float X = c*W;
      float Y = g*H;
      PImage p = modelImageGrid[c][g];
      p.resize((int)W, (int)H);
      image(p, X, Y);
    }
  }
  popStyle();
  popMatrix();
}

void drawGridSearchModels_SVR(float x, float y, float w, float h) {
  pushMatrix();
  pushStyle();
  translate(x, y);
  float W = w/(float)EpsList.length;
  float H = h/(float)gammaList.length;
  for (int g = 0; g < gammaList.length; g++) {
    for (int c = 0; c < EpsList.length; c++) {
      float X = c*W;
      float Y = g*H;
      PImage p = modelImageGrid[c][g];
      p.resize((int)W, (int)H);
      image(p, X, Y);
    }
  }
  popStyle();
  popMatrix();
}

void drawGridSearchResults_SVR(float x, float y, float w, float h) {
  pushMatrix();
  pushStyle();
  textSize(32);
  translate(x, y);
  float W = w/(float)EpsList.length;
  float H = h/(float)gammaList.length;
  for (int g = 0; g < gammaList.length; g++) {
    for (int c = 0; c < EpsList.length; c++) {
      float X = c*W;
      float Y = g*H;
      String s = "e="+nf((float)EpsList[c], 0, 4)+"\nG="+gammaList[g]+"\n"+nf((float)accuracyGrid[c][g], 0, 2);
      fill(255);
      text(s, X+10, Y+32);
    }
  }
  popStyle();
  popMatrix();
}


void drawGridSearchResults(float x, float y, float w, float h) {
  pushMatrix();
  pushStyle();
  textSize(32);
  translate(x, y);
  float W = w/(float)CList.length;
  float H = h/(float)gammaList.length;
  for (int g = 0; g < gammaList.length; g++) {
    for (int c = 0; c < CList.length; c++) {
      float X = c*W;
      float Y = g*H;
      String s = "C="+CList[c]+"\nG="+gammaList[g]+"\n"+nf((float)accuracyGrid[c][g], 0, 2)+"%";
      fill(255);
      text(s, X+10, Y+32);
    }
  }
  popStyle();
  popMatrix();
}

void readCSVNominal(String fileName) throws Exception {
  CSVLoader loader = new CSVLoader();
  Instances data;
  int index = loadTable(fileName, "header").getColumnCount()-1;
  //loader.setNoHeaderRowPresent(true);
  loader.setSource(new File(dataPath(fileName)));
  data = loader.getDataSet();
  data.setClassIndex(index);

  NumericToNominal convert= new NumericToNominal();
  //String[] options= new String[2];
  //options[0]="-R";
  //options[1]=""+(index+1); //range of variables to make numeric
  int[] val = {index};
  try {
    // have to get the data out of Instances
    //convert.setOptions(options);
    convert.setAttributeIndicesArray(val);
    convert.setInputFormat(data);
    train=Filter.useFilter(data, convert);
  } 
  catch (Exception e) {
    e.printStackTrace();
  }

  println("Attributes : " + train.numAttributes());
  println("Instances : " + train.numInstances());
  println("Name : " + train.classAttribute().toString());

  attributesTrain = new ArrayList<Attribute>();
  for (int i = 0; i < train.numAttributes(); i++) {
    attributesTrain.add(new Attribute(train.attribute(i).name()));
  }

  train.setClassIndex(index);
}
