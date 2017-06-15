#include "TFile.h"
#include "TString.h"

#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/DataLoader.h"
#include "TMVA/TMVAGui.h"

void anamva_rep(){

  TMVA::Tools::Instance();

  TFile* outputFile = TFile::Open("output_test.root", "RECREATE"); 

  TMVA::Factory *factory = new TMVA::Factory("TMVAClassification",outputFile,"V:!Silent:Color:Transformations=I:DrawProgressBar:AnalysisType=Classification"); 

  Double_t signalWeight = 1.0;
  Double_t backgroundWeight= 1.0;

  TMVA::DataLoader *loader=new TMVA::DataLoader("inputcorr_test");
  //TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification",outputFile,"AnalysisType=Classification" );

  TFile* signalA = new TFile("tmva_AntiTop_Hut.root");
  TTree* sigTreeA = (TTree*)(signalA->Get("tmva_AntiTop_Hut"));
  TFile* signalB = new TFile("tmva_AntiTop_Hct.root");
  TTree* sigTreeB = (TTree*)(signalB->Get("tmva_AntiTop_Hct"));
  TFile* signalC = new TFile("tmva_Top_Hut.root");
  TTree* sigTreeC = (TTree*)(signalC->Get("tmva_Top_Hut"));
  TFile* signalD = new TFile("tmva_Top_Hct.root");
  TTree* sigTreeD = (TTree*)(signalD->Get("tmva_Top_Hct"));

   
  TFile* background = new TFile("tmva_ttbb.root");
  TTree* backgroundTree = (TTree*)(background->Get("tmva_ttbb"));

  loader->AddSignalTree(sigTreeA,signalWeight);
  loader->AddSignalTree(sigTreeB,signalWeight);
  loader->AddSignalTree(sigTreeC,signalWeight);
  loader->AddSignalTree(sigTreeD,signalWeight);
  loader->AddBackgroundTree(backgroundTree,backgroundWeight);

  TCut mycuts="(NJets > 0 && NBJets_M > 0 && NCJets_M > 0 && H_Mass < 300 && BJet_Pt < 300 && CJet_Pt < 300 && Jet1_Pt < 300 && Jet2_Pt < 300 && Jet3_Pt < 300 && Jet1_CSV > 0 && Jet2_CSV > 0 && Jet3_CSV > 0 && Jet4_CSV > 0)";
  TCut mycutb="(NJets > 0 && NBJets_M > 0 && NCJets_M > 0 && H_Mass < 300 && BJet_Pt < 300 && CJet_Pt < 300 && Jet1_Pt < 300 && Jet2_Pt < 300 && Jet3_Pt < 300 && Jet1_CSV > 0 && Jet2_CSV > 0 && Jet3_CSV > 0 && Jet4_CSV > 0)";
  loader->AddVariable("NJets",'I');
  loader->AddVariable("NBJets_M",'I');
  loader->AddVariable("NCJets_M",'I');
  loader->AddVariable("BJet_M_delta_R",'F');
  loader->AddVariable("H_Mass",'F');
  loader->AddVariable("BJet_Pt",'F');
  loader->AddVariable("CJet_Pt",'F');
  loader->AddVariable("Jet1_Pt",'F');
  loader->AddVariable("Jet2_Pt",'F');
  loader->AddVariable("Jet3_Pt",'F');
  loader->AddVariable("Jet4_Pt",'F');
  loader->AddVariable("Jet1_CSV",'F');
  loader->AddVariable("Jet2_CSV",'F');
  loader->AddVariable("Jet3_CSV",'F');
  loader->AddVariable("Jet4_CSV",'F');


  TString dataString = "nTrain_Signal=30000:"
                       "nTrain_Background=30000:"
                       "nTest_Signal=10000:"
                       "nTest_Background=10000:"
                       "SplitMode=Random:"
                       "NormMode=NumEvents:"
                       "!V";

  loader->PrepareTrainingAndTestTree(mycuts,mycutb,dataString); 

  factory->BookMethod(loader,TMVA::Types::kLikelihood, "Likelihood","H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );
  factory->BookMethod(loader,TMVA::Types::kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=1000:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
  factory->BookMethod(loader,TMVA::Types::kBDT, "BDT", "!H:!V:NTrees=1000:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:UseBaggedBoost:SeparationType=GiniIndex");

  //DNN part  

    TString trainingString1 = "TrainingStrategy="
                              "LearningRate=0.01,"
                              "Momentum=0.4,"
                              "Repetitions=1,"
                              "ConvergenceSteps=300,"
                              "BatchSize=100,"
                              "DropConfig=0.0+0.2+0.0,"  // Dropout
                              "WeightDecay=0.001,"
                              "Regularization=L2,"
                              "TestRepetitions=10,"
                              "Multithreading=True";

    TString configString = "!H:V";
    configString += ":VarTransform=N";
    configString += ":ErrorStrategy=CROSSENTROPY";
    configString += ":WeightInitialization=XAVIERUNIFORM";
    TString layoutString = "Layout=TANH|30,TANH|20,LINEAR";

    configString += ":" + layoutString + ":" + trainingString1; // + ":Architecture=CPU";

  factory->BookMethod(loader, TMVA::Types::kDNN, "DNN", configString);


  factory->TrainAllMethods();
  
  factory->TestAllMethods();

  factory->EvaluateAllMethods();

  //if (!gROOT->IsBatch()) TMAV::TMVAGui( outputFile ); 

}
