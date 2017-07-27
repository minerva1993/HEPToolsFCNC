#include "TFile.h"
#include "TString.h"

#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/DataLoader.h"
#include "TMVA/TMVAGui.h"

void anamva_test(){

  TMVA::Tools::Instance();

  TFile* outputFile = TFile::Open("output_test.root", "RECREATE"); 

  TMVA::Factory *factory = new TMVA::Factory("TMVAClassification",outputFile,"V:!Silent:Color:Transformations=I:DrawProgressBar:AnalysisType=Classification"); 

  Double_t signalWeight = 1.0;
  Double_t backgroundWeight= 0.25;

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

   
  TFile* background1 = new TFile("tmva_tbarchannel.root");
  TTree* backgroundTree1 = (TTree*)(background->Get("tmva_tbarchannel"));
  TFile* background2 = new TFile("tmva_tbarWchannel.root");
  TTree* backgroundTree2 = (TTree*)(background->Get("tmva_tbarWchannel"));
  TFile* background3 = new TFile("tmva_tchannel.root");
  TTree* backgroundTree3 = (TTree*)(background->Get("tmva_tchannel"));
  TFile* background4 = new TFile("tmva_tWchannel.root");
  TTree* backgroundTree4 = (TTree*)(background->Get("tmva_tWchannel"));

  TFile* background5 = new TFile("tmva_ttbb.root");
  TTree* backgroundTree5 = (TTree*)(background->Get("tmva_ttbb"));
  TFile* background6 = new TFile("tmva_ttbj.root");
  TTree* backgroundTree6 = (TTree*)(background->Get("tmva_ttbj"));
  TFile* background7 = new TFile("tmva_ttcc.root");
  TTree* backgroundTree7 = (TTree*)(background->Get("tmva_ttcc"));
  TFile* background8 = new TFile("tmva_ttLF.root");
  TTree* backgroundTree8 = (TTree*)(background->Get("tmva_ttLF"));
  TFile* background9 = new TFile("tmva_tt.root");
  TTree* backgroundTree9 = (TTree*)(background->Get("tmva_tt"));

  TFile* background10 = new TFile("tmva_wjets.root");
  TTree* backgroundTree10 = (TTree*)(background->Get("tmva_wjets"));
  TFile* background11 = new TFile("tmva_ww.root");
  TTree* backgroundTree11 = (TTree*)(background->Get("tmva_ww"));
  TFile* background12 = new TFile("tmva_wz.root");
  TTree* backgroundTree12 = (TTree*)(background->Get("tmva_wz"));
  TFile* background13 = new TFile("tmva_zz.root");
  TTree* backgroundTree13 = (TTree*)(background->Get("tmva_zz"));
  TFile* background14 = new TFile("tmva_zjets10to50.root");
  TTree* backgroundTree14 = (TTree*)(background->Get("tmva_zjets10to50"));
  TFile* background15 = new TFile("tmva_zjets.root");
  TTree* backgroundTree15 = (TTree*)(background->Get("tmva_zjets"));

  loader->AddSignalTree(sigTreeA,0.156137331574);
  loader->AddSignalTree(sigTreeB,0.156137331574);
  loader->AddSignalTree(sigTreeC,0.113541253338);
  loader->AddSignalTree(sigTreeD,0.113541253338);

  loader->AddBackgroundTree(backgroundTree1,0.024575262909);
  loader->AddBackgroundTree(backgroundTree2,0.191680862856);
  loader->AddBackgroundTree(backgroundTree3,0.023844899675);
  loader->AddBackgroundTree(backgroundTree4,0.189008407839);
  loader->AddBackgroundTree(backgroundTree5,0.0910581123792);
  loader->AddBackgroundTree(backgroundTree6,0.0910581123792);
  loader->AddBackgroundTree(backgroundTree7,0.0910581123792);
  loader->AddBackgroundTree(backgroundTree8,0.0910581123792);
  loader->AddBackgroundTree(backgroundTree9,0.0888153017294);
  loader->AddBackgroundTree(backgroundTree10,134.276449237);
  loader->AddBackgroundTree(backgroundTree11,4.70407414855);
  loader->AddBackgroundTree(backgroundTree12,1.69041171);
  loader->AddBackgroundTree(backgroundTree13,0.598577911125);
  loader->AddBackgroundTree(backgroundTree14,22.7143748772);
  loader->AddBackgroundTree(backgroundTree15,2.65549294802);

// && Transvers_Mass < 200 && Missing_ET < 250 && H_Mass < 300 && BJet_Pt_H < 300 && CJet_Pt < 300 && Jet1_Pt < 300 && Jet2_Pt < 300 && Jet3_Pt < 300 && Kin_Lep_WMass > 40 && Kin_Lep_WMass < 12 && Kin_Lep_TopMass > 120 && Kin_Lep_TopMass < 220 && Kin_Had_WMass > 40 && Kin_Had_WMass < 120 && Kin_Had_TopMass > 120 && Kin_Had_TopMass < 220 && fcnhKin_Lep_WMass > 0 && fcnhKin_Lep_WMass < 130 && fcnhKin_Lep_TopMass > 100 && fcnhKin_Lep_TopMass < 250 && fcnhKin_HMass > 120 && fcnhKin_HMass < 130 && fcnhKin_Huc_TopMass > 100 && fcnhKin_Huc_TopMass < 250

  TCut mycuts="(NJets > 0 && NBJets_M > 0 && NCJets_M > 0 && Jet1_CSV > 0 && Jet2_CSV > 0 && Jet3_CSV > 0 && Jet4_CSV > 0)";

  TCut mycutb="(NJets > 0 && NBJets_M > 0 && NCJets_M > 0 && Jet1_CSV > 0 && Jet2_CSV > 0 && Jet3_CSV > 0 && Jet4_CSV > 0)";

  loader->AddVariable("NJets",'I');
  loader->AddVariable("NBJets_M",'I');
  loader->AddVariable("NCJets_M",'I');
  loader->AddVariable("Transvers_Mass",'F');
  loader->AddVariable("Lepton_DPhi",'F');
  loader->AddVariable("Missing_ET",'F');
  loader->AddVariable("BJet_M_delta_R",'F');
  loader->AddVariable("BJet_M_delta_Eta",'F');
  loader->AddVariable("BJet_M_delta_Phi",'F');
  loader->AddVariable("H_Mass",'F');
  loader->AddVariable("BJet_Pt_H",'F');
  loader->AddVariable("CJet_Pt",'F');
  loader->AddVariable("Jet1_Pt",'F');
  loader->AddVariable("Jet2_Pt",'F');
  loader->AddVariable("Jet3_Pt",'F');
  loader->AddVariable("Jet4_Pt",'F');
  loader->AddVariable("Jet1_CSV",'F');
  loader->AddVariable("Jet2_CSV",'F');
  loader->AddVariable("Jet3_CSV",'F');
  loader->AddVariable("Jet4_CSV",'F');
  //kinfit
  loader->AddVariable("Kin_Lep_WMass",'F');
  loader->AddVariable("Kin_Lep_TopMass",'F');
  loader->AddVariable("Kin_Had_WMass",'F');
  loader->AddVariable("Kin_Had_TopMass",'F');
  loader->AddVariable("fcnhKin_Lep_WMass",'F');
  loader->AddVariable("fcnhKin_Lep_TopMass",'F');
  loader->AddVariable("fcnhKin_HMass",'F');
  loader->AddVariable("fcnhKin_Huc_TopMass",'F');


  TString dataString = "nTrain_Signal=30000:"
                       "nTrain_Background=30000:"
                       "nTest_Signal=500:"
                       "nTest_Background=1000:"
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
                              "TestRepetitions=5,"
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
