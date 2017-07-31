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
  Double_t backgroundWeight= 1.0;

  TMVA::DataLoader *loader=new TMVA::DataLoader("inputcorr_test");
  //TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification",outputFile,"AnalysisType=Classification" );

  TFile* signalA = new TFile("mkNtuple_v2/tmva_AntiTop_Hut.root");
  TTree* sigTreeA = (TTree*)(signalA->Get("tmva_AntiTop_Hut"));
  TFile* signalB = new TFile("mkNtuple_v2/tmva_AntiTop_Hct.root");
  TTree* sigTreeB = (TTree*)(signalB->Get("tmva_AntiTop_Hct"));
  TFile* signalC = new TFile("mkNtuple_v2/tmva_Top_Hut.root");
  TTree* sigTreeC = (TTree*)(signalC->Get("tmva_Top_Hut"));
  TFile* signalD = new TFile("mkNtuple_v2/tmva_Top_Hct.root");
  TTree* sigTreeD = (TTree*)(signalD->Get("tmva_Top_Hct"));

   
  TFile* background1 = new TFile("mkNtuple_v2/tmva_tbarchannel.root");
  TTree* backgroundTree1 = (TTree*)(background1->Get("tmva_tbarchannel"));
  TFile* background2 = new TFile("mkNtuple_v2/tmva_tbarWchannel.root");
  TTree* backgroundTree2 = (TTree*)(background2->Get("tmva_tbarWchannel"));
  TFile* background3 = new TFile("mkNtuple_v2/tmva_tchannel.root");
  TTree* backgroundTree3 = (TTree*)(background3->Get("tmva_tchannel"));
  TFile* background4 = new TFile("mkNtuple_v2/tmva_tWchannel.root");
  TTree* backgroundTree4 = (TTree*)(background4->Get("tmva_tWchannel"));

  TFile* background5 = new TFile("mkNtuple_v2/tmva_ttbb.root");
  TTree* backgroundTree5 = (TTree*)(background5->Get("tmva_ttbb"));
  TFile* background6 = new TFile("mkNtuple_v2/tmva_ttbj.root");
  TTree* backgroundTree6 = (TTree*)(background6->Get("tmva_ttbj"));
  TFile* background7 = new TFile("mkNtuple_v2/tmva_ttcc.root");
  TTree* backgroundTree7 = (TTree*)(background7->Get("tmva_ttcc"));
  TFile* background8 = new TFile("mkNtuple_v2/tmva_ttLF.root");
  TTree* backgroundTree8 = (TTree*)(background8->Get("tmva_ttLF"));
  TFile* background9 = new TFile("mkNtuple_v2/tmva_tt.root");
  TTree* backgroundTree9 = (TTree*)(background9->Get("tmva_tt"));

  //TFile* background10 = new TFile("mkNtuple_v2/tmva_wjets.root");
 // TTree* backgroundTree10 = (TTree*)(background10->Get("tmva_wjets"));
  TFile* background11 = new TFile("mkNtuple_v2/tmva_ww.root");
  TTree* backgroundTree11 = (TTree*)(background11->Get("tmva_ww"));
  TFile* background12 = new TFile("mkNtuple_v2/tmva_wz.root");
  TTree* backgroundTree12 = (TTree*)(background12->Get("tmva_wz"));
  TFile* background13 = new TFile("mkNtuple_v2/tmva_zz.root");
  TTree* backgroundTree13 = (TTree*)(background13->Get("tmva_zz"));
  TFile* background14 = new TFile("mkNtuple_v2/tmva_zjet10to50.root");
  TTree* backgroundTree14 = (TTree*)(background14->Get("tmva_zjets10to50"));
  TFile* background15 = new TFile("mkNtuple_v2/tmva_zjets.root");
  TTree* backgroundTree15 = (TTree*)(background15->Get("tmva_zjets"));

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
  //loader->AddBackgroundTree(backgroundTree10,134.276449237);
  loader->AddBackgroundTree(backgroundTree11,4.70407414855);
  loader->AddBackgroundTree(backgroundTree12,1.69041171);
  loader->AddBackgroundTree(backgroundTree13,0.598577911125);
  loader->AddBackgroundTree(backgroundTree14,22.7143748772);
  loader->AddBackgroundTree(backgroundTree15,2.65549294802);

// && Transvers_Mass < 200 && Missing_ET < 250 && H_Mass < 300 && BJet_Pt_H < 300 && CJet_Pt < 300 && Jet1_Pt < 300 && Jet2_Pt < 300 && Jet3_Pt < 300 && Kin_Lep_WMass > 40 && Kin_Lep_WMass < 12 && Kin_Lep_TopMass > 120 && Kin_Lep_TopMass < 220 && Kin_Had_WMass > 40 && Kin_Had_WMass < 120 && Kin_Had_TopMass > 120 && Kin_Had_TopMass < 220 && fcnhKin_Lep_WMass > 0 && fcnhKin_Lep_WMass < 130 && fcnhKin_Lep_TopMass > 100 && fcnhKin_Lep_TopMass < 250 && fcnhKin_HMass > 120 && fcnhKin_HMass < 130 && fcnhKin_Huc_TopMass > 100 && fcnhKin_Huc_TopMass < 250

  TCut mycuts="(njets > 0 && nbjets_m  > 0 && ncjets_m > 0 &&  lepDPhi > 0 && missingET > 0 && bjetmDR < 5 && bjetmDEta < 5 && bjetmDEta < 5 && dibjetsMass < 9999 && bjetPt_dibjetsm < 9999 && cjetPt < 9999 && jetPt1 < 9999 && jetPt2 < 9999 && jetPt3 < 9999 && jetPt4 < 9999 &&  jetCSV1 < 1.01 && JetCSV2 < 1.01 && JetCSV3 < 1.01 && JetCSV4 < 1.01 && DRLepWMass > 0 && DRHadWMass > 0 && DRHMass > 0 && DRDR > 0 && DRLepTopMass > 0 && DRHucTopMass > 0)";

  TCut mycutb="(njets > 0 && nbjets_m  > 0 && ncjets_m > 0 && lepDPhi > 0 && missingET > 0 && bjetmDR < 5 && bjetmDEta < 5 && bjetmDEta < 5 && dibjetsMass < 9999 && bjetPt_dibjetsm < 9999 && cjetPt < 9999 && jetPt1 < 9999 && jetPt2 < 9999 && jetPt3 < 9999 && jetPt4 < 9999 &&  jetCSV1 < 1.01 && JetCSV2 < 1.01 && JetCSV3 < 1.01 && JetCSV4 < 1.01 && DRLepWMass > 0 && DRHadWMass > 0 && DRHMass > 0 && DRDR > 0 && DRLepTopMass > 0 && DRHucTopMass > 0)";

  loader->AddVariable("njets",'I');
  loader->AddVariable("nbjets_m",'I');
  loader->AddVariable("ncjets_m",'I');
  //loader->AddVariable("transverseMass",'F');
  loader->AddVariable("lepDPhi",'F');
  loader->AddVariable("missingET",'F');
  loader->AddVariable("bjetmDR",'F');
  loader->AddVariable("bjetmDEta",'F');
  loader->AddVariable("bjetmDPhi",'F');
  loader->AddVariable("dibjetsMass",'F');
  loader->AddVariable("bjetPt_dibjetsm",'F');
  loader->AddVariable("cjetPt",'F');
  loader->AddVariable("jetPt1",'F');
  loader->AddVariable("jetPt2",'F');
  loader->AddVariable("jetPt3",'F');
  loader->AddVariable("jetPt4",'F');
  loader->AddVariable("jetCSV1",'F');
  loader->AddVariable("JetCSV2",'F');
  loader->AddVariable("JetCSV3",'F');
  loader->AddVariable("JetCSV4",'F');
  //kinfit
/*
  loader->AddVariable("KinLepWMass",'F');
  loader->AddVariable("KinLepTopMass",'F');
  loader->AddVariable("KinHadWMass",'F');
  loader->AddVariable("KinHadTopMass",'F');
  loader->AddVariable("FCNHKinLepWMass",'F');
  loader->AddVariable("FCNHKinLepTopMass",'F');
  loader->AddVariable("FCNHKinHMass",'F');
  loader->AddVariable("FCNHKinHucTopMass",'F');
  loader->AddVariable("M3LepWMass",'F');
  loader->AddVariable("M3HadWMass",'F');
  loader->AddVariable("M3HMass",'F');
  loader->AddVariable("M3R",'F');
  loader->AddVariable("M3LepTopMass",'F');
  loader->AddVariable("M3HucTopMass",'F');
*/
  loader->AddVariable("DRLepWMass",'F');
  loader->AddVariable("DRHadWMass",'F');
  loader->AddVariable("DRHMass",'F');
  loader->AddVariable("DRDR",'F');
  loader->AddVariable("DRLepTopMass",'F');
  loader->AddVariable("DRHucTopMass",'F');



  TString dataString = "nTrain_Signal=100000:"
                       "nTrain_Background=300000:"
                       "nTest_Signal=10000:"
                       "nTest_Background=10000:"
                       "SplitMode=Random:"
                       "NormMode=NumEvents:"
                       "!V";

  loader->PrepareTrainingAndTestTree(mycuts,mycutb,dataString); 

  factory->BookMethod(loader,TMVA::Types::kLikelihood, "Likelihood","H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );
  //factory->BookMethod(loader,TMVA::Types::kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=1000:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
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

  //factory->BookMethod(loader, TMVA::Types::kDNN, "DNN", configString);

  factory->TrainAllMethods();
  
  factory->TestAllMethods();

  factory->EvaluateAllMethods();

  //if (!gROOT->IsBatch()) TMAV::TMVAGui( outputFile ); 

}
