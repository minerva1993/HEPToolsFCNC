void anamva(){

  TMVA::Tools::Instance();


  TFile* outputFile = TFile::Open("output.root", "RECREATE"); 

  TMVA::Factory *factory = new TMVA::Factory("TMVAClassification",outputFile,"V:!Silent:Color:Transformations=I:DrawProgressBar:AnalysisType=Classification"); 

  Double_t signalWeight = 1.0;
  Double_t backgroundWeight= 1.0;

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

  factory->AddSignalTree(sigTreeA,signalWeight);
  factory->AddSignalTree(sigTreeB,signalWeight);
  factory->AddSignalTree(sigTreeC,signalWeight);
  factory->AddSignalTree(sigTreeD,signalWeight);
  factory->AddBackgroundTree(backgroundTree,backgroundWeight);

  TCut mycuts="(NJets > 0 && NBJets_M > 0 && NCJets_M >0 && BJMDPhi < 10 && BJMDEta < 10)";
  TCut mycutb="(NJets > 0 && NBJets_M > 0 && NCJets_M >0 && BJMDPhi < 10 && BJMDEta < 10)";

  factory->AddVariable("NJets",'I');
  factory->AddVariable("NBJets_M",'I');
  factory->AddVariable("NCJets_M",'I');
  factory->AddVariable("BJMDPhi",'F');
  factory->AddVariable("BJMDEta",'F');  

  factory->PrepareTrainingAndTestTree(mycuts,mycutb,"random"); 

  factory->BookMethod( TMVA::Types::kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=1000:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
  factory->BookMethod( TMVA::Types::kBDT, "BDT", "!H:!V:NTrees=1500:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:UseBaggedBoost:SeparationType=GiniIndex");
  //factory->BookMethod( TMVA::Types::kDNN, "DNN CPU", "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM:Architecture=CPU"); 

  factory->TrainAllMethods();
  
  factory->TestAllMethods();

  factory->EvaluateAllMethods();

  //if (!gROOT->IsBatch()) TMAV::TMVAGui( outputFile ); 

}
