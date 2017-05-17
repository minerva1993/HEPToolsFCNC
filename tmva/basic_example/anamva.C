void anamva(){

  TMVA::Tools::Instance();


  TFile* outputFile = TFile::Open("output.root", "RECREATE"); 

  TMVA::Factory *factory = new TMVA::Factory("TMVAClassification",outputFile,"V:!Silent:Color:Transformations=I:DrawProgressBar:AnalysisType=Classification"); 

  Double_t signalWeight = 1.0;
  Double_t backgroundWeight= 1.0;

  TFile* signal = new TFile("tmva_class_example.root");
  TTree* sigTree = (TTree*)(signal->Get("TreeS"));
   
  TFile* data = new TFile("tmva_class_example.root");
  TTree* dataTree = (TTree*)(data->Get("TreeB"));

  factory->AddSignalTree(sigTree,signalWeight);
  factory->AddBackgroundTree(dataTree,backgroundWeight);

  TCut mycuts="(var1 > -9.0|| var2 > -9.0)";
  TCut mycutb="var3 > -9.0"; 

  factory->AddVariable("var1",'F');
  factory->AddVariable("var2",'F');
  factory->AddVariable("var3",'F');  

  factory->PrepareTrainingAndTestTree(mycuts,mycutb,"random"); 

  factory->BookMethod( TMVA::Types::kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
  factory->BookMethod( TMVA::Types::kBDT, "BDT", "!H:!V:NTrees=50:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:UseBaggedBoost:SeparationType=GiniIndex");

  factory->TrainAllMethods();
  
  factory->TestAllMethods();

  factory->EvaluateAllMethods();

  //if (!gROOT->IsBatch()) TMAV::TMVAGui( outputFile ); 

}
