void app(){

  TFile * data = new TFile("tmva_example.root");
  
  TTree* dataTree = (TTree*)(data->Get("TreeS")); 

  TFile *target = new TFile("real_data-mva_output.root","RECREATE" );
  TTree *tree = new TTree("tree","treelibrated tree");

  TMVA::Tools::Instance();
  TMVA::Reader *reader = new TMVA::Reader( "V:Color:!Silent" );  

  Float_t var[3];

  reader->AddVariable( "var1", &var[0] );
  reader->AddVariable( "var2", &var[1] );
  reader->AddVariable( "var3", &var[2] );

  reader->BookMVA("BDT method", "weights/TMVAClassification_BDT.weights.xml");  

  Float_t userVar[3];

  dataTree->SetBranchAddress( "var1", &userVar[0] );
  dataTree->SetBranchAddress( "var2", &userVar[1] );
  dataTree->SetBranchAddress( "var3", &userVar[2] );

  Float_t BDT_response;
  tree->Branch("BDT_response",&BDT_response);

  for (Long64_t ievt=0; ievt<dataTree->GetEntries();ievt++) {

    if (ievt%100000 == 0) std::cout << "--- ... Processing event: " << ievt <<std::endl;
    
    dataTree->GetEntry(ievt);

    var[0]=userVar[0];
    var[1]=userVar[1];
    var[2]=userVar[2];

    BDT_response=reader->EvaluateMVA("BDT method");

    tree->Fill();

  }
  
  tree->Write();


}
