#define FCNCAnalysis_cxx
// The class definition in FCNCAnalysis.h has been generated automatically
// by the ROOT utility TTree::MakeSelector(). This class is derived
// from the ROOT class TSelector. For more information on the TSelector
// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.


// The following methods are defined in this file:
//    Begin():        called every time a loop on the tree starts,
//                    a convenient place to create your histograms.
//    SlaveBegin():   called after Begin(), when on PROOF called only on the
//                    slave servers.
//    Process():      called for each event, in this function you decide what
//                    to read and fill your histograms.
//    SlaveTerminate: called at the end of the loop on the tree, when on PROOF
//                    called only on the slave servers.
//    Terminate():    called at the end of the loop on the tree,
//                    a convenient place to draw/fit your histograms.
//
// To use this file, try the following session on your Tree T:
//
// root> T->Process("FCNCAnalysis.C")
// root> T->Process("FCNCAnalysis.C","some options")
// root> T->Process("FCNCAnalysis.C+")
//


#include "FCNCAnalysis.h"
#include <TH2.h>
#include <TStyle.h>

void FCNCAnalysis::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
}

void FCNCAnalysis::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   for( int i=0; i < 5; i++){

     h_NJet[i] = new TH1D(Form("h_NJet_S%i_%s",i,option.Data()), "Number of jets", 12, 0, 12);
     h_NJet[i]->SetXTitle("Jet Multiplicity");
     h_NJet[i]->Sumw2();
     fOutput->Add(h_NJet[i]);

     h_NBJetCSVv2M[i] = new TH1D(Form("h_NBJetCSVv2M_S%i_%s",i,option.Data()), "Number of b tagged jets (medium)", 5, 0, 5);
     h_NBJetCSVv2M[i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2M)");
     h_NBJetCSVv2M[i]->Sumw2();
     fOutput->Add(h_NBJetCSVv2M[i]);

     h_NBJetCSVv2T[i] = new TH1D(Form("h_NBJetCSVv2T_S%i_%s",i,option.Data()), "Number of b tagged jets (tight)", 5, 0, 5);
     h_NBJetCSVv2T[i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2T)");
     h_NBJetCSVv2T[i]->Sumw2();
     fOutput->Add(h_NBJetCSVv2T[i]);

     h_NCJetM[i] = new TH1D(Form("h_NCJetM_S%i_%s",i,option.Data()), "Number of c tagged jets", 6, 0, 6);
     h_NCJetM[i]->SetXTitle("c-tagged Jet Multiplicity (M)");
     h_NCJetM[i]->Sumw2();
     fOutput->Add(h_NCJetM[i]);

     h_MET[i] = new TH1D(Form("h_MET_S%i_%s",i,option.Data()), "MET", 40,0,400);
     h_MET[i]->SetXTitle("MET (GeV)");
     h_MET[i]->Sumw2();
     fOutput->Add(h_MET[i]);
  
     h_WMass[i] = new TH1D(Form("h_WMass_S%i_%s",i,option.Data()), "WMass", 32 ,0 ,160);
     h_WMass[i]->SetXTitle("Transverse Mass (GeV)");
     h_WMass[i]->Sumw2();
     fOutput->Add(h_WMass[i]);

     h_DPhi[i] = new TH1D(Form("h_DPhi_S%i_%s",i,option.Data()), "DPhi", 64 ,0 ,3.2);
     h_DPhi[i]->SetXTitle("Angle between MET and lepton");
     h_DPhi[i]->Sumw2();
     fOutput->Add(h_DPhi[i]);

     h_LepIso[i] = new TH1D(Form("h_LepIso_S%i_%s",i,option.Data()), "LepIso", 100 ,0 ,1);
     h_LepIso[i]->SetXTitle("Relative Isolation");
     h_LepIso[i]->Sumw2();
     fOutput->Add(h_LepIso[i]);

     //make1DHistogram( h_kinhmass[i], i, "h_kinhmass", "Higgs mass (GeV)", 50, 100, 150); 
     //make1DHistogram( h_kinhdRbb[i], i, "h_kinhdRbb", "DR betweeen two b jets", 100, 0, 3.0); 
     //make1DHistogram( h_kinwmass[i], i, "h_kinwmass", "W boson mass (GeV)", 100, 50, 150); 

   }

}

Bool_t FCNCAnalysis::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // When processing keyed objects with PROOF, the object is already loaded
   // and is available via the fObject pointer.
   //
   // This function should contain the \"body\" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.
   //
   // The processing can be stopped by calling Abort().
   //
   // Use fStatus to set the return value of TTree::Process().
   //
   // The return value is currently not used.

   fReader.SetEntry(entry);

   EventWeight = 1.0;

   //Event selection 
   bool passmuon = *NMuon == 1 ; // && (*NLooseMuon + *NLooseElectron) == 0;
   bool passelectron = *NElectron == 1; // && (*NLooseMuon + *NLooseElectron) == 0;

   if ( passmuon ){
     h_NJet[0]->Fill( *NJet, EventWeight);
     h_NBJetCSVv2M[0]->Fill( *NBJet, EventWeight);
     h_NBJetCSVv2T[0]->Fill( *NBJet, EventWeight);
     h_NCJetM[0]->Fill(*NBJet, EventWeight);
     h_MET[0]->Fill(*MET, EventWeight);
     h_WMass[0]->Fill(MT_MuonMET[0], EventWeight);
     h_DPhi[0]->Fill(Phi_MuonMET[0], EventWeight);
     h_LepIso[0]->Fill(Muon_Iso03[0], EventWeight);
     //h_kinhmass[0]->Fill(*Kin_Hmass, EventWeight);
     //h_kinhdRbb[0]->Fill(*Kin_HdRbb, EventWeight);
     //h_kinwmass[0]->Fill(*Kin_Wmass, EventWeight);
   }

   return kTRUE;
}

void FCNCAnalysis::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

}

void FCNCAnalysis::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   TString option = GetOption();

   //write out the histogram at the end automatically if the name has "h_". 
   TFile * out = TFile::Open(Form("hist_%s.root",option.Data()),"RECREATE");

   TList * l = GetOutputList();
   TIter next(l);
   TObject *object = 0;
   while( ( object = next()) ){
     const char * name = object->GetName();
     std::string str(name);
     if (str.find("h_") !=std::string::npos ){
       object->Write();
     }
   }

   out->Write();
   out->Close();


}

void FCNCAnalysis::make1DHistogram(  TH1D * h, int & step, TString hname, TString title, double nbins, double low, double max){

   h = new TH1D(Form("%s_S%i",hname.Data(), step ), Form("%s",hname.Data()), nbins, low, max);
   h->SetXTitle(Form("%s",title.Data()));
   h->Sumw2();
   fOutput->Add(h);

}
