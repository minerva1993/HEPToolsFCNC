#define MyAnalysis_cxx

#include "MyAnalysis.h"
#include <TH2.h>
#include <TStyle.h>

void MyAnalysis::Begin(TTree * /*tree*/)
{
  TString option = GetOption();
  all_lep = all_mu = all_el = 0;
  hem_lep = hem_mu = hem_el = 0;
  all_mu_sig = all_mu_fake = all_el_sig = all_el_fake = 0;
  hem_mu_sig = hem_mu_fake = hem_el_sig = hem_el_fake = 0;
}

void MyAnalysis::SlaveBegin(TTree * /*tree*/)
{
  TString option = GetOption();
} 

Bool_t MyAnalysis::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();

  current_file_name = MyAnalysis::fReader.GetTree()->GetCurrentFile()->GetName();

  int mode = 999; 
  mode = *channel;
  if( mode > 2 ) return kTRUE;

  bool makeIso = true;
  bool isIso = *lepton_isIso; 
  if( makeIso && !isIso ) return kTRUE;
  if( !makeIso && isIso ) return kTRUE;
  if( *genlepton_pt < 10 ) return kTRUE;//We don't care tau hadnoric

  bool isMatched = false;
  TLorentzVector lepton, genlepton;
  lepton.SetPtEtaPhiE(*lepton_pt, *lepton_eta, *lepton_phi, *lepton_e);
  genlepton.SetPtEtaPhiE(*genlepton_pt, *genlepton_eta, *genlepton_phi, *genlepton_e);
  if( lepton.DeltaR(genlepton) < 0.4 ) isMatched = true;

  bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.4);
  bool passelectron = (mode == 1) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.4);
  if( !passmuon && !passelectron ) return kTRUE; //Only for MC!

  int njets = 0;
  int nbjets_m = 0;
  for (unsigned int iJet = 0; iJet < jet_pt.GetSize() ; ++iJet) {
    TLorentzVector jet;
    jet.SetPtEtaPhiE(jet_pt[iJet], jet_eta[iJet], jet_phi[iJet], jet_e[iJet]);
    jet = jet * jet_JER_Nom[iJet];
    if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
      njets++;
      if( jet_deepCSV[iJet] > 0.4184 ) nbjets_m++;
    }
  }

  bool isHEM = ( lepton.Phi() <= -0.87 and lepton.Phi() >= -1.57 ) and ( lepton.Eta() <= -1.3 and lepton.Eta() >= -3 );

  //if( ( njets < 3 ) or ( nbjets_m < 2 ) ) return kTRUE;

  /*
  /////Fill histograms
  bool eventSelection[Ncuts];
  for(int bcut=0; bcut < 7; bcut++) eventSelection[bcut] = false;

  eventSelection[0]  = true;
  eventSelection[1]  = ( njets >= 3 ) && ( nbjets_m >= 2 );
  eventSelection[2]  = ( njets == 3 ) && ( nbjets_m == 2 );
  eventSelection[3]  = ( njets == 3 ) && ( nbjets_m == 3 );
  eventSelection[4]  = ( njets >= 4 ) && ( nbjets_m == 2 ); 
  eventSelection[5]  = ( njets >= 4 ) && ( nbjets_m == 3 );
  eventSelection[6]  = ( njets >= 4 ) && ( nbjets_m == 4 );

  for( int cut = 0; cut < 7; cut++){
    if( eventSelection[cut] ){


    }//selection loopa
  }
  */

  all_lep++;
  if( isHEM ) hem_lep++;
  if( passmuon ){
    all_mu++;
    if( isMatched ) all_mu_sig++;
    else all_mu_fake++;

    if( isHEM ){
      hem_mu++;
      if( isMatched ) hem_mu_sig++;
      else hem_mu_fake++;
    }
  }
  if( passelectron ){
    all_el++;
    if( isMatched ) all_el_sig++;
    else all_el_fake++;

    if( isHEM ){
      hem_el++;
      if( isMatched ) hem_el_sig++;
      else hem_el_fake++;
    }
  }

  return kTRUE;
}

void MyAnalysis::SlaveTerminate()
{
  TString option = GetOption();
}
   

void MyAnalysis::Terminate()
{
  TString option = GetOption();

  cout << current_file_name << endl;
  cout << "all_lep = "      << all_lep     << endl;
  cout << "all_mu = "       << all_mu      << endl;
  cout << "all_mu_sig = "   << all_mu_sig  << endl;
  cout << "all_mu_fake = "  << all_mu_fake << endl;
  cout << "all_el = "       << all_el      << endl;
  cout << "all_el_sig = "   << all_el_sig  << endl;
  cout << "all_el_fake = "  << all_el_fake << endl;
  cout << "hem_lep = "      << hem_lep     << endl;
  cout << "hem_mu = "       << hem_mu      << endl;
  cout << "hem_mu_sig = "   << hem_mu_sig  << endl;
  cout << "hem_mu_fake = "  << hem_mu_fake << endl;
  cout << "hem_el = "       << hem_el      << endl;
  cout << "hem_el_sig = "   << hem_el_sig  << endl;
  cout << "hem_el_fake = "  << hem_el_fake << endl;

  cout << "all fake rate = " << (all_mu_fake+all_el_fake+0.000001) * 100 / all_lep
       << "+-" << sqrt( all_mu_fake+all_el_fake ) * 100 / all_lep << endl;
  cout << "all mu fake rate = " << (all_mu_fake+0.000001) * 100 / all_mu
       << "+-" << sqrt( all_mu_fake ) * 100 / all_mu << endl;
  cout << "all el fake rate = " << (all_el_fake+0.000001) * 100 / all_el
       << "+-" << sqrt( all_el_fake ) * 100 / all_el << endl;
  cout << "hem fake rate = " << (hem_mu_fake+hem_el_fake+0.000001) * 100 / hem_lep
       << "+-" << sqrt( hem_mu_fake+hem_el_fake ) * 100 / hem_lep << endl;
  cout << "hem mu fake rate = " << (hem_mu_fake+0.000001) * 100 / hem_mu
       << "+-" << sqrt( hem_mu_fake ) * 100 / hem_mu << endl;
  cout << "hem el fake rate = " << (hem_el_fake+0.000001) * 100 / hem_el
       << "+-" << sqrt( hem_el_fake ) * 100 / hem_el << endl;
}

