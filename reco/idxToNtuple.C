#define idxToNtuple_cxx
#include "idxToNtuple.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

using namespace std;

string replaceAll(std::string &str, const std::string& from, const std::string& to);

void idxToNtuple::Loop(const std::string channel)
{

  if (fChain == 0) return;

  int ncombi = 0;
  //int fij = 0;
  int totevt = fChain->GetMaximum("nevt");
  //cout << totevt << endl;

  string fn = fChain->GetCurrentFile()->GetName();
  replaceAll(fn, "score", "assign");
  const char *path = fn.c_str();
  TFile* fout = TFile::Open(path, "recreate");
  TTree* assignT = new TTree("tree", "tree");
  assignT->SetDirectory(fout);
  if ( totevt <= 0 ){
    assignT->Write();
    fout->Write();
    //fout->Close();
  }

  int signal_label = -1;
  if      ( channel == "STFCNC" ) signal_label = 1011;
  else if ( channel == "TTFCNC" or channel == "TTBKG") signal_label = 1111;
  if ( signal_label == -1 ){
    cout << "Wrong channel!" << endl;
    return;
  }

  float tmpScore[totevt];
  float tmpScoreDummy[totevt];
  int tmpMatch[totevt];
  int tmpMatchDummy[totevt];
  float tmpLepPt[totevt];
  float tmpMET[totevt];
  float tmpWHm[totevt];
  float tmpLepTm[totevt];
  float tmpHadTm[totevt];

  for (int i = 0; i < totevt; i++){
    tmpScore[i]      = -10;
    tmpMatch[i]      = -1;
    tmpScoreDummy[i] = -10;
    tmpMatchDummy[i] = -1;
    tmpLepPt[i]      = -1;
    tmpMET[i]        = -1;
    tmpWHm[i]        = -1;
    tmpLepTm[i]      = -1;
    tmpHadTm[i]      = -1;
  }
  int tmpJetIdx[totevt][4];

  Long64_t nentries = fChain->GetEntries();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;

    //if(not(njets>=4 and nbjets_m==4)) continue;

    if(genMatch == signal_label){
      tmpScoreDummy[nevt] = MLScore;
      tmpMatchDummy[nevt] = genMatch;
      //fij++;
    }

    //cout << nevt << endl;
    ncombi++;

    if(tmpScore[nevt] <= MLScore){
      tmpScore[nevt]     = MLScore;
      tmpJetIdx[nevt][0] = jet0Idx;
      tmpJetIdx[nevt][1] = jet1Idx;
      tmpJetIdx[nevt][2] = jet2Idx;
      tmpJetIdx[nevt][3] = jet3Idx;
      tmpMatch[nevt]     = genMatch;
      tmpMET[nevt]       = missinget;
      tmpLepPt[nevt]     = lepPt;
      tmpWHm[nevt]       = whMass;
      tmpLepTm[nevt]     = leptMass;
      tmpHadTm[nevt]     = hadtMass;
    }
    else continue;
  }

  int matchCount = 0;
  int dummyCount = 0;
  for (int i = 0; i < totevt; ++ i){
    //cout << "nevt = " << i << " and score = " << tmpScore[i] << " and jet indicies are " << tmpJetIdx[i][0] << ", " << tmpJetIdx[i][1] << ", " << mtmpJjetIdx[i][2] << ", " << tmpJetIdx[i][3] << " and gen match is " << tmpMatch[i] << endl;
    if(tmpMatch[i] == signal_label) matchCount++;
    if(tmpMatchDummy[i] == signal_label) dummyCount++;
  }
  cout <<  matchCount << " , " << dummyCount << " , " << totevt  << " , " << ncombi << " , efficiency = " << float(matchCount)/dummyCount << endl;
  //cout << fij << endl;

  /////////////////////////////////////////////////
  float score = -1;
  int idx0, idx1, idx2, idx3, match;
  float leptonPt, missingET, whMass, lepTopM, hadTopM;

  assignT->Branch("score"    , &score    , "score/F");
  assignT->Branch("match"    , &match    , "match/I");
  assignT->Branch("idx0"     , &idx0     , "idx0/I");
  assignT->Branch("idx1"     , &idx1     , "idx1/I");
  assignT->Branch("idx2"     , &idx2     , "idx2/I");
  assignT->Branch("idx3"     , &idx3     , "idx3/I");
  assignT->Branch("leptonPt" , &leptonPt , "leptonPt/F");
  assignT->Branch("missingET", &missingET, "missingET/F");
  assignT->Branch("whMass"   , &whMass   , "whMass/F");
  assignT->Branch("lepTopM"  , &lepTopM  , "lepTopM/F");
  assignT->Branch("hadTopM"  , &hadTopM  , "hadTopM/F");

  for (int i = 0; i < totevt; i++){
    //if (tmpScore[i] < -1 ) continue;
    score = tmpScore[i];
    match = tmpMatch[i];
    idx0 = tmpJetIdx[i][0]; idx1 = tmpJetIdx[i][1]; idx2 = tmpJetIdx[i][2]; idx3 = tmpJetIdx[i][3];
    leptonPt = tmpLepPt[i]; missingET = tmpMET[i]; whMass = tmpWHm[i]; lepTopM = tmpLepTm[i]; hadTopM = tmpHadTm[i];
    
    assignT->Fill();
  }
  fout->Write();
}

std::string replaceAll(std::string &str, const std::string& from, const std::string& to){
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos){
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
    }
    return str;
}
