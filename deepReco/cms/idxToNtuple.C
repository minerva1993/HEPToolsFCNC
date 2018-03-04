#define idxToNtuple_cxx
#include "idxToNtuple.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

#include <iostream>
#include <string>

void idxToNtuple::Loop()
{

  if (fChain == 0) return;

  int totevt = fChain->GetMaximum("nevt");
  //cout << totevt << endl;

  double tmpScore[totevt+1];
  double tmpScoreDummy[totevt+1];
  int tmpMatch[totevt+1];
  int tmpMatchDummy[totevt+1];
  double tmpLepPt[totevt+1];

  for (int i = 0; i < totevt+1; i++){
    tmpScore[i] = -10;
    tmpMatch[i] = -1;
    tmpScoreDummy[i] = -10;
    tmpMatchDummy[i] = -1;
    tmpLepPt[i] = -1;
  }
  int tmpJetIdx[totevt+1][4];

  Long64_t nentries = fChain->GetEntries();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;

    if(genMatch%1000 == 111){
      tmpScoreDummy[nevt] = BDTScore;
      tmpMatchDummy[nevt] = genMatch;
    }

    if(tmpScore[nevt] < BDTScore){
      tmpScore[nevt] = BDTScore;
      tmpJetIdx[nevt][0] = jet0Idx;
      tmpJetIdx[nevt][1] = jet1Idx;
      tmpJetIdx[nevt][2] = jet2Idx;
      tmpJetIdx[nevt][3] = jet3Idx;
      tmpMatch[nevt] = genMatch;
      tmpLepPt[nevt] = lepPt;
    }
    else continue;
  }

  int matchCount = 0;
  int dummyCount = 0;
  for (int i = 0; i <= totevt; ++ i){
    //cout << "nevt = " << i << " and score = " << tmpScore[i] << " and jet indicies are " << tmpJetIdx[i][0] << ", " << tmpJetIdx[i][1] << ", " << mtmpJjetIdx[i][2] << ", " << tmpJetIdx[i][3] << " and gen match is " << tmpMatch[i] << endl;
    if(tmpMatch[i]%1000 == 111) matchCount++;
    if(tmpMatchDummy[i]%1000 == 111) dummyCount++;
  }
  cout <<  matchCount << " , " << dummyCount  << endl;

  /////////////////////////////////////////////////
  string fn = fChain->GetCurrentFile()->GetName();
  string fn2 = "assign_" + fn.substr(14);
  const char *filename = fn2.c_str();

  double score = -1;
  int idx0, idx1, idx2, idx3;
  double leptonPt = 0;

  TFile* fout = new TFile(filename, "recreate");
  TTree* assignT = new TTree("tree", "tree");
  assignT->SetDirectory(fout);
  assignT->Branch("score"   , &score   , "score/D");
  assignT->Branch("idx0"    , &idx0    , "idx0/I");
  assignT->Branch("idx1"    , &idx1    , "idx1/I");
  assignT->Branch("idx2"    , &idx2    , "idx2/I");
  assignT->Branch("idx3"    , &idx3    , "idx3/I");
  assignT->Branch("leptonPt", &leptonPt, "leptonPt/D");

  for (int i = 0; i <= totevt; i++){
    //if (tmpScore[i] < -1 ) continue;
    score = tmpScore[i];
    idx0 = tmpJetIdx[i][0]; idx1 = tmpJetIdx[i][1]; idx2 = tmpJetIdx[i][2]; idx3 = tmpJetIdx[i][3];
    leptonPt = tmpLepPt[i];
    assignT->Fill();
  }
  fout->Write();
}
