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

  string fn = fChain->GetCurrentFile()->GetName();
  //string fn2 = "assign_" + fn.substr(73);
  //string pt = "root://cms-xrdr.sdfarm.kr:1094//xrd/store/user/minerva1993/reco/assign04/";
  //string fn2 = "assign_" + fn.substr(56);
  //string pt = "/home/minerva1993/recoFCNC/classifier/cms/assign04/";
  string fn2 = "assign_" + fn.substr(75);
  string pt = "/home/minerva1993/fcnc/analysis_2017/reco/classifier/2016/assignST03/";
  string pt2 = pt + fn2;
  const char *path = pt2.c_str();
  TFile* fout = TFile::Open(path, "recreate");
  TTree* assignT = new TTree("tree", "tree");
  assignT->SetDirectory(fout);
  if ( totevt <= 0 ){
    assignT->Write();
    fout->Write();
    //fout->Close();
  }

  double tmpScore[totevt+1];
  double tmpScoreDummy[totevt+1];
  int tmpMatch[totevt+1];
  int tmpMatchDummy[totevt+1];
  double tmpLepPt[totevt+1];
  double tmpMET[totevt+1];
  double tmpWHm[totevt+1];
  double tmpLepTm[totevt+1];
  double tmpHadTm[totevt+1];

  for (int i = 0; i < totevt+1; i++){
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
  int tmpJetIdx[totevt+1][4];

  Long64_t nentries = fChain->GetEntries();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;

    if(genMatch == 1011){
      tmpScoreDummy[nevt] = BDTScore;
      tmpMatchDummy[nevt] = genMatch;
    }

    if(tmpScore[nevt] < BDTScore){
      tmpScore[nevt]     = BDTScore;
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
  for (int i = 0; i <= totevt; ++ i){
    //cout << "nevt = " << i << " and score = " << tmpScore[i] << " and jet indicies are " << tmpJetIdx[i][0] << ", " << tmpJetIdx[i][1] << ", " << mtmpJjetIdx[i][2] << ", " << tmpJetIdx[i][3] << " and gen match is " << tmpMatch[i] << endl;
    if(tmpMatch[i] == 1011) matchCount++;
    if(tmpMatchDummy[i] == 1011) dummyCount++;
  }
  cout <<  matchCount << " , " << dummyCount  << " , " << totevt  << endl;

  /////////////////////////////////////////////////
  double score = -1;
  int idx0, idx1, idx2, idx3;
  double leptonPt, missingET, whMass, lepTopM, hadTopM;

  assignT->Branch("score"    , &score    , "score/D");
  assignT->Branch("idx0"     , &idx0     , "idx0/I");
  assignT->Branch("idx1"     , &idx1     , "idx1/I");
  assignT->Branch("idx2"     , &idx2     , "idx2/I");
  assignT->Branch("idx3"     , &idx3     , "idx3/I");
  assignT->Branch("leptonPt" , &leptonPt , "leptonPt/D");
  assignT->Branch("missingET", &missingET, "missingET/D");
  assignT->Branch("whMass"   , &whMass   , "whMass/D");
  assignT->Branch("lepTopM"  , &lepTopM  , "lepTopM/D");
  assignT->Branch("hadTopM"  , &hadTopM  , "hadTopM/D");

  for (int i = 0; i <= totevt; i++){
    //if (tmpScore[i] < -1 ) continue;
    score = tmpScore[i];
    idx0 = tmpJetIdx[i][0]; idx1 = tmpJetIdx[i][1]; idx2 = tmpJetIdx[i][2]; idx3 = tmpJetIdx[i][3];
    leptonPt = tmpLepPt[i]; missingET = tmpMET[i]; whMass = tmpWHm[i]; lepTopM = tmpLepTm[i]; hadTopM = tmpHadTm[i];
    
    assignT->Fill();
  }
  fout->Write();
}
