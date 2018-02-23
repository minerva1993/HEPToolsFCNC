#define idxToNtuple_cxx
#include "idxToNtuple.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

#include <iostream>

void idxToNtuple::Loop()
{
//   In a ROOT session, you can do:
//      root> .L idxToNtuple.C
//      root> idxToNtuple t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
  if (fChain == 0) return;

  int totevt = fChain->GetMaximum("nevt");
  cout << totevt << endl;

  double scoreBDT[totevt+1];
  int genMatch_check[totevt+1];
  for (int i = 0; i < totevt+1; i++){
    scoreBDT[i] = -10;
    genMatch_check[i] = -1;
  }
  int jetIdx[totevt+1][4];
  

  Long64_t nentries = fChain->GetEntriesFast();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;

    if( scoreBDT[nevt] < BDTScore ){
      scoreBDT[nevt] = BDTScore;
      jetIdx[nevt][0] = jet0Idx;
      jetIdx[nevt][1] = jet1Idx;
      jetIdx[nevt][2] = jet2Idx;
      jetIdx[nevt][3] = jet3Idx;
      genMatch_check[nevt] = genMatch;
    }
    else continue;
  }

  int matchCount = 0;
  for (int i = 0; i <= totevt; ++ i){
    cout << "nevt = " << i << " and score = " << scoreBDT[i] << " and jet indicies are " << jetIdx[i][0] << ", " << jetIdx[i][1] << ", " << jetIdx[i][2] << ", " << jetIdx[i][3] << " and gen match is " << genMatch_check[i] << endl;
    if(genMatch_check[i] == 1111) matchCount++;
  }
  cout << matchCount << endl;
}
