#include "idxToNtuple.h"
#include <TSystem.h>
#include <string>
#include <TROOT.h>

R__LOAD_LIBRARY(idxToNtuple.C+)

void run( const char* ch, const char* ver, const char* syst_ext, const char* name )
{
  gROOT->SetBatch(1);
  string name_str = name;
  string::size_type pos;
  pos=name_str.find(' ',0);
  name_str.erase(0, pos+1);

  string base_path = "/home/minerva1993/HEPToolsFCNC/analysis_2017/reco";
  string file_path = base_path + "/score" + ch + ver + syst_ext + "/score_deepReco_" + name_str + ".root";

  ifstream file(file_path);
  if(!file.is_open()){
    cout << "No file!" << endl;
    gSystem->Exit(0);
  }

  TChain assign("tree");
  assign.Add(file_path.c_str());
  idxToNtuple t(&assign);
  t.Loop(ch);
  gSystem->Exit(0);
}
