#include "idxToNtuple.h"
#include <TSystem.h>
#include <string>

R__LOAD_LIBRARY(idxToNtuple.C+)

void run( const char* ver, const char* name )
{
  string name_str = name;
  string::size_type pos;
  pos=name_str.find(' ',0);
  name_str.erase(0, pos+1);

  string base_path = "/home/minerva1993/HEPToolsFCNC/analysis_2017/reco/classifier/2017";
  string file_path = base_path + "/scoreSTFCNC" + ver + "/score_deepReco_" + name_str + ".root";

  ifstream file(file_path);
  if(!file.is_open()){
    cout << "No file!" << endl;
    return true;
  }

  TChain assign("tree");
  assign.Add(file_path.c_str());
  idxToNtuple t(&assign);
  t.Loop();
  gSystem->Exit(0);
}
