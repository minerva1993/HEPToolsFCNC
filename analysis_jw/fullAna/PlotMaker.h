/*******************************************************************
 * Last Update : 2017-07-30                                        *
 * Developer : Seohyun                                             *
 * This macro is used as substitute for makePlots.py               *
 *******************************************************************/

#include <iostream>
#include <string>
#include <vector>

#include <TFile.h>
#include <TObject.h>

//-----------------------------------------------------------------//
//  Histogram file reader                                          //
//-----------------------------------------------------------------//
class DataFile{
  private:
  public:
    TFile *file;
    string data;
    vector<string> v_histName;
    int nHist;
    
    DataFile(string data);
    DataFile(DataFile &dataFile){};
    ~DataFile(){};
};

class MonteFile{
  private:
  public:
    TFile *file;
    string sample;
    string category;
    vector<string> v_histName;
    double scale_Mu;
    double scale_EG;
    int color;
    int nHist;
    
    MonteFile(string sample, string category, int color, double xsec);
    MonteFile(MonteFile &MonetFile){};
    ~MonteFile(){};
};

std::string ReplaceAll(std::string &str, const std::string &from, const std::string &to){
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos){
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return str;
}
