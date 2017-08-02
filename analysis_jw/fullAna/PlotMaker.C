#include <map>

#include <TH1.h>
#include <TH2.h>
#include <THStack.h>

#include "PlotMaker.h"
const double LUMINOSITY = 35867;
string INPUT = "../output/root/";
string OUTPUT = "../output/pdf/";


DataFile::DataFile(string input_data){
  string fileName = "hist_" + input_data + ".root";
  this->file = TFile::Open(fileName.c_str());
  this->data = input_data;

  int tmp = 0;
  TIter next(file->GetListOfKeys());
  TKey *key;
  TObject *obj;
  while((key = (TKey*)next())){
    obj = key->ReadObj();
    ++tmp;
    string tmp = obj->GetName();
    ReplaceAll(tmp, this->data, "");
    this->v_histName.push_back(tmp);
  }
  this->nHist = tmp;
}

MonteFile::MonteFile(string input_sample, string input_category,  int input_color, double xsec){
  string fileName = "hist_" + input_sample + ".root";
  this->file = TFile::Open(fileName.c_str());
  this->color = input_color;
  this->sample = input_sample;
  this->category = input_category;

  int tmp = 0;
  TIter next(file->GetListOfKeys());
  TKey *key;
  TObject *obj;
  while((key = (TKey*)next())){
    obj = key->ReadObj();
    ++tmp;
    string tmp = obj->GetName();
    ReplaceAll(tmp, this->sample, "");
    this->v_histName.push_back(tmp);
  }
  this->nHist = tmp;
  TH1D *EventInfo = (TH1D *)this->file->Get("EventInfo");

  this->scale_Mu = (LUMINOSITY*xsec)/EventInfo->GetBinContent(2);
  this->scale_EG = (LUMINOSITY*xsec)/EventInfo->GetBinContent(2);
}


void PlotMaker(){
  std::map<string, DataFile *> m_Data;
  std::map<char *, MonteFile *> m_Monte;

  //sample List
  char *ttbb = Form("ttbb");
  char *ttbj = Form("ttbj");
  char *ttcc = Form("ttcc");
  char *tt = Form("tt");
  char *ttLF = Form("ttLF");
  //char *ttbkg = Form("ttbkg");
  char *wjets = Form("wjets");
  char *zjets = Form("zjets");
  char *zjets10to50 = Form("zjets10to50");
  char *tchannel = Form("tchannel");
  char *tbarchannel = Form("tbarchannel");
  char *tWchannel = Form("tWchannel");
  char *tbarWchannel = Form("tbarWchannel");
  char *ww = Form("ww");
  char *wz = Form("wz");
  char *zz = Form("zz");

  m_Data["Mu"] = new DataFile("DataSingleMu");
  m_Data["EG"] = new DataFile("DataSingleEG");
  m_Monte[ttbb] = new MonteFile("ttbb","ttbb", kRed+4, 356.4);
  m_Monte[ttbj] = new MonteFile("ttbj", "ttbj", kRed+3, 356.4);
  m_Monte[ttcc] = new MonteFile("ttcc", "ttcc", kRed+2, 356.4);
  m_Monte[tt] = new MonteFile("tt", "ttLF", kRed, 356.4);
  m_Monte[ttLF] = new MonteFile("ttLF", "ttLF", kRed, 356.4);
  //m_Monte[ttbkg] = new MonteFile("ttbkg", "ttLF", kRed, 831.8);
  m_Monte[wjets] = new MonteFile("wjets", "wjets", kYellow, 61524);
  m_Monte[zjets] = new MonteFile("zjets", "zjets", kBlue, 6025.2);
  m_Monte[zjets10to50] = new MonteFile("zjets10to50", "zjets", kBlue, 18610.0);
  m_Monte[tchannel] = new MonteFile("tchannel", "single t", 6, 44.33);
  m_Monte[tbarchannel] = new MonteFile("tbarchannel", "single t", 6, 26.38);
  m_Monte[tWchannel] = new MonteFile("tWchannel", "single t", 6, 35.6);
  m_Monte[tbarWchannel] = new MonteFile("tbarWchannel", "single t", 6, 35.6);
  m_Monte[ww] = new MonteFile("ww", "diBoson", kCyan, 118.7);
  m_Monte[wz] = new MonteFile("wz", "diBoson", kCyan, 47.13);
  m_Monte[zz] = new MonteFile("zz", "diBoson", kCyan, 16.523);
 
  int nHist=0;
  for(auto v_itr = m_Monte[ttbb]->v_histName.begin(); v_itr != m_Monte[ttbb]->v_histName.end(); ++v_itr){
    cout << "HIST NAME : " << *v_itr << endl;
    if( *v_itr == "EventInfo" ) continue;
    std::vector<string> v_postSample;
    
    //Histogram form
    TCanvas *canvas = new TCanvas("c","canvas",500,500);
    
    THStack *hs_tmp = new THStack("st","stack");
   
    TPaveText *label = new TPaveText();
    label->SetX1NDC(0.63);
    label->SetX2NDC(1.0);
    label->SetY1NDC(0.88);
    label->SetY2NDC(0.93);
    label->SetTextFont(42);
    label->AddText("CMS, 35.9 fb^{-1} at #sqrt{s} = 13TeV");
    label->SetFillStyle(0);
    label->SetBorderSize(0);
    label->SetTextSize(0.04);

    TLegend *leg = new TLegend(0.15, 0.71, 0.89, 0.87);
    leg->SetNColumns(4);
    leg->SetTextSize(0.05);
    leg->SetLineColor(0);
    leg->SetFillColor(0);

    TPad *pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
    pad1->SetBottomMargin(0.02);
    //pad1->SetGridx();
    pad1->SetFrameFillColor(0);
    pad1->Draw();

    TPad *pad2 = new TPad("pad2", "pad2", 0, 0, 1, 0.3);
    pad2->SetTopMargin(0.02);
    pad2->SetBottomMargin(0.3);
    pad2->SetGridx();
    pad2->SetGridy();
    pad2->SetFrameFillColor(0);
    pad2->Draw();

    pad1->cd();
    
    int channel = 999;
    if(v_itr->find("Ch0")) channel = 0;
    else if(v_itr->find("Ch1")) channel = 1;

    for(auto m_itr = m_Monte.begin(); m_itr != m_Monte.end(); ++m_itr){
      cout << "SAMPLE : " << m_itr->second->sample << endl;
      string histName = *v_itr + m_itr->second->sample;

      auto h_tmp = (TH1D *)m_itr->second->file->Get(histName.c_str());
      if( h_tmp == NULL ){
        cout << "FUCKING NULL POINTER" << endl;
      }
      if(channel == 0) {
        h_tmp->Scale(m_itr->second->scale_Mu); 
      }
      else if(channel == 1){
        cout << "scale : " << m_itr->second->scale_EG << endl;
        h_tmp->Scale(m_itr->second->scale_EG);
      }
      else break;

      auto v_itr2 = find_if(v_postSample.begin(), v_postSample.end(),
        [m_itr](string postSample)->bool{ return m_itr->second->category == postSample; });
      h_tmp->SetFillColor(m_itr->second->color);
      h_tmp->SetLineColor(m_itr->second->color);
      if( v_itr2 == v_postSample.end() )
        leg->AddEntry(h_tmp, (m_itr->second->category).c_str(), "F");
      if( std::next(m_itr) != m_Monte.end() && m_itr->second->category != std::next(m_itr)->second->category )
        h_tmp->SetLineColor(kBlack);


      v_postSample.push_back(m_itr->second->category);
      
      int nEvent = h_tmp->Integral();
      int nRaw = h_tmp->Integral();

      h_tmp->Draw("hist");
      hs_tmp->Add(h_tmp);
    }
    TH1D * h_data;
    if(channel == 0) {
      string hName = *v_itr + "DataSingleMu";
      h_data = (TH1D *)m_Data["Mu"]->file->Get(hName.c_str());
    }
    else if(channel == 1){
      string hName = *v_itr + "DataSingleEG";
      h_data = (TH1D *)m_Data["EG"]->file->Get(hName.c_str());
    }
    else break;

    TH1D *h_stackTohisto = (TH1D *)hs_tmp->GetStack()->Last();

    h_data->GetYaxis()->SetLabelSize(0.04);
    h_data->GetYaxis()->SetTitle("Number of Event");
    
    h_data->GetXaxis()->SetLabelSize(0.0);

    h_data->SetStats(0);
    h_data->SetMarkerStyle(20);
    h_data->SetMarkerSize(0.9);
    if( h_data->GetMaximum() > hs_tmp->GetMaximum() )
      h_data->SetMaximum(hs_tmp->GetMaximum()+h_data->GetMaximum()*1000);
    else
      h_data->SetMaximum(hs_tmp->GetMaximum()*1000+h_data->GetMaximum());
    h_data->SetMinimum(0.01);
    
    h_data->Draw("P");
    hs_tmp->Draw("hist SAME");
    h_data->Draw("PSAME");
    label->Draw("SAME");
    leg->Draw("SAME");
    //pad1->SetLogy();

    canvas->cd();
    pad2->cd();

    TH1D * h_div = (TH1D *)h_data->Clone("h_div");
    h_div->Divide(h_stackTohisto);
    
    h_div->SetTitle("");
    
    h_div->GetYaxis()->SetTitle("Data/MC");
    h_div->GetYaxis()->SetTitleSize(0.1);
    h_div->GetYaxis()->SetTitleOffset(0.43);
    h_div->GetYaxis()->SetLabelSize(0.1);
    
    h_div->GetXaxis()->SetTitleSize(0.1);
    h_div->GetXaxis()->SetTitleOffset(0.90);
    h_div->GetXaxis()->SetLabelSize(0.1);
    
    h_div->SetLineColor(kBlack);
    h_div->SetMinimum(0.8);
    h_div->SetMaximum(1.2);
    h_div->Sumw2();
    h_div->SetStats(0);
    
    h_div->Draw();

    if(nHist==0)
      canvas->Print("result_ratio1.pdf(","pdf");
    else if(nHist == (m_Monte[ttbb]->nHist)-2)
      canvas->Print("result_ratio1.pdf)","pdf");
    else
      canvas->Print("result_ratio1.pdf","pdf");
    ++nHist;
  }
}
