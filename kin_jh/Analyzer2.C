#define Analyzer_cxx
#include "Analyzer2.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

#include <TLorentzVector.h>
#include <Minuit2/Minuit2Minimizer.h>

struct KinFitFtn : public ROOT::Math::IBaseFunctionMultiDim
{
  const double hadW_m_, hadT_m_, lepW_m_, lepT_m_;

  KinFitFtn(const double hadW_m, const double hadT_m,
            const double lepW_m, const double lepT_m):
    ROOT::Math::IBaseFunctionMultiDim(),
    hadW_m_(hadW_m), hadT_m_(hadT_m), lepW_m_(lepW_m), lepT_m_(lepT_m) {};

  IBaseFunctionMultiDim* Clone() const override
  {
    KinFitFtn* obj = new KinFitFtn(hadW_m_, hadT_m_, lepW_m_, lepT_m_);
    for ( size_t i=0; i<6; ++i ) obj->lvs_[i] = this->lvs_[i];
    return obj;
  }

  unsigned int NDim() const override { return 1; }

  double DoEval(const double* x) const override
  {
    const auto sol = getSolution(x);
    const auto nu = sol[0], lep = sol[1], lb = sol[2];
    const auto wj1 = sol[3], wj2 = sol[4], hb = sol[5];

    const double wLepDm = (lep+nu).M()-lepW_m_;
    const double tLepDm = (lep+nu+lb).M()-lepT_m_;
    const double wHadDm = (wj1+wj2).M()-hadW_m_;
    const double tHadDm = (wj1+wj2+hb).M()-hadT_m_;

    return wLepDm*wLepDm + tLepDm*tLepDm + wHadDm*wHadDm + tHadDm*tHadDm;
  }

  TLorentzVector solveLepTopNu(TLorentzVector nu, const TLorentzVector& lep, const TLorentzVector& lb) const
  {
    // Solve neutrino pz with M=80.4 constraint, M^2 = (E_l+E_v)^2 - (P_l+P_v)^2
    // There are two fold ambiguity, resolve one of them by minimizing the chi2
    const double pp = lepW_m_*lepW_m_/2 + (nu.Px()*lep.Px()+nu.Py()*lep.Py());
    const double a = pp*lep.Pz()/lep.Perp2();
    const double det = a*a + (pp*pp - lep.P()*lep.P()*nu.Perp2())/lep.Perp2();
    const double b = sqrt(max(0., det));
    const double wpx = nu.Px()+lep.Px();
    const double wpy = nu.Py()+lep.Py();
    TLorentzVector nuPos, nuNeg;
    nuPos.SetXYZM(nu.Px(), nu.Py(), a+b, 0);
    nuNeg.SetXYZM(nu.Px(), nu.Py(), a-b, 0);
    const auto wPos = lep+nuPos, wNeg = lep+nuNeg;

    const double tLepDmPos = abs((wPos+lb).M() - lepT_m_);
    const double tLepDmNeg = abs((wNeg+lb).M() - lepT_m_);
    const double nupz = (tLepDmPos < tLepDmNeg ) ? a+b : a-b;

    nu.SetXYZM(nu.Px(), nu.Py(), nupz, 0);
    return nu;
  }

  std::vector<TLorentzVector> getSolution(const double* x) const
  {
    const double scale_jet = x[0];
    std::vector<TLorentzVector> out;
    out.push_back(lvs_[0]); // MET
    out.push_back(lvs_[1]); // Lepton
    double dx = 0, dy = 0;
    for ( size_t i=2; i<6; ++i ) {
      out.push_back(lvs_[i]*scale_jet);
      dx += out[i].X() - lvs_[i].X();
      dy += out[i].Y() - lvs_[i].Y();
    }
    out[0].SetXYZM(out[0].Px()-dx, out[0].Py()-dy, 0, 0); // MET correction by JEC FIXME: this is an approximation, does not consider additional jets
    out[0] = solveLepTopNu(out[0], out[1], out[2]);

    return out;
  }

  TLorentzVector lvs_[6];
};

struct KinFit
{
  KinFit(const double hadW_m = 80.4, const double hadT_m = 172.5,
         const double lepW_m = 80.4, const double lepT_m = 172.5):
    ftn_(hadW_m, hadT_m, lepW_m, lepT_m)
  {
    min_.reset(new ROOT::Minuit2::Minuit2Minimizer(ROOT::Minuit2::kMigrad));
    //min_.reset(new ROOT::Minuit2::Minuit2Minimizer(ROOT::Minuit2::kCombined));
    //min_.reset(new ROOT::Minuit2::Minuit2Minimizer(ROOT::Minuit2::kScan));
    min_->SetVariable(0, "jes", 1, 0.01);
    //min_->SetLimitedVariable(0, "jes", 1, 0.1, 0.5, 1.5);
    min_->SetFunction(ftn_);
  }

  double compute(const TLorentzVector metP4, const TLorentzVector l1, const TLorentzVector b1,
                 const TLorentzVector wj1, const TLorentzVector wj2, const TLorentzVector hb);

  std::vector<TLorentzVector> getSolution() const
  {
    return ftn_.getSolution(min_->X());
  }

  std::unique_ptr<ROOT::Math::Minimizer> min_;
  KinFitFtn ftn_;
  double chi2_;

};

double KinFit::compute(const TLorentzVector metP4, const TLorentzVector l1, const TLorentzVector b1,
                       const TLorentzVector wj1, const TLorentzVector wj2, const TLorentzVector hb)
{
  ftn_.lvs_[0] = metP4;
  ftn_.lvs_[1] = l1;
  ftn_.lvs_[2] = b1;
  ftn_.lvs_[3] = wj1;
  ftn_.lvs_[4] = wj2;
  ftn_.lvs_[5] = hb;

  min_->SetVariableValue(0, 1.0);
  //min_->SetPrecision(1e-2);
  bool isValid = false;
  for ( int i=0; i<4; ++i ) {
    isValid = min_->Minimize();
    if ( isValid ) break;
  }
  chi2_ = !isValid ? 1e9 : ftn_.DoEval(min_->X());

  return chi2_;
}

void Analyzer::Loop(const int mode)
{
//   In a ROOT session, you can do:
//      root> .L Analyzer.C
//      root> Analyzer t
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
  TFile* fout = new TFile("hist.root", "recreate");
  TTree* tree = new TTree("tree", "tree");
  tree->SetDirectory(fout);

  TH1F* hLW_m = new TH1F("hLW_m", "Leptonic W Mass", 100, 0, 300);
  TH1F* hLT_m = new TH1F("hLT_m", "Leptonic Top Mass", 100, 0, 300);
  TH1F* hHW_m = new TH1F("hHW_m", "Hadronic W Mass", 100, 0, 300);
  TH1F* hHT_m = new TH1F("hHT_m", "Hadronic Top Mass", 100, 0, 300);
  TH1F* hJES = new TH1F("hJES", "Residual JES", 100, 0, 2);
  TH1F* hHadJJ_dR = new TH1F("hHadJJ_dR", "hHadJJ_dR", 100, 0, 5);
  TH1F* hAddJJ_dR = new TH1F("hAddJJ_dR", "hAddJJ_dR", 100, 0, 5);
  TH1F* hAddJJ_m = new TH1F("hAddJJ_m", "hAddJJ_m", 100, 0, 300);

  int b_run, b_event, b_vertex_n;
  int b_jets_n, b_bjets_n;
  float b_weight_gen;
  float b_lepton_pt, b_lepton_eta, b_lepton_phi;
  float b_met_pt, b_met_phi;
  float b_kin_lep_pt, b_kin_lep_eta, b_kin_lep_phi;
  float b_kin_nu_pt, b_kin_nu_eta, b_kin_nu_phi;
  float b_kin_lepB_pt, b_kin_lepB_eta, b_kin_lepB_phi, b_kin_lepB_m;
  float b_kin_lepW_pt, b_kin_lepW_eta, b_kin_lepW_phi, b_kin_lepW_m;
  float b_kin_lepT_pt, b_kin_lepT_eta, b_kin_lepT_phi, b_kin_lepT_m;
  float b_kin_hadJ1_pt, b_kin_hadJ1_eta, b_kin_hadJ1_phi, b_kin_hadJ1_m;
  float b_kin_hadJ2_pt, b_kin_hadJ2_eta, b_kin_hadJ2_phi, b_kin_hadJ2_m;
  float b_kin_hadB_pt, b_kin_hadB_eta, b_kin_hadB_phi, b_kin_hadB_m;
  float b_kin_hadW_pt, b_kin_hadW_eta, b_kin_hadW_phi, b_kin_hadW_m;
  float b_kin_hadT_pt, b_kin_hadT_eta, b_kin_hadT_phi, b_kin_hadT_m;
  float b_kin_lepB_CSV, b_kin_hadB_CSV, b_kin_hadJ1_CSV, b_kin_hadJ2_CSV;
  float b_kin_lepB_CvsB, b_kin_hadB_CvsB, b_kin_hadJ1_CvsB, b_kin_hadJ2_CvsB;
  float b_kin_lepB_CvsL, b_kin_hadB_CvsL, b_kin_hadJ1_CvsL, b_kin_hadJ2_CvsL;
  float b_kin_addJetByPt1_pt, b_kin_addJetByPt1_CSV;
  float b_kin_addJetByPt2_pt, b_kin_addJetByPt2_CSV;
  float b_kin_addJetByCSV1_pt, b_kin_addJetByCSV1_CSV;
  float b_kin_addJetByCSV2_pt, b_kin_addJetByCSV2_CSV;
  float b_kin_addJetsByPt_m, b_kin_addJetsByPt_dR;
  float b_kin_addJetsByCSV_m, b_kin_addJetsByCSV_dR;

  tree->Branch("run", &b_run, "run/I");
  tree->Branch("event", &b_event, "event/I");
  tree->Branch("vertex_n", &b_vertex_n, "vertex_n/I");
  tree->Branch("weight_gen", &b_weight_gen, "weight_gen/F");
  tree->Branch("lepton_pt", &b_lepton_pt, "lepton_pt/F");
  tree->Branch("lepton_eta", &b_lepton_eta, "lepton_eta/F");
  tree->Branch("lepton_phi", &b_lepton_phi, "lepton_phi/F");
  tree->Branch("met_pt", &b_met_pt, "met_pt/F");
  tree->Branch("met_phi", &b_met_phi, "met_phi/F");

  tree->Branch("jets_n", &b_jets_n, "jets_n/I");
  tree->Branch("bjets_n", &b_bjets_n, "bjets_n/I");
  tree->Branch("kin_lepW_m", &b_kin_lepW_m, "kin_lepW_m/F");
  tree->Branch("kin_lepT_m", &b_kin_lepT_m, "kin_lepT_m/F");
  tree->Branch("kin_hadW_m", &b_kin_hadW_m, "kin_hadW_m/F");
  tree->Branch("kin_hadT_m", &b_kin_hadT_m, "kin_hadT_m/F");

  tree->Branch("kin_lep_pt", &b_kin_lep_pt, "kin_lep_pt/F");
  tree->Branch("kin_lep_eta", &b_kin_lep_eta, "kin_lep_eta/F");
  tree->Branch("kin_lep_phi", &b_kin_lep_phi, "kin_lep_phi/F");
  tree->Branch("kin_nu_pt", &b_kin_nu_pt, "kin_nu_pt/F");
  tree->Branch("kin_nu_eta", &b_kin_nu_eta, "kin_nu_eta/F");
  tree->Branch("kin_nu_phi", &b_kin_nu_phi, "kin_nu_phi/F");
  tree->Branch("kin_lepB_pt", &b_kin_lepB_pt, "kin_lepB_pt/F");
  tree->Branch("kin_lepB_eta", &b_kin_lepB_eta, "kin_lepB_eta/F");
  tree->Branch("kin_lepB_phi", &b_kin_lepB_phi, "kin_lepB_phi/F");
  tree->Branch("kin_lepB_m", &b_kin_lepB_m, "kin_lepB_m/F");
  tree->Branch("kin_lepW_pt", &b_kin_lepW_pt, "kin_lepW_pt/F");
  tree->Branch("kin_lepW_eta", &b_kin_lepW_eta, "kin_lepW_eta/F");
  tree->Branch("kin_lepW_phi", &b_kin_lepW_phi, "kin_lepW_phi/F");
  tree->Branch("kin_lepT_pt", &b_kin_lepT_pt, "kin_lepT_pt/F");
  tree->Branch("kin_lepT_eta", &b_kin_lepT_eta, "kin_lepT_eta/F");
  tree->Branch("kin_lepT_phi", &b_kin_lepT_phi, "kin_lepT_phi/F");
  tree->Branch("kin_hadJ1_pt", &b_kin_hadJ1_pt, "kin_hadJ1_pt/F");
  tree->Branch("kin_hadJ1_eta", &b_kin_hadJ1_eta, "kin_hadJ1_eta/F");
  tree->Branch("kin_hadJ1_phi", &b_kin_hadJ1_phi, "kin_hadJ1_phi/F");
  tree->Branch("kin_hadJ1_m", &b_kin_hadJ1_m, "kin_hadJ1_m/F");
  tree->Branch("kin_hadJ2_pt", &b_kin_hadJ2_pt, "kin_hadJ2_pt/F");
  tree->Branch("kin_hadJ2_eta", &b_kin_hadJ2_eta, "kin_hadJ2_eta/F");
  tree->Branch("kin_hadJ2_phi", &b_kin_hadJ2_phi, "kin_hadJ2_phi/F");
  tree->Branch("kin_hadJ2_m", &b_kin_hadJ2_m, "kin_hadJ2_m/F");
  tree->Branch("kin_hadB_pt", &b_kin_hadB_pt, "kin_hadB_pt/F");
  tree->Branch("kin_hadB_eta", &b_kin_hadB_eta, "kin_hadB_eta/F");
  tree->Branch("kin_hadB_phi", &b_kin_hadB_phi, "kin_hadB_phi/F");
  tree->Branch("kin_hadB_m", &b_kin_hadB_m, "kin_hadB_m/F");
  tree->Branch("kin_hadW_pt", &b_kin_hadW_pt, "kin_hadW_pt/F");
  tree->Branch("kin_hadW_eta", &b_kin_hadW_eta, "kin_hadW_eta/F");
  tree->Branch("kin_hadW_phi", &b_kin_hadW_phi, "kin_hadW_phi/F");
  tree->Branch("kin_hadT_pt", &b_kin_hadT_pt, "kin_hadT_pt/F");
  tree->Branch("kin_hadT_eta", &b_kin_hadT_eta, "kin_hadT_eta/F");
  tree->Branch("kin_hadT_phi", &b_kin_hadT_phi, "kin_hadT_phi/F");

  tree->Branch("kin_lepB_CvsB", &b_kin_lepB_CvsB, "kin_lepB_CvsB/F");
  tree->Branch("kin_hadB_CvsB", &b_kin_hadB_CvsB, "kin_hadB_CvsB/F");
  tree->Branch("kin_hadJ1_CvsB", &b_kin_hadJ1_CvsB, "kin_hadJ1_CvsB/F");
  tree->Branch("kin_hadJ2_CvsB", &b_kin_hadJ2_CvsB, "kin_hadJ2_CvsB/F");

  tree->Branch("kin_lepB_CvsL", &b_kin_lepB_CvsL, "kin_lepB_CvsL/F");
  tree->Branch("kin_hadB_CvsL", &b_kin_hadB_CvsL, "kin_hadB_CvsL/F");
  tree->Branch("kin_hadJ1_CvsL", &b_kin_hadJ1_CvsL, "kin_hadJ1_CvsL/F");
  tree->Branch("kin_hadJ2_CvsL", &b_kin_hadJ2_CvsL, "kin_hadJ2_CvsL/F");

  tree->Branch("kin_addJetsByPt_m", &b_kin_addJetsByPt_m, "kin_addJetsByPt_m/F");
  tree->Branch("kin_addJetsByPt_dR", &b_kin_addJetsByPt_dR, "kin_addJetsByPt_dR/F");
  tree->Branch("kin_addJetsByCSV_m", &b_kin_addJetsByCSV_m, "kin_addJetsByCSV_m/F");
  tree->Branch("kin_addJetsByCSV_dR", &b_kin_addJetsByCSV_dR, "kin_addJetsByCSV_dR/F");

  tree->Branch("kin_addJetByPt1_pt", &b_kin_addJetByPt1_pt, "kin_addJetByPt1_pt/F");
  tree->Branch("kin_addJetByPt2_pt", &b_kin_addJetByPt2_pt, "kin_addJetByPt2_pt/F");
  tree->Branch("kin_addJetByPt1_CSV", &b_kin_addJetByPt1_CSV, "kin_addJetByPt1_CSV/F");
  tree->Branch("kin_addJetByPt2_CSV", &b_kin_addJetByPt2_CSV, "kin_addJetByPt2_CSV/F");

  tree->Branch("kin_addJetByCSV1_pt", &b_kin_addJetByCSV1_pt, "kin_addJetByCSV1_pt/F");
  tree->Branch("kin_addJetByCSV2_pt", &b_kin_addJetByCSV2_pt, "kin_addJetByCSV2_pt/F");
  tree->Branch("kin_addJetByCSV1_CSV", &b_kin_addJetByCSV1_CSV, "kin_addJetByCSV1_CSV/F");
  tree->Branch("kin_addJetByCSV2_CSV", &b_kin_addJetByCSV2_CSV, "kin_addJetByCSV2_CSV/F");

  if (fChain == 0) return;

  KinFit fit(mode == 2 ? 125 : 80.4);

  //Long64_t nentries = 10000;
  Long64_t nentries = fChain->GetEntries();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
    cout << jentry << '/' << nentries << '\r';

    b_run = run;
    b_event = event;
    b_vertex_n = GoodPV;
    b_weight_gen = genweight;

    TLorentzVector leptonP4;
    leptonP4.SetPtEtaPhiE(lepton_pT, lepton_eta, lepton_phi, lepton_E);
    if ( leptonP4.Pt() < 30 or std::abs(leptonP4.Eta()) > 2.1 ) continue;

    b_lepton_pt = lepton_pT;
    b_lepton_eta = lepton_eta;
    b_lepton_phi = lepton_phi;

    TLorentzVector metP4;
    metP4.SetPtEtaPhiM(MET, 0, MET_phi, 0);

    b_met_pt = MET;
    b_met_phi = MET_phi;

    std::vector<size_t> jetIdxs;
    const auto jets_pt = *jet_pT, jets_eta = *jet_eta, jets_phi = *jet_phi, jets_e = *jet_E;
    const auto jets_CSV = *jet_CSV, jets_CvsB = *jet_CvsB, jets_CvsL = *jet_CvsL;
    b_bjets_n = 0;
    for ( int j=0; j<jet_number; ++j ) {
      if ( jets_pt[j] < 30 or std::abs(jets_eta[j]) > 2.5 ) continue;
      if ( jets_CSV[j] > 0.9535 ) ++b_jets_n;
      jetIdxs.push_back(j);
    }
    if ( jetIdxs.size() < 4 ) continue;
    b_jets_n = jetIdxs.size();

    double bestChi2 = 1e9;
    std::vector<size_t> bestIdxs;
    TLorentzVector jetP4s[4];
    for ( size_t j1 : jetIdxs ) {
      //if ( !jets_bTag[j1] ) continue;
      jetP4s[0].SetPtEtaPhiE(jets_pt[j1], jets_eta[j1], jets_phi[j1], jets_e[j1]);
      for ( size_t j2 : jetIdxs ) {
        if ( j2 == j1 ) continue;
        //if ( mode != 1 and !jets_bTag[j2] ) continue;
        jetP4s[1].SetPtEtaPhiE(jets_pt[j2], jets_eta[j2], jets_phi[j2], jets_e[j2]);
        for ( size_t j3 : jetIdxs ) {
          if ( j2 >= j3 ) continue;
          if ( j3 == j1 ) continue;
          //if ( mode != 1 and !jets_bTag[j3] ) continue;
          jetP4s[2].SetPtEtaPhiE(jets_pt[j3], jets_eta[j3], jets_phi[j3], jets_e[j3]);
          for ( size_t j4 : jetIdxs ) {
            if ( j4 == j3 or j4 == j2 or j4 == j1 ) continue;
            //if ( mode == 1 and !jets_bTag[j4] ) continue;
            jetP4s[3].SetPtEtaPhiE(jets_pt[j4], jets_eta[j4], jets_phi[j4], jets_e[j4]);
            const double chi2 = fit.compute(metP4, leptonP4, jetP4s[0], jetP4s[1], jetP4s[2], jetP4s[3]);

            if ( chi2 < bestChi2 ) {
              bestChi2 = chi2;
              bestIdxs = {j1, j2, j3, j4};
            }
          }
        }
      }
    }
    if ( !bestIdxs.empty() ) {
      for ( size_t i=0; i<4; ++i ) {
        const size_t j = bestIdxs[i];
        jetP4s[i].SetPtEtaPhiE(jets_pt[j], jets_eta[j], jets_phi[j], jets_e[j]);
      }
      const double chi2 = fit.compute(metP4, leptonP4, jetP4s[0], jetP4s[1], jetP4s[2], jetP4s[3]);
      const std::vector<TLorentzVector> solution = fit.getSolution();
      const auto& sol_nuP4 = solution[0], sol_lepP4 = solution[1], sol_ljP4 = solution[2];
      const auto& sol_wj1P4 = solution[3], sol_wj2P4 = solution[4], sol_hbP4 = solution[5];

      const double hadJJ_dR = sol_wj1P4.DeltaR(sol_wj2P4);
      hHadJJ_dR->Fill(hadJJ_dR);
      hLW_m->Fill( (sol_lepP4+sol_nuP4).M() );
      hLT_m->Fill( (sol_lepP4+sol_nuP4+sol_ljP4).M() );
      hHW_m->Fill( (sol_wj1P4+sol_wj2P4).M() );
      hHT_m->Fill( (sol_wj1P4+sol_wj2P4+sol_hbP4).M() );

      hJES->Fill(fit.min_->X()[0]);

      b_kin_lep_pt = sol_lepP4.Pt(); b_kin_lep_eta = sol_lepP4.Eta(); b_kin_lep_phi = sol_lepP4.Phi();
      b_kin_nu_pt = sol_nuP4.Pt(); b_kin_nu_eta = sol_nuP4.Eta(); b_kin_nu_phi = sol_nuP4.Phi(); 
      b_kin_lepB_pt = sol_ljP4.Pt(); b_kin_lepB_eta = sol_ljP4.Eta(); b_kin_lepB_phi = sol_ljP4.Phi(); b_kin_lepB_m = sol_ljP4.M(); 
      b_kin_hadJ1_pt = sol_wj1P4.Pt(); b_kin_hadJ1_eta = sol_wj1P4.Eta(); b_kin_hadJ1_phi = sol_wj1P4.Phi(); b_kin_hadJ1_m = sol_wj1P4.M(); 
      b_kin_hadJ2_pt = sol_wj2P4.Pt(); b_kin_hadJ2_eta = sol_wj2P4.Eta(); b_kin_hadJ2_phi = sol_wj2P4.Phi(); b_kin_hadJ2_m = sol_wj2P4.M(); 
      b_kin_lepB_pt = sol_hbP4.Pt(); b_kin_lepB_eta = sol_hbP4.Eta(); b_kin_lepB_phi = sol_hbP4.Phi(); b_kin_lepB_m = sol_hbP4.M(); 

      const auto lepW = sol_lepP4+sol_nuP4;
      const auto lepT = lepW+sol_ljP4;
      b_kin_lepW_pt = lepW.Pt(); b_kin_lepW_eta = lepW.Eta(); b_kin_lepW_phi = lepW.Phi(); b_kin_lepW_m = lepW.M(); 
      b_kin_lepT_pt = lepT.Pt(); b_kin_lepT_eta = lepT.Eta(); b_kin_lepT_phi = lepT.Phi(); b_kin_lepT_m = lepT.M(); 

      const auto hadW = sol_wj1P4+sol_wj2P4;
      const auto hadT = hadW+sol_hbP4;
      b_kin_hadW_pt = hadW.Pt(); b_kin_hadW_eta = hadW.Eta(); b_kin_hadW_phi = hadW.Phi(); b_kin_hadW_m = hadW.M(); 
      b_kin_hadT_pt = hadT.Pt(); b_kin_hadT_eta = hadT.Eta(); b_kin_hadT_phi = hadT.Phi(); b_kin_hadT_m = hadT.M(); 

      b_kin_lepB_CSV = jets_CSV[bestIdxs[0]];
      b_kin_hadJ1_CSV = jets_CSV[bestIdxs[1]];
      b_kin_hadJ2_CSV = jets_CSV[bestIdxs[2]];
      b_kin_hadB_CSV = jets_CSV[bestIdxs[3]];

      b_kin_lepB_CvsB = jets_CvsB[bestIdxs[0]];
      b_kin_hadJ1_CvsB = jets_CvsB[bestIdxs[1]];
      b_kin_hadJ2_CvsB = jets_CvsB[bestIdxs[2]];
      b_kin_hadB_CvsB = jets_CvsB[bestIdxs[3]];

      b_kin_lepB_CvsL = jets_CvsL[bestIdxs[0]];
      b_kin_hadJ1_CvsL = jets_CvsL[bestIdxs[1]];
      b_kin_hadJ2_CvsL = jets_CvsL[bestIdxs[2]];
      b_kin_hadB_CvsL = jets_CvsL[bestIdxs[3]];

      std::vector<size_t> addJetIdxs;
      for ( size_t j : jetIdxs ) {
        bool isTaken = false;
        for ( size_t bestIdx : bestIdxs ) {
          if ( bestIdx == j ) { isTaken = true; break; }
        }
        if ( isTaken ) continue;

        addJetIdxs.push_back(j);
      }
      if ( addJetIdxs.size() < 2 ) {
        b_kin_addJetByPt1_pt = b_kin_addJetByPt2_pt = b_kin_addJetByCSV1_pt = b_kin_addJetByCSV2_pt = 0;
        b_kin_addJetByPt1_CSV = b_kin_addJetByPt2_CSV = b_kin_addJetByCSV1_CSV = b_kin_addJetByCSV2_CSV = -10;
        b_kin_addJetsByPt_m = b_kin_addJetsByCSV_m = 0;
        b_kin_addJetsByPt_dR = b_kin_addJetsByCSV_dR = 0;
      }
      else {
        auto addJetIdxsByPt = addJetIdxs;
        std::sort(addJetIdxsByPt.begin(), addJetIdxsByPt.end(), [&](size_t a, size_t b) { return jets_pt[a] > jets_pt[b]; });
        std::sort(addJetIdxs.begin(), addJetIdxs.end(), [&](size_t a, size_t b) { return jets_CSV[a] > jets_CSV[b]; });
        const size_t jByPt1 = addJetIdxsByPt[0], jByPt2 = addJetIdxsByPt[1];
        const size_t jByCSV1 = addJetIdxs[0], jByCSV2 = addJetIdxs[1];
        TLorentzVector addJetByCSV1, addJetByCSV2, addJetByPt1, addJetByPt2;
        addJetByCSV1.SetPtEtaPhiE(jets_pt[jByPt1], jets_eta[jByPt1], jets_phi[jByPt1], jets_e[jByPt1]);
        addJetByCSV1.SetPtEtaPhiE(jets_pt[jByPt2], jets_eta[jByPt2], jets_phi[jByPt2], jets_e[jByPt2]);
        addJetByPt1.SetPtEtaPhiE(jets_pt[jByCSV1], jets_eta[jByCSV1], jets_phi[jByCSV1], jets_e[jByCSV1]);
        addJetByPt1.SetPtEtaPhiE(jets_pt[jByCSV2], jets_eta[jByCSV2], jets_phi[jByCSV2], jets_e[jByCSV2]);

        b_kin_addJetByPt1_pt = jets_pt[jByPt1];
        b_kin_addJetByPt2_pt = jets_pt[jByPt2];
        b_kin_addJetByCSV1_pt = jets_pt[jByCSV1];
        b_kin_addJetByCSV2_pt = jets_pt[jByCSV2];
        b_kin_addJetByPt1_CSV = jets_CSV[jByPt1];
        b_kin_addJetByPt2_CSV = jets_CSV[jByPt2];
        b_kin_addJetByCSV1_CSV = jets_CSV[jByCSV1];
        b_kin_addJetByCSV2_CSV = jets_CSV[jByCSV2];
        b_kin_addJetsByPt_dR = addJetByPt1.DeltaR(addJetByPt2);
        b_kin_addJetsByCSV_dR = addJetByCSV1.DeltaR(addJetByCSV2);
        b_kin_addJetsByPt_m = (addJetByPt1+addJetByPt2).M();
        b_kin_addJetsByCSV_m = (addJetByCSV1+addJetByCSV2).M();

        hAddJJ_dR->Fill(b_kin_addJetsByCSV_dR);
        hAddJJ_m->Fill(b_kin_addJetsByCSV_m);
        
      }
    }

    tree->Fill();
  }

  fout->Write();

  TCanvas* c = 0;
  c = new TCanvas("cHadJJ_dR", "hadJJ dR", 500, 500); hHadJJ_dR->Draw();
  c = new TCanvas("cAddJJ_dR", "addJJ dR", 500, 500); hAddJJ_dR->Draw();
  c = new TCanvas("cHadJJ_m", "hadJJ m", 500, 500); hAddJJ_m->Draw();
  c = new TCanvas("cLW_m", "MW lep", 500, 500); hLW_m->Draw();
  c = new TCanvas("cLT_m", "MTop lep", 500, 500); hLT_m->Draw();
  c = new TCanvas("cHW_m", "MW had", 500, 500); hHW_m->Draw();
  c = new TCanvas("cHT_m", "MTop had", 500, 500); hHT_m->Draw();
  c = new TCanvas("cJES", "JES", 500, 500); hJES->Draw();
}
