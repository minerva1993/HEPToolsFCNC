#define makeTuple_cxx

#include "makeTuple.h"
#include <TH2.h>
#include <TStyle.h>
#include <string>

void makeTuple::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void makeTuple::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();

    //////////////////////////////////////////////////////////////////
    sigTree = new TTree("sig_tree","signal tree");

    sigTree->Branch("nevt"         , &b_nevt         , "nevt/I");
    sigTree->Branch("file"         , &b_file         , "file/I");
    sigTree->Branch("GoodPV"       , &b_GoodPV       , "GoodPV/I");
    sigTree->Branch("EventCategory", &b_EventCategory, "EventCategory/I");
    sigTree->Branch("genMatch"     , &b_genMatch     , "genMatch/I");
    sigTree->Branch("EventWeight"  , &b_EventWeight  , "EventWeight/F");

    sigTree->Branch("njets"     , &b_njets     , "njets/I");
    sigTree->Branch("nbjets_m"  , &b_nbjets_m  , "nbjets_m/I");
    sigTree->Branch("lepton_pt" , &b_lepton_pt , "lepton_pt/F");
    sigTree->Branch("lepton_eta", &b_lepton_eta, "lepton_eta/F");
    sigTree->Branch("lepton_phi", &b_lepton_phi, "lepton_phi/F");
    sigTree->Branch("MET"       , &b_met       , "MET/F");
    sigTree->Branch("MET_phi"   , &b_met_phi   , "MET_phi/F");
    sigTree->Branch("lepDPhi"   , &b_lepdphi   , "lepDPhi/F");

    //jet assignment
    sigTree->Branch("jet0pt"  , &b_jet0pt  , "jet0pt/F");
    sigTree->Branch("jet0eta" , &b_jet0eta , "jet0eta/F");
    sigTree->Branch("jet0phi" , &b_jet0phi , "jet0phi/F");
    sigTree->Branch("jet0m"   , &b_jet0m   , "jet0m/F");
    sigTree->Branch("jet0csv" , &b_jet0csv , "jet0csv/F");
    sigTree->Branch("jet0cvsl", &b_jet0cvsl, "jet0cvsl/F");
    sigTree->Branch("jet0cvsb", &b_jet0cvsb, "jet0cvsb/F");
    sigTree->Branch("jet0Idx" , &b_jet0Idx , "jet0Idx/I");

    sigTree->Branch("jet1pt"  , &b_jet1pt  , "jet1pt/F");
    sigTree->Branch("jet1eta" , &b_jet1eta , "jet1eta/F");
    sigTree->Branch("jet1phi" , &b_jet1phi , "jet1phi/F");
    sigTree->Branch("jet1m"   , &b_jet1m   , "jet1m/F");
    sigTree->Branch("jet1csv" , &b_jet1csv , "jet1csv/F");
    sigTree->Branch("jet1cvsl", &b_jet1cvsl, "jet1cvsl/F");
    sigTree->Branch("jet1cvsb", &b_jet1cvsb, "jet1cvsb/F");
    sigTree->Branch("jet1Idx" , &b_jet1Idx , "jet1Idx/I");

    sigTree->Branch("jet2pt"  , &b_jet2pt  , "jet2pt/F");
    sigTree->Branch("jet2eta" , &b_jet2eta , "jet2eta/F");
    sigTree->Branch("jet2phi" , &b_jet2phi , "jet2phi/F");
    sigTree->Branch("jet2m"   , &b_jet2m   , "jet2m/F");
    sigTree->Branch("jet2csv" , &b_jet2csv , "jet2csv/F");
    sigTree->Branch("jet2cvsl", &b_jet2cvsl, "jet2cvsl/F");
    sigTree->Branch("jet2cvsb", &b_jet2cvsb, "jet2cvsb/F");
    sigTree->Branch("jet2Idx" , &b_jet2Idx , "jet2Idx/I");

    sigTree->Branch("jet3pt"  , &b_jet3pt  , "jet3pt/F");
    sigTree->Branch("jet3eta" , &b_jet3eta , "jet3eta/F");
    sigTree->Branch("jet3phi" , &b_jet3phi , "jet3phi/F");
    sigTree->Branch("jet3m"   , &b_jet3m   , "jet3m/F");
    sigTree->Branch("jet3csv" , &b_jet3csv , "jet3csv/F");
    sigTree->Branch("jet3cvsl", &b_jet3cvsl, "jet3cvsl/F");
    sigTree->Branch("jet3cvsb", &b_jet3cvsb, "jet3cvsb/F");
    sigTree->Branch("jet3Idx" , &b_jet3Idx , "jet3Idx/I");

    sigTree->Branch("jet12pt"  , &b_jet12pt  , "jet12pt/F");
    sigTree->Branch("jet12eta" , &b_jet12eta , "jet12eta/F");
    sigTree->Branch("jet12deta", &b_jet12deta, "jet12deta/F");
    sigTree->Branch("jet12phi" , &b_jet12phi , "jet12phi/F");
    sigTree->Branch("jet12dphi", &b_jet12dphi, "jet12dphi/F");
    sigTree->Branch("jet12dR"  , &b_jet12dR  , "jet12dR/F");
    sigTree->Branch("jet12m"   , &b_jet12m   , "jet12m/F");

    sigTree->Branch("jet23pt"  , &b_jet23pt  , "jet23pt/F");
    sigTree->Branch("jet23eta" , &b_jet23eta , "jet23eta/F");
    sigTree->Branch("jet23deta", &b_jet23deta, "jet23deta/F");
    sigTree->Branch("jet23phi" , &b_jet23phi , "jet23phi/F");
    sigTree->Branch("jet23dphi", &b_jet23dphi, "jet23dphi/F");
    sigTree->Branch("jet23dR"  , &b_jet23dR  , "jet23dR/F");
    sigTree->Branch("jet23m"   , &b_jet23m   , "jet23m/F");

    sigTree->Branch("jet31pt"  , &b_jet31pt  , "jet31pt/F");
    sigTree->Branch("jet31eta" , &b_jet31eta , "jet31eta/F");
    sigTree->Branch("jet31deta", &b_jet31deta, "jet31deta/F");
    sigTree->Branch("jet31phi" , &b_jet31phi , "jet31phi/F");
    sigTree->Branch("jet31dphi", &b_jet31dphi, "jet31dphi/F");
    sigTree->Branch("jet31dR"  , &b_jet31dR  , "jet31dR/F");
    sigTree->Branch("jet31m"   , &b_jet31m   , "jet31m/F");

    sigTree->Branch("lepWpt"  , &b_lepWpt  , "lepWpt/F");
    sigTree->Branch("lepWeta" , &b_lepWeta , "lepWeta/F");
    sigTree->Branch("lepWphi" , &b_lepWphi , "lepWphi/F");
    sigTree->Branch("lepWdphi", &b_lepWdphi, "lepWdphi/F");
    sigTree->Branch("lepWm"   , &b_lepWm   , "lepWm/F");

    sigTree->Branch("lepTpt"  , &b_lepTpt  , "lepTpt/F");
    sigTree->Branch("lepTeta" , &b_lepTeta , "lepTeta/F");
    sigTree->Branch("lepTdeta", &b_lepTdeta, "lepTdeta/F");
    sigTree->Branch("lepTphi" , &b_lepTphi , "lepTphi/F");
    sigTree->Branch("lepTdphi", &b_lepTdphi, "lepTdphi/F");//W and b
    sigTree->Branch("lepTdR"  , &b_lepTdR  , "lepTdR/F");
    sigTree->Branch("lepTm"   , &b_lepTm   , "lepTm/F");

    sigTree->Branch("hadTpt"      , &b_hadTpt      , "hadTpt/F");
    sigTree->Branch("hadTeta"     , &b_hadTeta     , "hadTeta/F");
    sigTree->Branch("hadT12_3deta", &b_hadT12_3deta, "hadT12_3deta/F");
    sigTree->Branch("hadT23_1deta", &b_hadT23_1deta, "hadT23_1deta/F");
    sigTree->Branch("hadT31_2deta", &b_hadT31_2deta, "hadT31_2deta/F");
    sigTree->Branch("hadTphi"     , &b_hadTphi     , "hadTphi/F");
    sigTree->Branch("hadT12_3dphi", &b_hadT12_3dphi, "hadT12_3dphi/F");
    sigTree->Branch("hadT23_1dphi", &b_hadT23_1dphi, "hadT23_1dphi/F");
    sigTree->Branch("hadT31_2dphi", &b_hadT31_2dphi, "hadT31_2dphi/F");
    sigTree->Branch("hadT12_3dR"  , &b_hadT12_3dR  , "hadT12_3dR/F");
    sigTree->Branch("hadT23_1dR"  , &b_hadT23_1dR  , "hadT23_1dR/F");
    sigTree->Branch("hadT31_2dR"  , &b_hadT31_2dR  , "hadT31_2dR/F");
    sigTree->Branch("hadTm"       , &b_hadTm       , "hadTm/F");
    sigTree->Branch("genHadW"     , &b_genHadW     , "genHadW/F");

    fOutput->Add(sigTree);

    ///////////////////////////////////////////////////////////////////
    bkgTree = new TTree("bkg_tree","background tree");

    bkgTree->Branch("nevt"         , &b_nevt         , "nevt/I");
    bkgTree->Branch("file"         , &b_file         , "file/I");
    bkgTree->Branch("GoodPV"       , &b_GoodPV       , "GoodPV/I");
    bkgTree->Branch("EventCategory", &b_EventCategory, "EventCategory/I");
    bkgTree->Branch("genMatch"     , &b_genMatch     , "genMatch/I");
    bkgTree->Branch("EventWeight"  , &b_EventWeight  , "EventWeight/F");

    bkgTree->Branch("njets"     , &b_njets     , "njets/I");
    bkgTree->Branch("nbjets_m"  , &b_nbjets_m  , "nbjets_m/I");
    bkgTree->Branch("lepton_pt" , &b_lepton_pt , "lepton_pt/F");
    bkgTree->Branch("lepton_eta", &b_lepton_eta, "lepton_eta/F");
    bkgTree->Branch("lepton_phi", &b_lepton_phi, "lepton_phi/F");
    bkgTree->Branch("MET"       , &b_met       , "MET/F");
    bkgTree->Branch("MET_phi"   , &b_met_phi   , "MET_phi/F");
    bkgTree->Branch("lepDPhi"   , &b_lepdphi   , "lepDPhi/F");

    //jet assignment
    bkgTree->Branch("jet0pt"  , &b_jet0pt  , "jet0pt/F");
    bkgTree->Branch("jet0eta" , &b_jet0eta , "jet0eta/F");
    bkgTree->Branch("jet0phi" , &b_jet0phi , "jet0phi/F");
    bkgTree->Branch("jet0m"   , &b_jet0m   , "jet0m/F");
    bkgTree->Branch("jet0csv" , &b_jet0csv , "jet0csv/F");
    bkgTree->Branch("jet0cvsl", &b_jet0cvsl, "jet0cvsl/F");
    bkgTree->Branch("jet0cvsb", &b_jet0cvsb, "jet0cvsb/F");
    bkgTree->Branch("jet0Idx" , &b_jet0Idx , "jet0Idx/I");

    bkgTree->Branch("jet1pt"  , &b_jet1pt  , "jet1pt/F");
    bkgTree->Branch("jet1eta" , &b_jet1eta , "jet1eta/F");
    bkgTree->Branch("jet1phi" , &b_jet1phi , "jet1phi/F");
    bkgTree->Branch("jet1m"   , &b_jet1m   , "jet1m/F");
    bkgTree->Branch("jet1csv" , &b_jet1csv , "jet1csv/F");
    bkgTree->Branch("jet1cvsl", &b_jet1cvsl, "jet1cvsl/F");
    bkgTree->Branch("jet1cvsb", &b_jet1cvsb, "jet1cvsb/F");
    bkgTree->Branch("jet1Idx" , &b_jet1Idx , "jet1Idx/I");

    bkgTree->Branch("jet2pt"  , &b_jet2pt  , "jet2pt/F");
    bkgTree->Branch("jet2eta" , &b_jet2eta , "jet2eta/F");
    bkgTree->Branch("jet2phi" , &b_jet2phi , "jet2phi/F");
    bkgTree->Branch("jet2m"   , &b_jet2m   , "jet2m/F");
    bkgTree->Branch("jet2csv" , &b_jet2csv , "jet2csv/F");
    bkgTree->Branch("jet2cvsl", &b_jet2cvsl, "jet2cvsl/F");
    bkgTree->Branch("jet2cvsb", &b_jet2cvsb, "jet2cvsb/F");
    bkgTree->Branch("jet2Idx" , &b_jet2Idx , "jet2Idx/I");

    bkgTree->Branch("jet3pt"  , &b_jet3pt  , "jet3pt/F");
    bkgTree->Branch("jet3eta" , &b_jet3eta , "jet3eta/F");
    bkgTree->Branch("jet3phi" , &b_jet3phi , "jet3phi/F");
    bkgTree->Branch("jet3m"   , &b_jet3m   , "jet3m/F");
    bkgTree->Branch("jet3csv" , &b_jet3csv , "jet3csv/F");
    bkgTree->Branch("jet3cvsl", &b_jet3cvsl, "jet3cvsl/F");
    bkgTree->Branch("jet3cvsb", &b_jet3cvsb, "jet3cvsb/F");
    bkgTree->Branch("jet3Idx" , &b_jet3Idx , "jet3Idx/I");

    bkgTree->Branch("jet12pt"  , &b_jet12pt  , "jet12pt/F");
    bkgTree->Branch("jet12eta" , &b_jet12eta , "jet12eta/F");
    bkgTree->Branch("jet12deta", &b_jet12deta, "jet12deta/F");
    bkgTree->Branch("jet12phi" , &b_jet12phi , "jet12phi/F");
    bkgTree->Branch("jet12dphi", &b_jet12dphi, "jet12dphi/F");
    bkgTree->Branch("jet12dR"  , &b_jet12dR  , "jet12dR/F");
    bkgTree->Branch("jet12m"   , &b_jet12m   , "jet12m/F");

    bkgTree->Branch("jet23pt"  , &b_jet23pt  , "jet23pt/F");
    bkgTree->Branch("jet23eta" , &b_jet23eta , "jet23eta/F");
    bkgTree->Branch("jet23deta", &b_jet23deta, "jet23deta/F");
    bkgTree->Branch("jet23phi" , &b_jet23phi , "jet23phi/F");
    bkgTree->Branch("jet23dphi", &b_jet23dphi, "jet23dphi/F");
    bkgTree->Branch("jet23dR"  , &b_jet23dR  , "jet23dR/F");
    bkgTree->Branch("jet23m"   , &b_jet23m   , "jet23m/F");

    bkgTree->Branch("jet31pt"  , &b_jet31pt  , "jet31pt/F");
    bkgTree->Branch("jet31eta" , &b_jet31eta , "jet31eta/F");
    bkgTree->Branch("jet31deta", &b_jet31deta, "jet31deta/F");
    bkgTree->Branch("jet31phi" , &b_jet31phi , "jet31phi/F");
    bkgTree->Branch("jet31dphi", &b_jet31dphi, "jet31dphi/F");
    bkgTree->Branch("jet31dR"  , &b_jet31dR  , "jet31dR/F");
    bkgTree->Branch("jet31m"   , &b_jet31m   , "jet31m/F");

    bkgTree->Branch("lepWpt"  , &b_lepWpt  , "lepWpt/F");
    bkgTree->Branch("lepWeta" , &b_lepWeta , "lepWeta/F");
    bkgTree->Branch("lepWphi" , &b_lepWphi , "lepWphi/F");
    bkgTree->Branch("lepWdphi", &b_lepWdphi, "lepWdphi/F");
    bkgTree->Branch("lepWm"   , &b_lepWm   , "lepWm/F");

    bkgTree->Branch("lepTpt"  , &b_lepTpt  , "lepTpt/F");
    bkgTree->Branch("lepTeta" , &b_lepTeta , "lepTeta/F");
    bkgTree->Branch("lepTdeta", &b_lepTdeta, "lepTdeta/F");
    bkgTree->Branch("lepTphi" , &b_lepTphi , "lepTphi/F");
    bkgTree->Branch("lepTdphi", &b_lepTdphi, "lepTdphi/F");//W and b
    bkgTree->Branch("lepTdR"  , &b_lepTdR  , "lepTdR/F");
    bkgTree->Branch("lepTm"   , &b_lepTm   , "lepTm/F");

    bkgTree->Branch("hadTpt"      , &b_hadTpt      , "hadTpt/F");
    bkgTree->Branch("hadTeta"     , &b_hadTeta     , "hadTeta/F");
    bkgTree->Branch("hadT12_3deta", &b_hadT12_3deta, "hadT12_3deta/F");
    bkgTree->Branch("hadT23_1deta", &b_hadT23_1deta, "hadT23_1deta/F");
    bkgTree->Branch("hadT31_2deta", &b_hadT31_2deta, "hadT31_2deta/F");
    bkgTree->Branch("hadTphi"     , &b_hadTphi     , "hadTphi/F");
    bkgTree->Branch("hadT12_3dphi", &b_hadT12_3dphi, "hadT12_3dphi/F");
    bkgTree->Branch("hadT23_1dphi", &b_hadT23_1dphi, "hadT23_1dphi/F");
    bkgTree->Branch("hadT31_2dphi", &b_hadT31_2dphi, "hadT31_2dphi/F");
    bkgTree->Branch("hadT12_3dR"  , &b_hadT12_3dR  , "hadT12_3dR/F");
    bkgTree->Branch("hadT23_1dR"  , &b_hadT23_1dR  , "hadT23_1dR/F");
    bkgTree->Branch("hadT31_2dR"  , &b_hadT31_2dR  , "hadT31_2dR/F");
    bkgTree->Branch("hadTm"       , &b_hadTm       , "hadTm/F");

    fOutput->Add(bkgTree);

    ////////////////////////////////////////////////////////////////////////
    testTree = new TTree("test_tree","background tree");

    testTree->Branch("nevt"         , &b_nevt         , "nevt/I");
    testTree->Branch("file"         , &b_file         , "file/I");
    testTree->Branch("GoodPV"       , &b_GoodPV       , "GoodPV/I");
    testTree->Branch("EventCategory", &b_EventCategory, "EventCategory/I");
    testTree->Branch("genMatch"     , &b_genMatch     , "genMatch/I");
    testTree->Branch("EventWeight"  , &b_EventWeight  , "EventWeight/F");

    testTree->Branch("njets"     , &b_njets     , "njets/I");
    testTree->Branch("nbjets_m"  , &b_nbjets_m  , "nbjets_m/I");
    testTree->Branch("lepton_pt" , &b_lepton_pt , "lepton_pt/F");
    testTree->Branch("lepton_eta", &b_lepton_eta, "lepton_eta/F");
    testTree->Branch("lepton_phi", &b_lepton_phi, "lepton_phi/F");
    testTree->Branch("MET"       , &b_met       , "MET/F");
    testTree->Branch("MET_phi"   , &b_met_phi   , "MET_phi/F");
    testTree->Branch("lepDPhi"   , &b_lepdphi   , "lepDPhi/F");

    //jet assignment
    testTree->Branch("jet0pt"  , &b_jet0pt  , "jet0pt/F");
    testTree->Branch("jet0eta" , &b_jet0eta , "jet0eta/F");
    testTree->Branch("jet0phi" , &b_jet0phi , "jet0phi/F");
    testTree->Branch("jet0m"   , &b_jet0m   , "jet0m/F");
    testTree->Branch("jet0csv" , &b_jet0csv , "jet0csv/F");
    testTree->Branch("jet0cvsl", &b_jet0cvsl, "jet0cvsl/F");
    testTree->Branch("jet0cvsb", &b_jet0cvsb, "jet0cvsb/F");
    testTree->Branch("jet0Idx" , &b_jet0Idx , "jet0Idx/I");

    testTree->Branch("jet1pt"  , &b_jet1pt  , "jet1pt/F");
    testTree->Branch("jet1eta" , &b_jet1eta , "jet1eta/F");
    testTree->Branch("jet1phi" , &b_jet1phi , "jet1phi/F");
    testTree->Branch("jet1m"   , &b_jet1m   , "jet1m/F");
    testTree->Branch("jet1csv" , &b_jet1csv , "jet1csv/F");
    testTree->Branch("jet1cvsl", &b_jet1cvsl, "jet1cvsl/F");
    testTree->Branch("jet1cvsb", &b_jet1cvsb, "jet1cvsb/F");
    testTree->Branch("jet1Idx" , &b_jet1Idx , "jet1Idx/I");

    testTree->Branch("jet2pt"  , &b_jet2pt  , "jet2pt/F");
    testTree->Branch("jet2eta" , &b_jet2eta , "jet2eta/F");
    testTree->Branch("jet2phi" , &b_jet2phi , "jet2phi/F");
    testTree->Branch("jet2m"   , &b_jet2m   , "jet2m/F");
    testTree->Branch("jet2csv" , &b_jet2csv , "jet2csv/F");
    testTree->Branch("jet2cvsl", &b_jet2cvsl, "jet2cvsl/F");
    testTree->Branch("jet2cvsb", &b_jet2cvsb, "jet2cvsb/F");
    testTree->Branch("jet2Idx" , &b_jet2Idx , "jet2Idx/I");

    testTree->Branch("jet3pt"  , &b_jet3pt  , "jet3pt/F");
    testTree->Branch("jet3eta" , &b_jet3eta , "jet3eta/F");
    testTree->Branch("jet3phi" , &b_jet3phi , "jet3phi/F");
    testTree->Branch("jet3m"   , &b_jet3m   , "jet3m/F");
    testTree->Branch("jet3csv" , &b_jet3csv , "jet3csv/F");
    testTree->Branch("jet3cvsl", &b_jet3cvsl, "jet3cvsl/F");
    testTree->Branch("jet3cvsb", &b_jet3cvsb, "jet3cvsb/F");
    testTree->Branch("jet3Idx" , &b_jet3Idx , "jet3Idx/I");

    testTree->Branch("jet12pt"  , &b_jet12pt  , "jet12pt/F");
    testTree->Branch("jet12eta" , &b_jet12eta , "jet12eta/F");
    testTree->Branch("jet12deta", &b_jet12deta, "jet12deta/F");
    testTree->Branch("jet12phi" , &b_jet12phi , "jet12phi/F");
    testTree->Branch("jet12dphi", &b_jet12dphi, "jet12dphi/F");
    testTree->Branch("jet12dR"  , &b_jet12dR  , "jet12dR/F");
    testTree->Branch("jet12m"   , &b_jet12m   , "jet12m/F");

    testTree->Branch("jet23pt"  , &b_jet23pt  , "jet23pt/F");
    testTree->Branch("jet23eta" , &b_jet23eta , "jet23eta/F");
    testTree->Branch("jet23deta", &b_jet23deta, "jet23deta/F");
    testTree->Branch("jet23phi" , &b_jet23phi , "jet23phi/F");
    testTree->Branch("jet23dphi", &b_jet23dphi, "jet23dphi/F");
    testTree->Branch("jet23dR"  , &b_jet23dR  , "jet23dR/F");
    testTree->Branch("jet23m"   , &b_jet23m   , "jet23m/F");

    testTree->Branch("jet31pt"  , &b_jet31pt  , "jet31pt/F");
    testTree->Branch("jet31eta" , &b_jet31eta , "jet31eta/F");
    testTree->Branch("jet31deta", &b_jet31deta, "jet31deta/F");
    testTree->Branch("jet31phi" , &b_jet31phi , "jet31phi/F");
    testTree->Branch("jet31dphi", &b_jet31dphi, "jet31dphi/F");
    testTree->Branch("jet31dR"  , &b_jet31dR  , "jet31dR/F");
    testTree->Branch("jet31m"   , &b_jet31m   , "jet31m/F");

    testTree->Branch("lepWpt"  , &b_lepWpt  , "lepWpt/F");
    testTree->Branch("lepWeta" , &b_lepWeta , "lepWeta/F");
    testTree->Branch("lepWphi" , &b_lepWphi , "lepWphi/F");
    testTree->Branch("lepWdphi", &b_lepWdphi, "lepWdphi/F");
    testTree->Branch("lepWm"   , &b_lepWm   , "lepWm/F");

    testTree->Branch("lepTpt"  , &b_lepTpt  , "lepTpt/F");
    testTree->Branch("lepTeta" , &b_lepTeta , "lepTeta/F");
    testTree->Branch("lepTdeta", &b_lepTdeta, "lepTdeta/F");
    testTree->Branch("lepTphi" , &b_lepTphi , "lepTphi/F");
    testTree->Branch("lepTdphi", &b_lepTdphi, "lepTdphi/F");//W and b
    testTree->Branch("lepTdR"  , &b_lepTdR  , "lepTdR/F");
    testTree->Branch("lepTm"   , &b_lepTm   , "lepTm/F");

    testTree->Branch("hadTpt"      , &b_hadTpt      , "hadTpt/F");
    testTree->Branch("hadTeta"     , &b_hadTeta     , "hadTeta/F");
    testTree->Branch("hadT12_3deta", &b_hadT12_3deta, "hadT12_3deta/F");
    testTree->Branch("hadT23_1deta", &b_hadT23_1deta, "hadT23_1deta/F");
    testTree->Branch("hadT31_2deta", &b_hadT31_2deta, "hadT31_2deta/F");
    testTree->Branch("hadTphi"     , &b_hadTphi     , "hadTphi/F");
    testTree->Branch("hadT12_3dphi", &b_hadT12_3dphi, "hadT12_3dphi/F");
    testTree->Branch("hadT23_1dphi", &b_hadT23_1dphi, "hadT23_1dphi/F");
    testTree->Branch("hadT31_2dphi", &b_hadT31_2dphi, "hadT31_2dphi/F");
    testTree->Branch("hadT12_3dR"  , &b_hadT12_3dR  , "hadT12_3dR/F");
    testTree->Branch("hadT23_1dR"  , &b_hadT23_1dR  , "hadT23_1dR/F");
    testTree->Branch("hadT31_2dR"  , &b_hadT31_2dR  , "hadT31_2dR/F");
    testTree->Branch("hadTm"       , &b_hadTm       , "hadTm/F");

    fOutput->Add(testTree);
} 

Bool_t makeTuple::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();

  //Ntuple for ST, TT respectively
  bool isTT = false;

  int mode = 999; 
  mode = *channel;

  if( mode > 2) return kTRUE;

  float lep_SF = 1.0;
  if( !option.Contains("Data") ) lep_SF = lepton_SF[0]; //for datam turn this off
  float genweight = *genWeight;
  float puweight = PUWeight[0];
  float jetsf = jet_SF_CSV_30[0];
  float EventWeight = puweight*genweight*lep_SF*jetsf;

  float relIso = *lepton_relIso; 

  //Object selection
  int njets = 0;
  int nbjets_m = 0; 
  int ncjets_m = 0; 

  TLorentzVector metP4;
  double met = *MET;
  double met_phi = *MET_phi;
  double apt = TMath::Abs(met);
  double met_x =  apt*TMath::Cos(met_phi);
  double met_y =  apt*TMath::Sin(met_phi);
  metP4.SetPxPyPzE( met_x, met_y, 0, met);

  TLorentzVector lepton;
  lepton.SetPtEtaPhiE(*lepton_pT, *lepton_eta, *lepton_phi, *lepton_E);

  double transverseM = transverseMass(lepton, metP4);
  double lepDphi = lepton.DeltaPhi(metP4);

  //Selection Option
  bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
  bool makeIso = true;
  bool isIso = *lepton_isIso; 
  if( makeIso && !isIso ) return kTRUE;
  if( !makeIso && isIso ) return kTRUE;

  //Event selection 
  bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.1);
  bool passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);

  if( option.Contains("DataSingleMu") ){
    if( !passmuon ) return kTRUE;//RDMu
    if( passelectron) return kTRUE;//RDMu
  }
  else if( option.Contains("DataSingleEG") ){
    if( !passelectron ) return kTRUE;//RDelec
    if( passmuon ) return kTRUE;//RDelec
  }
  else{
    if( !passmuon && !passelectron ) return kTRUE;
  }

  b_lepton_pt = lepton.Pt(); b_lepton_eta = lepton.Eta(); b_lepton_phi = lepton.Phi();

  vector<size_t> jetIdxs;
  for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

    TLorentzVector jet;
    jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);
    if( !option.Contains("Data") ) jet = jet * jet_JER_Nom[iJet];

    if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
      jetIdxs.push_back(iJet);

      if( jet_CSV[iJet] > 0.8484 ){
        nbjets_m++;
      }
    }
  }
  njets = jetIdxs.size();
  const int nCombi = njets * (njets -1) * (njets -2) *(njets -3) / 2;
  const int nCombi_st = njets *  (njets -1) *(njets -2) / 2;
  if ( isTT && nCombi < 12 ) return kTRUE;
  if ( !isTT && nCombi_st < 3 ) return kTRUE;

  if ( isTT ){
    if ( njets <  4 || nbjets_m < 2 ) return kTRUE;
  }
  else if ( njets <  3 || nbjets_m < 2 ) return kTRUE;

  //cout << nevt << endl;
  if( option.Contains("Data") ) b_EventCategory = -1;
  else if( option.Contains("Hct") || option.Contains("Hut") ) b_EventCategory = 0;
  else if( option.Contains("ttbb") ) b_EventCategory = 1;
  else if( option.Contains("ttbj") ) b_EventCategory = 2;
  else if( option.Contains("ttcc") ) b_EventCategory = 3;
  else if( option.Contains("ttLF") ) b_EventCategory = 4;
  else if( option.Contains("ttother") ) b_EventCategory = 5;
  else if( option.Contains("channel") ) b_EventCategory = 5; //singletop
  else if( option.Contains("zjets") ) b_EventCategory = 6; //DY
  else b_EventCategory = 10;

  /////////////////////////////////////
  // REMARK :: genjet2+3 = W
  //           jet1+2 = W
  ////////////////////////////////////

  //gen particles
  TLorentzVector gen_lep, gen_nu, gen_lepB, gen_hadJ1, gen_hadJ2, gen_hadJ3;
  //leptonic W is okay
  gen_lep.SetPtEtaPhiE(*genlepton_pT, *genlepton_eta, *genlepton_phi, *genlepton_E);
  gen_nu.SetPtEtaPhiE(*gennu_pT, *gennu_eta, *gennu_phi, *gennu_E);

  if( option.Contains("Hct") || option.Contains("Hut") ){
    gen_hadJ2.SetPtEtaPhiE(*addHbjet1_pt, *addHbjet1_eta, *addHbjet1_phi, *addHbjet1_e);
    gen_hadJ3.SetPtEtaPhiE(*addHbjet2_pt, *addHbjet2_eta, *addHbjet2_phi, *addHbjet2_e);
    gen_hadJ1.SetPtEtaPhiE(gencone_gjet_pT[1], gencone_gjet_eta[1], gencone_gjet_phi[1], gencone_gjet_E[1]);
    gen_lepB.SetPtEtaPhiE(gencone_gjet_pT[0], gencone_gjet_eta[0], gencone_gjet_phi[0], gencone_gjet_E[0]);
  }
  else{
    gen_hadJ2.SetPtEtaPhiE(gencone_gjet_pT[2], gencone_gjet_eta[2], gencone_gjet_phi[2], gencone_gjet_E[2]); //fcnc=>addHbjets
    gen_hadJ3.SetPtEtaPhiE(gencone_gjet_pT[3], gencone_gjet_eta[3], gencone_gjet_phi[3], gencone_gjet_E[3]);

    TLorentzVector cand1, cand2;
    cand1.SetPtEtaPhiE(gencone_gjet_pT[0], gencone_gjet_eta[0], gencone_gjet_phi[0], gencone_gjet_E[0]);
    cand2.SetPtEtaPhiE(gencone_gjet_pT[1], gencone_gjet_eta[1], gencone_gjet_phi[1], gencone_gjet_E[1]);

    if ( (172.5 - (cand1+gen_hadJ2+gen_hadJ3).M())*(172.5 - (cand1+gen_hadJ2+gen_hadJ3).M()) + (172.5 - (cand2+gen_nu+gen_lep).M())*(172.5 - (cand2+gen_nu+gen_lep).M()) < (172.5 - (cand2+gen_hadJ2+gen_hadJ3).M())*(172.5 - (cand2+gen_hadJ2+gen_hadJ3).M()) + (172.5 - (cand1+gen_nu+gen_lep).M())*(172.5 - (cand1+gen_nu+gen_lep).M()) ){
      gen_hadJ1 = cand1;
      gen_lepB = cand2;
    }
    else{
      gen_hadJ1 = cand2;
      gen_lepB = cand1;
    }
  }

  b_genHadW = (gen_hadJ2+gen_hadJ3).M();


  //fill trees
  b_EventWeight = EventWeight;
  b_GoodPV = *GoodPV;
  b_nevt = nevt;

  b_njets = njets;
  b_nbjets_m = nbjets_m;

  b_met = met;
  b_met_phi = met_phi;
  b_lepdphi = lepDphi;
  b_transversem = transverseMass(lepton, metP4);


  //jet assignments
  TLorentzVector lepW = lepton + metP4;
  b_lepWpt    = lepW.Pt();
  b_lepWeta   = lepW.Eta();
  b_lepWphi   = lepW.Phi();
  b_lepWdphi  = lepton.DeltaPhi(metP4);
  b_lepWm     = lepW.M();

  //int count = 0;
  TLorentzVector jetP4[4];
  for ( auto ii0 = jetIdxs.begin(); ii0 != jetIdxs.end(); ++ii0 ){
    //if ( jet_CSV[*ii0] < 0.8484 ) continue; //for ttbar reco signal
    jetP4[0].SetPtEtaPhiE(jet_pT[*ii0], jet_eta[*ii0], jet_phi[*ii0], jet_E[*ii0]);

    for ( auto ii1 = jetIdxs.begin(); ii1 != jetIdxs.end(); ++ii1 ) {
      if ( *ii1 == *ii0 ) continue;
      //if ( jet_CSV[*ii1] < 0.8484 ) continue; //for ttbar reco signal
      if ( isTT ) jetP4[3].SetPtEtaPhiE(jet_pT[*ii1], jet_eta[*ii1], jet_phi[*ii1], jet_E[*ii1]);

      for ( auto ii2 = jetIdxs.begin(); ii2 != jetIdxs.end(); ++ii2 ) {
        if ( *ii2 == *ii0 ) continue;
        if ( isTT && *ii2 == *ii1) continue;
        if ( jet_CSV[*ii2] < 0.8484 ) continue;
        jetP4[2].SetPtEtaPhiE(jet_pT[*ii2], jet_eta[*ii2], jet_phi[*ii2], jet_E[*ii2]);

        for ( auto ii3 = ii2+1; ii3 != jetIdxs.end(); ++ii3 ) {
          if ( *ii3 == *ii0 or *ii3 == *ii1 or *ii3 == *ii2 ) continue;
          if ( jet_CSV[*ii3] < 0.8484 ) continue;
          jetP4[1].SetPtEtaPhiE(jet_pT[*ii3], jet_eta[*ii3], jet_phi[*ii3], jet_E[*ii3]);
          //count++;

          //construct particles: lepB = j0, hadB = j3, hadW = j1+j2
          if( !option.Contains("Data") ){
            jetP4[0] = jetP4[0] * jet_JER_Nom[*ii0];
            jetP4[1] = jetP4[1] * jet_JER_Nom[*ii3];
            jetP4[2] = jetP4[2] * jet_JER_Nom[*ii2];
            jetP4[3] = jetP4[3] * jet_JER_Nom[*ii1];
          }
          b_jet0pt = jetP4[0].Pt(); b_jet0eta = jetP4[0].Eta(); b_jet0phi = jetP4[0].Phi(); b_jet0m = jetP4[0].M();
          b_jet1pt = jetP4[1].Pt(); b_jet1eta = jetP4[1].Eta(); b_jet1phi = jetP4[1].Phi(); b_jet1m = jetP4[1].M();
          b_jet2pt = jetP4[2].Pt(); b_jet2eta = jetP4[2].Eta(); b_jet2phi = jetP4[2].Phi(); b_jet2m = jetP4[2].M();
          b_jet3pt = jetP4[3].Pt(); b_jet3eta = jetP4[3].Eta(); b_jet3phi = jetP4[3].Phi(); b_jet3m = jetP4[3].M();
          b_jet0csv = jet_CSV[*ii0]; b_jet0cvsl = jet_CvsL[*ii0]; b_jet0cvsb = jet_CvsB[*ii0];
          b_jet1csv = jet_CSV[*ii3]; b_jet1cvsl = jet_CvsL[*ii3]; b_jet1cvsb = jet_CvsB[*ii3];
          b_jet2csv = jet_CSV[*ii2]; b_jet2cvsl = jet_CvsL[*ii2]; b_jet2cvsb = jet_CvsB[*ii2];
          b_jet3csv = jet_CSV[*ii1]; b_jet3cvsl = jet_CvsL[*ii1]; b_jet3cvsb = jet_CvsB[*ii1];//ii3 = jet1
          b_jet0Idx = *ii0; b_jet1Idx = *ii3; b_jet2Idx = *ii2; b_jet3Idx = *ii1;

          const auto lepT = lepW + jetP4[0];
          const auto had12 = jetP4[1] + jetP4[2];//This is W or H
          const auto had23 = jetP4[2] + jetP4[3]; 
          const auto had31 = jetP4[3] + jetP4[1];
          const auto hadT = jetP4[1] + jetP4[2] + jetP4[3];

          b_jet12pt = had12.Pt(); b_jet12eta = had12.Eta(); b_jet12phi = had12.Phi(); b_jet12m = had12.M();
          b_jet23pt = had23.Pt(); b_jet23eta = had23.Eta(); b_jet23phi = had23.Phi(); b_jet23m = had23.M();
          b_jet31pt = had31.Pt(); b_jet31eta = had31.Eta(); b_jet31phi = had31.Phi(); b_jet31m = had31.M();
          b_jet12deta = (jetP4[1]-jetP4[2]).Eta(); b_jet12dphi = jetP4[1].DeltaPhi(jetP4[2]);
          b_jet23deta = (jetP4[2]-jetP4[3]).Eta(); b_jet23dphi = jetP4[2].DeltaPhi(jetP4[3]);
          b_jet31deta = (jetP4[3]-jetP4[1]).Eta(); b_jet31dphi = jetP4[3].DeltaPhi(jetP4[1]);
          b_jet12dR = jetP4[1].DeltaR(jetP4[2]); b_jet23dR = jetP4[2].DeltaR(jetP4[3]); b_jet31dR = jetP4[3].DeltaR(jetP4[1]);

          b_lepTpt = lepT.Pt(); b_lepTeta = lepT.Eta(); b_lepTphi = lepT.Phi(); b_lepTm = lepT.M();
          b_lepTdeta = (lepW-jetP4[0]).Eta(); b_lepTdphi = lepW.DeltaPhi(jetP4[0]); b_lepTdR = lepW.DeltaPhi(jetP4[0]);

          b_hadTpt = hadT.Pt(); b_hadTeta = hadT.Eta(); b_hadTphi = hadT.Phi(); b_hadTm = hadT.M();
          b_hadT12_3deta = (had12-jetP4[3]).Eta(); b_hadT23_1deta = (had23-jetP4[1]).Eta(); b_hadT31_2deta = (had31-jetP4[2]).Eta();
          b_hadT12_3dphi = had12.DeltaPhi(jetP4[3]); b_hadT23_1dphi = had23.DeltaPhi(jetP4[1]); b_hadT31_2dphi = had31.DeltaPhi(jetP4[2]);
          b_hadT12_3dR = had12.DeltaR(jetP4[3]); b_hadT23_1dR = had23.DeltaR(jetP4[1]); b_hadT31_2dR = had31.DeltaR(jetP4[2]);

          //some genmatch here
          b_genMatch = 0; // [lep][nu][lepB][hadJ3][hadJ2][hadJ1]
          //if ( gen_lep.Pt()   > 0 and gen_lep.DeltaR(lepton)    < 0.1 ) b_genMatch += 100000;
          //if ( gen_nu.Pt()    > 0 and gen_nu.DeltaPhi(metP4)      < 0.1 ) b_genMatch += 10000;
          if ( gen_lepB.Pt()  > 0 and gen_lepB.DeltaR(jetP4[0])   < 0.4 ) b_genMatch += 1000;
          if ( isTT ){
            if ( gen_hadJ1.Pt()  > 0 and gen_hadJ1.DeltaR(jetP4[3])  < 0.4 ) b_genMatch += 100;
          }
          if ( gen_hadJ2.Pt() > 0 and (gen_hadJ2.DeltaR(jetP4[1]) < 0.4 or gen_hadJ2.DeltaR(jetP4[2]) < 0.4) ) b_genMatch += 10;
          if ( gen_hadJ3.Pt() > 0 and (gen_hadJ3.DeltaR(jetP4[1]) < 0.4 or gen_hadJ3.DeltaR(jetP4[2]) < 0.4) ) b_genMatch += 1;

          testTree->Fill();

          if ( isTT ){
            if ( b_genMatch == 1111 ) sigTree->Fill();
            else bkgTree->Fill();
          }
          else{
            if ( b_genMatch == 1011 ) sigTree->Fill();
            else bkgTree->Fill();
          }
          //else continue;

        }
      }
    }
  }

  //cout << nCombi << " real count is " << count << endl;

  nevt++;

  //if ( gen_lepB.Pt() > 0 and gen_hadJ1.Pt() > 0 and gen_hadJ2.Pt() > 0 and gen_hadJ3.Pt() > 0 ) sigTree->Fill();

  //sigTree->Fill();
  //bkgTree->Fill();

  return kTRUE;
}

void makeTuple::SlaveTerminate()
{
  TString option = GetOption();
}
   

void makeTuple::Terminate()
{
  TString option = GetOption();

    //TFile *hfile = TFile::Open(Form("root://cms-xrdr.sdfarm.kr:1094//xrd/store/user/minerva1993/reco/ntuple/deepReco_%s.root",option.Data()), "RECREATE");
    //TFile *hfile = TFile::Open(Form("j4b2/deepReco_%s.root",option.Data()), "RECREATE"); //FCNC
    TFile *hfile = TFile::Open(Form("j3b2_st/deepReco_%s.root",option.Data()), "RECREATE"); //ttbar

    fOutput->FindObject("sig_tree")->Write();
    fOutput->FindObject("bkg_tree")->Write();
    fOutput->FindObject("test_tree")->Write();

    hfile->Write();
    hfile->Close();

}

double makeTuple::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}
