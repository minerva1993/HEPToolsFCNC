#include "ttbarljetsfitter.h"

TMinuit *tm=0;

TLorentzVector tmplep;
TLorentzVector tmpnu;
TLorentzVector tmpbl;
TLorentzVector tmpbj;
TLorentzVector tmpj1;
TLorentzVector tmpj2;

double blres;
double bjres;
double j1res;
double j2res;
double metres;

double jetEresolution(double energy) // relative jet energy resolution
{
    return TMath::Sqrt(TMath::Power(0.05*energy,2.0) + TMath::Power(1.5*sqrt(energy),2.0))/energy;
}

// met phi resolution as a function of reconstructed met from Delphes
double metphiresolution(double met)
{
    return 0.05539 - 0.5183*exp(-0.01507*met);
}

double metresolution(double met)
{
    return 15.0;
}

double twoobjectmassresolution(TLorentzVector &j1, double releres1, TLorentzVector &j2, double releres2)
{
    // crude, but OK
    double massnominal = (j1+j2).M();
    TLorentzVector j1smeared = (1.0+releres1)*j1;
    TLorentzVector j2smeared = (1.0+releres2)*j2;

    double deltamass1up = (j1smeared+j2).M()-massnominal;
    double deltamass2up = (j1+j2smeared).M()-massnominal;
    return TMath::Hypot(deltamass1up, deltamass2up)/massnominal;
}

double twojetmassresolution(TLorentzVector &j1, TLorentzVector &j2) // relative mass resolution
{
    double releres1 = jetEresolution(j1.E());
    double releres2 = jetEresolution(j2.E());
   
    return twoobjectmassresolution(j1, releres1, j2, releres2);
}


void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{

//calculate chisquare 
    tmpnu.SetPz(par[0]);
    double massres=twoobjectmassresolution(tmplep, 0.0, tmpnu, 15.0/tmpnu.Pt())*80.4;
    f = TMath::Power(((tmpnu+tmplep).M()-80.4)/massres, 2.0);
}

// full solution
// par[0] - neutrino Pz
// par[1] - missing Et scale factor 
// par[2] - b-jet energy scale factor (leptonic side)
// par[3] - b-jet energy scale factor (hadronic side)
// par[4] - jet energy scale factor for jet 1
// par[5] - jet energy scale factor for jet 2
void fcnfull(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{

    TLorentzVector nuscaled = tmpnu*par[1];
    nuscaled.SetPz(par[0]);
    TLorentzVector blscaled = tmpbl*par[2];
    TLorentzVector bjscaled = tmpbj*par[3];
    TLorentzVector j1scaled = tmpj1*par[4];
    TLorentzVector j2scaled = tmpj2*par[5];

//calculate chisquare 
    f = TMath::Power(((nuscaled+tmplep).M()-80.4)/2.085, 2.0)
     + TMath::Power(((blscaled + nuscaled+tmplep).M()-172.0)/1.5, 2.0)
     + TMath::Power(((j1scaled+j2scaled).M()-80.4)/2.085, 2.0)
     + TMath::Power(((bjscaled + j1scaled+j2scaled).M()-172.0)/1.5, 2.0)
     + TMath::Power((par[1]-1.0)/metres, 2.0)
     + TMath::Power((par[2]-1.0)/blres, 2.0)
     + TMath::Power((par[3]-1.0)/bjres, 2.0)
     + TMath::Power((par[4]-1.0)/j1res, 2.0)
     + TMath::Power((par[5]-1.0)/j2res, 2.0);
}


void initminuit()
{
    tm= new TMinuit(6);
    tm->SetFCN(fcnfull);
    Double_t arglist[10];
    Int_t ierflg = 0;

    arglist[0] = 1;
    gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);


}


double solvettbarljets(double &nupz, double &metscale, double &blscale, double &bjscale, double &j1scale, double &j2scale)
{
    if (tm==0) initminuit();

// Set starting values and step sizes for parameters
    static Double_t vstart[6] = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    static Double_t step[6] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    int ierflg;

    tm->mnparm(0, "nupz", vstart[0], step[0], 0,0,ierflg);
    tm->mnparm(1, "metsf", vstart[1], step[1], 0,0,ierflg);
    tm->mnparm(2, "bjetlsf", vstart[2], step[2], 0,0,ierflg);
    tm->mnparm(3, "bjethsf", vstart[3], step[3], 0,0,ierflg);
    tm->mnparm(4, "wjet1", vstart[4], step[4], 0,0,ierflg);
    tm->mnparm(5, "wjet2", vstart[5], step[5], 0,0,ierflg);

// Now ready for minimization step
    double arglist[10];
    arglist[0] = 500;
    arglist[1] = 1.;
    tm->mnexcm("MIGRAD", arglist ,2,ierflg);

// Print results
    Double_t amin,edm,errdef;
    Int_t nvpar,nparx,icstat;
    double err;
    tm->mnstat(amin,edm,errdef,nvpar,nparx,icstat);
    tm->GetParameter(0, nupz, err);
    tm->GetParameter(1, metscale, err);
    tm->GetParameter(2, blscale, err);
    tm->GetParameter(3, bjscale, err);
    tm->GetParameter(4, j1scale, err);
    tm->GetParameter(5, j2scale, err);

    return amin;
}


void findhadronictop(TLorentzVector &lepton, vector<Jet *> &jets, MissingET *met, bool usebtaginfo, vector<int> &bestindices, float &bestchi2, TLorentzVector &nusol, TLorentzVector &blrefit, TLorentzVector &bjrefit, TLorentzVector &j1refit, TLorentzVector &j2refit)
{
    int njets = jets.size();

    TLorentzVector trialW;
    TLorentzVector trialb;
    TLorentzVector trialtop;
    TLorentzVector trialWjet1;
    TLorentzVector trialWjet2;
    TLorentzVector trialblepton;
    TLorentzVector trialtoplepton;
    TLorentzVector trialwlepton;
    // re-sort

    int bestidx1=-1;
    int bestidx2=-1;
    int bestidx3=-1;
    int bestidx4=-1;

    bestchi2=1.0e6;
    float chi2;

    bestindices[0]=-1;
    bestindices[1]=-1;
    bestindices[2]=-1;
    bestindices[3]=-1;

    nusol.SetPtEtaPhiM(met->MET, 0.0, met->Phi, 0.0);
    metres = metresolution(nusol.Pt())/nusol.Pt();
    double wlmassrelres=twoobjectmassresolution(lepton, 0.0, nusol, 15.0/nusol.Pt());

    if (!usebtaginfo) // not using b-tagging information
    {
        if (njets>=4) // at least there should be 4 hadronic jets
        {
            for (int i1=0; i1<njets; i1++)
            {
                for (int i2=0; i2<njets-1 ; i2++)
                {
                    for (int i3=i2+1; i3<njets; i3++)
                    {
                        for (int i4=0; i4<njets; i4++)
                        {
                            if (i2 != i1 && i3 != i1 && i4!=i1 && i4 != i2 && i4 != i3)
                            {
                                trialb = jets[i1]->P4();
                                trialWjet1 = jets[i2]->P4();
                                trialWjet2 = jets[i3]->P4();
                                trialW = trialWjet1 + trialWjet2;
                                trialtop = trialb + trialW;
                                trialblepton = jets[i4]->P4();
                                trialtoplepton = trialwlepton + trialblepton;

                                // set global variables - ugly!
                                tmplep = lepton;
                                tmpnu = nusol;
                                tmpbl = trialblepton;
                                tmpbj = trialb;
                                tmpj1 = trialWjet1;
                                tmpj2 = trialWjet2;

                                blres = jetEresolution(tmpbl.E());
                                bjres = jetEresolution(tmpbj.E());
                                j1res = jetEresolution(tmpj1.E());
                                j2res = jetEresolution(tmpj2.E());


                                double nupz, metscale, blscale, bjscale, j1scale, j2scale;

                                // dynamic resolutions
                                chi2 = solvettbarljets(nupz, metscale, blscale, bjscale, j1scale, j2scale);

                                if (chi2 < bestchi2)
                                {
                                    bestchi2 = chi2;
                                    nusol = tmpnu*metscale;
                                    nusol.SetPz(nupz);
                                    blrefit = tmpbl*blscale;
                                    bjrefit = tmpbj*bjscale;
                                    j1refit = tmpj1*j1scale;
                                    j2refit = tmpj2*j2scale;
                                    bestidx1 = i1;
                                    bestidx2 = i2;
                                    bestidx3 = i3;
                                    bestidx4 = i4;

                                }
                            }
                        }
                    }
                }
            }
            bestindices[0]=bestidx1; // b for hadronic side
            bestindices[1]=bestidx2; // W jet
            bestindices[2]=bestidx3; // W jet
            bestindices[3]=bestidx4; // b for leptonic side
        }
    }
    else // use b-tagging information
    {
        if (njets>=4) // at least there should be 4 hadronic jets
        {
            int nbjets=0;
            for (int i1=0; i1<njets; i1++)
            {
                if (jets[i1]->BTag==1) nbjets++;
            }

            for (int i1=0; i1<njets; i1++)
            {
                for (int i2=0; i2<njets-1 ; i2++)
                {
                    for (int i3=i2+1; i3<njets; i3++)
                    {
                        for (int i4=0; i4<njets; i4++)
                        {
                            if (i2 != i1 && i3 != i1 && i4!=i1 && i4 != i2 && i4 != i3
                                && ((lepton + jets[i4]->P4()).M() < 170.0)
                                && ( nbjets==0 
                                     ||  (nbjets==1 && (jets[i2]->BTag==0 && jets[i3]->BTag==0))
                                     ||  (nbjets==2 && (jets[i2]->BTag==0 && jets[i3]->BTag==0))
                                     ||  (nbjets==3 && (jets[i2]->BTag==0 || jets[i3]->BTag==0))
                                     ||  (nbjets>3 && (jets[i2]->BTag==1 || jets[i3]->BTag==1))
                                   )
                            )
                            {
                                trialb = jets[i1]->P4();
                                trialWjet1 = jets[i2]->P4();
                                trialWjet2 = jets[i3]->P4();
                                trialW = trialWjet1 + trialWjet2;
                                trialtop = trialb + trialW;
                                trialblepton = jets[i4]->P4();
                                trialtoplepton = trialwlepton + trialblepton;
                                double bjetreleres = jetEresolution(trialb.E());

                                // set global variables - ugly!
                                tmplep = lepton;
                                tmpnu = nusol;
                                tmpbl = trialblepton;
                                tmpbj = trialb;
                                tmpj1 = trialWjet1;
                                tmpj2 = trialWjet2;

                                blres = jetEresolution(tmpbl.E());
                                bjres = jetEresolution(tmpbj.E());
                                j1res = jetEresolution(tmpj1.E());
                                j2res = jetEresolution(tmpj2.E());


                                double nupz, metscale, blscale, bjscale, j1scale, j2scale;

                                // dynamic resolutions
                                chi2 = solvettbarljets(nupz, metscale, blscale, bjscale, j1scale, j2scale);

                                if (chi2 < bestchi2)
                                {
                                    bestchi2 = chi2;
                                    nusol = tmpnu*metscale;
                                    nusol.SetPz(nupz);
                                    blrefit = tmpbl*blscale;
                                    bjrefit = tmpbj*bjscale;
                                    j1refit = tmpj1*j1scale;
                                    j2refit = tmpj2*j2scale;
                                    bestidx1 = i1;
                                    bestidx2 = i2;
                                    bestidx3 = i3;
                                    bestidx4 = i4;

                                }
                            }
                        }
                    }
                }
            }
            bestindices[0]=bestidx1; // b for hadronic side
            bestindices[1]=bestidx2; // W jet
            bestindices[2]=bestidx3; // W jet
            bestindices[3]=bestidx4; // b for leptonic side
        }
    }


}
