#include "TLorentzVector.h"
#include "TMinuit.h"
#include "classes/DelphesClasses.h"

double jetEresolution(double energy);
double metphiresolution(double met);
double metresolution(double met);
double twoobjectmassresolution(TLorentzVector &j1, double releres1, TLorentzVector &j2, double releres2);
double twojetmassresolution(TLorentzVector &j1, TLorentzVector &j2);// relative mass resolution
double solvettbarljets(double &nupz, double &metscale, double &blscale, double &bjscale, double &j1scale, double &j2scale);
void findhadronictop(TLorentzVector &lepton, vector<Jet *> &jets, MissingET *met, bool usebtaginfo, vector<int> &bestindices, float &bestchi2, TLorentzVector &nusol, TLorentzVector &blrefit, TLorentzVector &bjrefit, TLorentzVector &j1refit, TLorentzVector &j2refit);
