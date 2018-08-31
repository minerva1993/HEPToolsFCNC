#include "TColor.h"

void getHexCol(){

TColor *ttbb = gROOT->GetColor(kRed+4);
cout << "ttbb" << " " << ttbb->TColor::AsHexString() << endl;

TColor *ttbj = gROOT->GetColor(kRed+3);
cout << "ttbj" << " " << ttbj->TColor::AsHexString() << endl;

TColor *ttcc = gROOT->GetColor(kRed+2);
cout << "ttcc" << " " << ttcc->TColor::AsHexString() << endl;

TColor *ttlf = gROOT->GetColor(kRed);
cout << "ttlf" << " " << ttlf->TColor::AsHexString() << endl;

TColor *ttbkg = gROOT->GetColor(kRed-7);
cout << "ttbkg" << " " << ttbkg->TColor::AsHexString() << endl;

TColor *ttv = gROOT->GetColor(kMagenta-7);
cout << "ttv" << " " << ttv->TColor::AsHexString() << endl;

TColor *wj = gROOT->GetColor(kYellow);
cout << "wjet" << " " << wj->TColor::AsHexString() << endl;

TColor *zj = gROOT->GetColor(kBlue);
cout << "zjet" << " " << zj->TColor::AsHexString() << endl;

TColor *st = gROOT->GetColor(6);
cout << "st" << " " << st->TColor::AsHexString() << endl;

TColor *vv = gROOT->GetColor(kCyan);
cout << "vv" << " " << vv->TColor::AsHexString() << endl;

TColor *tthct = gROOT->GetColor(433);
cout << "tthct" << " " << tthct->TColor::AsHexString() << endl;

TColor *tthut = gROOT->GetColor(401);
cout << "tthut" << " " << tthut->TColor::AsHexString() << endl;

TColor *sthct = gROOT->GetColor(435);
cout << "sthct" << " " << sthct->TColor::AsHexString() << endl;

TColor *sthut = gROOT->GetColor(403);
cout << "sthut" << " " << sthut->TColor::AsHexString() << endl;
}
