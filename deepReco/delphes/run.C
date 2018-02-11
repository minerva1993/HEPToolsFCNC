#include "analyzer.h"
R__LOAD_LIBRARY(analyzer.C+)

void run()
{
   gSystem->CompileMacro("analyzer.C");
   TChain chain("tree");
   chain.Add("flat_ttLJ.root");

   analyzer t(&chain);
   t.Loop("ntuple_ttLJ.root");
}
