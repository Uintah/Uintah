
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>

using namespace Uintah;
using namespace std;

InletStream::InletStream()
{
  d_axialLoc = 0;
  d_enthalpy = 0;
}

InletStream::InletStream(int numMixVars, int numMixStatVars, int numRxnVars)
{
  d_mixVars = vector<double>(numMixVars);
  d_mixVarVariance = vector<double>(numMixStatVars);
  d_rxnVars = vector<double>(numRxnVars);
  d_axialLoc = 0;
  d_enthalpy = 0;
  d_initEnthalpy = false;
  //???What about enthalpy???
}

InletStream::~InletStream()
{
}

//
//
