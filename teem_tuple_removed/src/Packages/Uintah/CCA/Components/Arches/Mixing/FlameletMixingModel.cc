//----- FlameletMixingModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/FlameletMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/KDTree.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/VectorTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/StanjanEquilibriumReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ILDMReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <iostream>
#include <math.h>
#include <Core/Math/MiscMath.h>
#include <iomanip>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
//****************************************************************************
// Default constructor for FlameletMixingModel
//****************************************************************************
FlameletMixingModel::FlameletMixingModel():MixingModel()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
FlameletMixingModel::~FlameletMixingModel()
{
}

//****************************************************************************
// Problem Setup for FlameletMixingModel
//****************************************************************************
void 
FlameletMixingModel::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("FlameletMixingModel");
  db->require("adiabatic",d_adiabatic);
  db->require("rxnvars",d_numRxnVars);
  db->require("mixstatvars",d_numMixStatVars);
  // Set up for MixingModel table
  d_numMixingVars = 1;
  int numExtraVar = 1; // axial location 
  // Define mixing table, which includes call reaction model constructor
  d_tableDimension = d_numMixingVars + d_numRxnVars + d_numMixStatVars + !(d_adiabatic)
                     + numExtraVar;
  readFlamelet();

}

      

//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
FlameletMixingModel::computeProps(const InletStream& inStream,
			     Stream& outStream)
{
  // convert inStream to array
  double mixFrac = inStream.d_mixVars[0];
  double mixFracVars = 0.0;
  if (inStream.d_mixVarVariance.size() != 0)
    mixFracVars = inStream.d_mixVarVariance[0];
  int axialLoc = inStream.d_axialLoc;
  tableLookUp(mixFrac, mixFracVars, axialLoc, outStream);  
 
 
}


void
FlameletMixingModel::tableLookUp(double mixfrac, double mixfracVars, int axial_loc,
				 Stream& outStream) {

  // compute index
  // nx - mixfrac, ny - mixfracVars, nz - axialLoc
  // assuming uniform in mixfrac direction
  double min_Z = 0.0;
  double max_Z = 1.0;
  int nx_lo, nx_hi;
  double w_nxlo, w_nxhi;
  double incrZ = (max_Z-min_Z)/(d_numMixfrac-1);
  nx_lo = (floor)((mixfrac-min_Z)/incrZ);
  if (nx_lo < 0) 
    nx_lo = 0;
  if (nx_lo > d_numMixfrac-2)
    nx_lo = d_numMixfrac-2;
  nx_hi = nx_lo + 1;
  w_nxlo = (meanMix[nx_hi]-mixfrac)/(meanMix[nx_hi]-meanMix[nx_lo]);
  if (w_nxlo < 0.0)
    w_nxlo = 0.0;
  if (w_nxlo > 1.0)
    w_nxlo = 1.0;
  w_nxhi = 1.0 - w_nxlo;
  
  // variance is a nonuniform table
  double min_Zvar = 0.0;
  double max_Zvar = meanVars[d_numMixvars-1];
  double curr_Zvar = 0.0;
  int ii;
  for (ii = 0; ii < d_numMixvars-1; ii++) {
    if (mixfracVars < meanVars[ii]) {
      ii--;
      break;
    }
  }
  if (ii < 0)
    ii = 0;
  if (ii > d_numMixvars-2)
    ii = d_numMixvars-2;
  int ny_lo = ii;
  int ny_hi = ii+1;
  double w_nylo = (meanVars[ny_hi]-mixfracVars)/(meanVars[ny_hi]-meanVars[ny_lo]);
  if (w_nylo < 0)
    w_nylo = 0.0;
  if (w_nylo > 1.0)
    w_nylo = 1.0;
  double w_nyhi = 1.0 - w_nylo;
  
  // linear interpolation
  int indexlo11 = nx_lo+ny_lo*d_numMixfrac+axial_loc*d_numMixvars*d_numMixfrac;
  int indexlo12 = nx_hi+ny_lo*d_numMixfrac+axial_loc*d_numMixvars*d_numMixfrac;
  int indexhi21 = nx_lo+ny_hi*d_numMixfrac+axial_loc*d_numMixvars*d_numMixfrac;
  int indexhi22 = nx_hi+ny_hi*d_numMixfrac+axial_loc*d_numMixvars*d_numMixfrac;
  outStream.d_density = w_nylo*(w_nxlo*table[0][indexlo11]+
				w_nxhi*table[0][indexlo12])+
                        w_nyhi*(w_nxlo*table[0][indexhi21]+
				w_nxhi*table[0][indexhi22]);
  outStream.d_temperature = w_nylo*(w_nxlo*table[1][indexlo11]+
				w_nxhi*table[1][indexlo12])+
                        w_nyhi*(w_nxlo*table[1][indexhi21]+
				w_nxhi*table[1][indexhi22]);
  outStream.d_sootFV = w_nylo*(w_nxlo*table[2][indexlo11]+
				w_nxhi*table[2][indexlo12])+
                        w_nyhi*(w_nxlo*table[2][indexhi21]+
				w_nxhi*table[2][indexhi22]);
  outStream.d_co2 = w_nylo*(w_nxlo*table[3][indexlo11]+
				w_nxhi*table[3][indexlo12])+
                        w_nyhi*(w_nxlo*table[3][indexhi21]+
				w_nxhi*table[3][indexhi22]);
  outStream.d_h2o = w_nylo*(w_nxlo*table[4][indexlo11]+
				w_nxhi*table[4][indexlo12])+
                        w_nyhi*(w_nxlo*table[4][indexhi21]+
				w_nxhi*table[4][indexhi22]);
  outStream.d_fvtfive = w_nylo*(w_nxlo*table[5][indexlo11]+
				w_nxhi*table[5][indexlo12])+
                        w_nyhi*(w_nxlo*table[5][indexhi21]+
				w_nxhi*table[5][indexhi22]);
  outStream.d_tfour = w_nylo*(w_nxlo*table[6][indexlo11]+
				w_nxhi*table[6][indexlo12])+
                        w_nyhi*(w_nxlo*table[6][indexhi21]+
				w_nxhi*table[6][indexhi22]);
  outStream.d_tfive = w_nylo*(w_nxlo*table[7][indexlo11]+
				w_nxhi*table[7][indexlo12])+
                        w_nyhi*(w_nxlo*table[7][indexhi21]+
				w_nxhi*table[7][indexhi22]);
  outStream.d_tnine = w_nylo*(w_nxlo*table[8][indexlo11]+
				w_nxhi*table[8][indexlo12])+
                        w_nyhi*(w_nxlo*table[8][indexhi21]+
				w_nxhi*table[8][indexhi22]);
  outStream.d_qrg = w_nylo*(w_nxlo*table[9][indexlo11]+
				w_nxhi*table[9][indexlo12])+
                        w_nyhi*(w_nxlo*table[9][indexhi21]+
				w_nxhi*table[9][indexhi22]);
  outStream.d_qrs = w_nylo*(w_nxlo*table[10][indexlo11]+
				w_nxhi*table[10][indexlo12])+
                        w_nyhi*(w_nxlo*table[10][indexhi21]+
				w_nxhi*table[10][indexhi22]);

}

void
FlameletMixingModel::readFlamelet()
{
  string inputfile1;
  inputfile1 = "chemtable"; // contains information about mixture fraction
  int count = 1;
  cerr << inputfile1 << endl;
  ifstream fd("chemtable", ios::in);
  fd >> d_numMixfrac >> d_numMixvars >>d_numAxialLocs >> d_numVars;
  cerr << d_numMixfrac << " " << d_numMixvars << " " << d_numAxialLocs << " " 
       << d_numVars << endl;
  meanMix = vector<double>(d_numMixfrac);
  meanVars = vector<double>(d_numMixvars);
  meanAxialLocs = vector<int>(d_numAxialLocs);
  for (int ii = 0; ii < meanMix.size(); ii++) {
    fd >> meanMix[ii];
  }
  for (int ii = 0; ii < meanVars.size(); ii++) {
    fd >> meanVars[ii];
  }
  for (int ii = 0; ii < meanAxialLocs.size(); ii++) {
    fd >> meanAxialLocs[ii];
  }
  tags = vector<string> (d_numVars);
  int size = d_numMixfrac*d_numMixvars*d_numAxialLocs;
  table = vector <vector <double> > (d_numVars);
  for (int ii = 0; ii < d_numVars; ii++)
    table[ii] = vector <double> (size);
  for (int ii = 0; ii < d_numVars; ii++) {
    fd >> tags[ii];
    cerr << tags[ii] << endl;
    for (int index = 0; index < size; index++) {
      fd >> table[ii][index];
      if ((table[ii][index]< -99)&&(ii < 7)){
	table[ii][index-1] = 0.0;
	index --;
      }
    }
  }
  fd.close();
  //  outfile.close();
}





