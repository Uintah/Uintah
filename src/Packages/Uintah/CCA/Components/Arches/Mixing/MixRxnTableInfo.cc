//----- MixRxnTableInfo.cc --------------------------------------------------


#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <iostream>
using namespace std;
using namespace Uintah;

//****************************************************************************
// Default constructor for MixRxnTableInfo
//****************************************************************************
MixRxnTableInfo::MixRxnTableInfo(int numTableDim) {
  d_numTableDim = numTableDim;
  d_numDivisions = new int[numTableDim];
  d_maxValue = new double[numTableDim];
  d_minValue = new double[numTableDim];
  d_incrValue = new double[numTableDim];
  d_stoicValue = new double[numTableDim];
      

}

MixRxnTableInfo::~MixRxnTableInfo() {
  delete [] d_numDivisions;
  delete [] d_maxValue;
  delete [] d_minValue;
  delete [] d_incrValue;
  delete [] d_stoicValue;

}
// Problem Setup for MixRxnTableInfo
//****************************************************************************
void 
MixRxnTableInfo::problemSetup(const ProblemSpecP& params, bool mixTableFlag,
			      const PDFMixingModel* mixModel)
{
  ProblemSpecP db = params->findBlock("MixRxnTableInfo");
  int count = 0;
  if (!(mixModel->isAdiabatic())) { // if not adiabatic
    ProblemSpecP enth_db = db->findBlock("Enthalpy");
    enth_db->require("NumDivs",d_numDivisions[count]);
    enth_db->require("MaxValue",d_maxValue[count]);
    enth_db->require("MinValue",d_minValue[count]);
    enth_db->require("MidValue",d_stoicValue[count]);
  count ++;
  }
  if (mixModel->getNumMixVars()) {
    for (ProblemSpecP mixfrac_db = db->findBlock("MixtureFraction");
	 mixfrac_db != 0; mixfrac_db = db->findNextBlock("MixtureFraction")) {
      mixfrac_db->require("NumDivs",d_numDivisions[count]);
      mixfrac_db->require("MaxValue",d_maxValue[count]);
      mixfrac_db->require("MinValue",d_minValue[count]);
      mixfrac_db->require("MidValue",d_stoicValue[count]);
    count ++;
    }
  }
 
  if (mixTableFlag) {
    if (mixModel->getNumMixStatVars()) {
      for (ProblemSpecP mixvar_db = db->findBlock("MixFracVariance");
	   mixvar_db != 0; mixvar_db = db->findNextBlock("MixFracVariance")) {
	mixvar_db->require("NumDivs",d_numDivisions[count]);
	mixvar_db->require("MaxValue",d_maxValue[count]);
	mixvar_db->require("MinValue",d_minValue[count]);
	mixvar_db->require("MidValue",d_stoicValue[count]);
	count ++;
      }
    }
  }

  if (mixModel->getNumRxnVars()) {
    for (ProblemSpecP rxnvar_db = db->findBlock("RxnVars");
	 rxnvar_db != 0; rxnvar_db = db->findNextBlock("RxnVars")) {
      rxnvar_db->require("NumDivs",d_numDivisions[count]);
      rxnvar_db->require("MaxValue",d_maxValue[count]);
      rxnvar_db->require("MinValue",d_minValue[count]);
      rxnvar_db->require("MidValue",d_stoicValue[count]);
    count ++;
    }
  }
  for (int i = 0; i < count; i++)
    d_incrValue[i] = (d_maxValue[i] - d_minValue[i])/d_numDivisions[i];
  cerr << "count after table problemsetup: " << count << std::endl;

}



#if 0
int
MixRxnTableInfo::getTemperatureIndex () const {
  return d_temperatureIndex;
}

int
MixRxnTableInfo::getDensityIndex () const {
  return d_densityIndex;
}

#endif

//
// $Log$
// Revision 1.4  2001/10/11 18:48:58  divyar
// Made changes to Mixing
//
// Revision 1.2  2001/07/16 21:15:38  rawat
// added enthalpy solver and Jennifer's changes in Mixing and Reaction model required for ILDM and non-adiabatic cases
//
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
//

