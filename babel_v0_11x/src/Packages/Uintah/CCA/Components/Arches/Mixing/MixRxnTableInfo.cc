//----- MixRxnTableInfo.cc --------------------------------------------------


#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>
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
  d_maxValue = new double[numTableDim];
  d_minValue = new double[numTableDim];
  d_numDivsBelow = new int[numTableDim];
  d_numDivsAbove = new int[numTableDim];
  d_incrValueBelow = new double[numTableDim];
  d_incrValueAbove = new double[numTableDim];
  d_stoicValue = new double[numTableDim];
      

}

MixRxnTableInfo::~MixRxnTableInfo() {
  delete [] d_maxValue;
  delete [] d_minValue;
  delete [] d_numDivsBelow;
  delete [] d_numDivsAbove;
  delete [] d_incrValueBelow;  
  delete [] d_incrValueAbove;
  delete [] d_stoicValue;

}
// Problem Setup for MixRxnTableInfo
//****************************************************************************
void 
MixRxnTableInfo::problemSetup(const ProblemSpecP& params, bool varFlag,
			      const MixingModel* mixModel)
{
  ProblemSpecP db = params->findBlock("MixRxnTableInfo");
  int count = 0;
  if (!(mixModel->isAdiabatic())) { // if not adiabatic
    ProblemSpecP enth_db = db->findBlock("Enthalpy");
 
    enth_db->require("MaxValue",d_maxValue[count]);
    enth_db->require("MinValue",d_minValue[count]);
    enth_db->require("MidValue",d_stoicValue[count]);
    enth_db->require("NumDivsBelow",d_numDivsBelow[count]);
    enth_db->require("NumDivsAbove",d_numDivsAbove[count]);
  count ++;
  }
  if (mixModel->getNumMixVars()) {
    for (ProblemSpecP mixfrac_db = db->findBlock("MixtureFraction");
	 mixfrac_db != 0; mixfrac_db = db->findNextBlock("MixtureFraction")) {
      mixfrac_db->require("MaxValue",d_maxValue[count]);
      mixfrac_db->require("MinValue",d_minValue[count]);
      mixfrac_db->require("MidValue",d_stoicValue[count]);
      mixfrac_db->require("NumDivsBelow",d_numDivsBelow[count]);
      mixfrac_db->require("NumDivsAbove",d_numDivsAbove[count]);
    count ++;
    }
  }

  if (varFlag) {
    if (mixModel->getNumMixStatVars()) {
      for (ProblemSpecP mixvar_db = db->findBlock("MixFracVariance");
	   mixvar_db != 0; mixvar_db = db->findNextBlock("MixFracVariance")) {
	mixvar_db->require("MaxValue",d_maxValue[count]);
	mixvar_db->require("MinValue",d_minValue[count]);
	mixvar_db->require("MidValue",d_stoicValue[count]);
	mixvar_db->require("NumDivsBelow",d_numDivsBelow[count]);
	mixvar_db->require("NumDivsAbove",d_numDivsAbove[count]);
	count ++;
      }
    }
  }

  if (mixModel->getNumRxnVars()) {
    for (ProblemSpecP rxnvar_db = db->findBlock("ReactionVariables");
	 rxnvar_db != 0; rxnvar_db = db->findNextBlock("ReactionVariables")) {
      rxnvar_db->require("MaxValue",d_maxValue[count]);
      rxnvar_db->require("MinValue",d_minValue[count]);
      rxnvar_db->require("MidValue",d_stoicValue[count]);
      rxnvar_db->require("NumDivsBelow",d_numDivsBelow[count]);
      rxnvar_db->require("NumDivsAbove",d_numDivsAbove[count]);
    count ++;
    }
  }

  for (int i = 0; i < count; i++) {
    d_incrValueBelow[i] = (d_stoicValue[i] - d_minValue[i])/d_numDivsBelow[i];
    d_incrValueAbove[i] = (d_maxValue[i] - d_stoicValue[i])/d_numDivsAbove[i];
    //cout << d_stoicValue[i] << " " << d_minValue[i] << " " << d_numDivsBelow[i] << endl;
    //cerr << "incr value = " << d_incrValueBelow[i] << " " << d_incrValueAbove[i] << endl;
  } 
  //cerr << "count after table problemsetup: " << count << std::endl;

}


//
// $Log$
// Revision 1.9  2003/01/22 00:43:04  spinti
// Added improved BetaPDF mixing model and capability to create a betaPDF table a priori. Cleaned up favre averaging and streamlined sections of code.
//
// Revision 1.8  2002/05/31 22:04:44  spinti
// *** empty log message ***
//
// Revision 1.6  2002/03/28 23:14:51  spinti
// 1. Added in capability to save mixing and reaction tables as KDTree or 2DVector
// 2. Tables can be declared either static or dynamic
// 3. Added capability to run using static clipped Gaussian MixingModel table
// 4. Removed mean values mixing model option from PDFMixingModel and made it
//    a separate mixing model, MeanMixingModel.
//
// Revision 1.5  2001/11/08 19:13:44  spinti
// 1. Corrected minor problems in ILDMReactionModel.cc
// 2. Added tabulation capability to StanjanEquilibriumReactionModel.cc. Now,
//    a reaction table is created dynamically. The limits and spacing in the
//    table are specified in the *.ups file.
// 3. Corrected the mixture temperature computation in Stream::addStream. It
//    now is computed using a Newton search.
// 4. Made other minor corrections to various reaction model files.
//
// Revision 1.2  2001/07/16 21:15:38  rawat
// added enthalpy solver and Jennifer's changes in Mixing and Reaction model required for ILDM and non-adiabatic cases
//
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
//

