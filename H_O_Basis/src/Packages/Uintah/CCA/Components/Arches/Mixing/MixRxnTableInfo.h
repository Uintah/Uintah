//----- MixRxnTableInfo.h --------------------------------------------------

#ifndef Uintah_Component_Arches_MixRxnTableInfo_h
#define Uintah_Component_Arches_MixRxnTableInfo_h

/****************************************************************************
CLASS
    MixRxnTableInfo
      MixRxnTableInfo class reads and stores in table information for
      both mixing and reaction table

GENERAL INFORMATION
    MixRxnTableInfo.h - Declaration of MixRxnTableInfo Class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    Revised by: Jennifer Spinti (spinti@crsim.utah.edu)

    Creation Date: 2 June 1999
    Last Revision: 9 July 2001

    C-SAFE

    Copyright U of U 1999

KEYWORDS
    Mixing_Model, Presumed_PDF, EDC, PDF

DESCRIPTION
    MixRxnTableInfo class reads the table information from an input file. The file
    includes maximum, minimum and midpoint values and the total number of divisions 
    on each side of the midpoint value for each one of the independent variables.

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS:
   1. Non-Uniform Table


  **************************************************************************/

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class MixingModel;
  class MixRxnTableInfo {
 public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Constructs an instance of MixRxnTableInfo.
  // Parameters:
  // [in] numTableDim is the number of independent variables including
  // their statistics.
  //
  MixRxnTableInfo(int numTableDim);
  // GROUP: Destructor:
  ///////////////////////////////////////////////////////////////////////
  //
  // Destructor 
  //
  ~MixRxnTableInfo();
  
  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  //
  // Set up the problem specification database
  //
  void problemSetup(const ProblemSpecP& params, bool varFlag,
		    const MixingModel* mixModel);

  // GROUP: Access functions
  //////////////////////////////////////////////////////////////////////
  // getMaxValue returns the max value of the variable for any given index
  // Parameters:
  // [in] index is the index of the variable for which max value is required
  // 
  double inline getMaxValue(int index) const{
    return d_maxValue[index];
  }

  //////////////////////////////////////////////////////////////////////
  // getMinValue returns the min value of the variable for any given index
  // Parameters:
  // [in] index is the index of the variable for which min value is required
  // 
  double inline getMinValue(int index) const {
    return d_minValue[index];
  }

  //////////////////////////////////////////////////////////////////////
  // getIncrValueBelow returns the increment at which variable is tabulated
  // below the specified midpoint.
  // Parameters:
  // [in] index is the index of the variable for which increment is required
  // 
  double inline getIncrValueBelow(int index) const {
    return d_incrValueBelow[index];
  }

  //////////////////////////////////////////////////////////////////////
  // getIncrValueAbove returns the increment at which variable is tabulated
  // above the specified midpoint.
  // Parameters:
  // [in] index is the index of the variable for which increment is required
  // 
  double inline getIncrValueAbove(int index) const {
    return d_incrValueAbove[index];
  }

  //////////////////////////////////////////////////////////////////////
  // getNumDivsBelow returns the num of divisions at which variable is 
  // tabulated below the specified midpoint.
  // Parameters:
  // [in] index is the index of the variable for which numDivisions is required
  // 
  int inline getNumDivsBelow(int index) const {
    return d_numDivsBelow[index];
  }

  //////////////////////////////////////////////////////////////////////
  // getNumDivsAbove returns the num of divisions at which variable is 
  // tabulated above the specified midpoint.
  // Parameters:
  // [in] index is the index of the variable for which numDivisions is required
  // 
  int inline getNumDivsAbove(int index) const {
    return d_numDivsAbove[index];
  }
 
  //////////////////////////////////////////////////////////////////////
  // getStoicValue returns the stoichiometric (or midpoint) value of the
  // variable
  // Parameters:
  // [in] index is the index of the variable for which stoic value is required
  // 
  double inline getStoicValue(int index) const{
    return d_stoicValue[index];
  }

  inline int getDensityIndex () const {
    return 0; //Density is in first position in output vector
  }
 
  inline int getTemperatureIndex () const {
    return 2; //Temperature is in third position in output vector
  }

 private:
  int d_numTableDim;

  double *d_maxValue;
  double *d_minValue;
  double *d_incrValueAbove;
  double *d_incrValueBelow;
  double *d_stoicValue;
  int *d_numDivsAbove;
  int *d_numDivsBelow;
    
}; // End Class MixRxnTableInfo

} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.10  2003/08/07 00:48:14  sparker
// SGI 64 bit warnings rampage
//
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
// Revision 1.1  2000/12/18 17:53:10  rawat
// adding mixing model for reacting flows
//
//
