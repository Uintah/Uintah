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
#include <vector>
#include <iostream>

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
  void problemSetup(const ProblemSpecP& params, bool mixTableFlag,
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
// Revision 1.3  2001/09/04 23:44:27  rawat
// Added ReactingScalar transport equation to run ILDM.
// Also, merged Jennifer's changes to run ILDM in the mixing directory.
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
