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

    Creation Date: 2 June 1999

    C-SAFE

    Copyright U of U 1999

KEYWORDS
    Mixing_Model, Presumed_PDF, EDC, PDF

DESCRIPTION
    MixRxnTableInfo class reads the table information from an input file. The file
    includes maximum, minimum and total number of divisions for each one of the 
    independent variables.

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS:
   1. Non-Uniform Table


  **************************************************************************/

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <vector>

namespace Uintah {
class PDFMixingModel;
class MixRxnTableInfo {
 public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Constructs an instance of MixRxnTableInfo.
  // Parameters:
  // [in] numTableDim is the number of independent variables including there
  // statistics.
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
  void problemSetup(const ProblemSpecP& params, const PDFMixingModel* mixModel);
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
  //////////////////////////////////////////////////////////////////////
  // getMinValue returns the min value of the variable for any given index
  // Parameters:
  // [in] index is the index of the variable for which min value is required
  // 
  double inline getMinValue(int index) const {
    return d_minValue[index];
  }

  //////////////////////////////////////////////////////////////////////
  // getIncrValue returns the increment at which variable is tabulated
  // Parameters:
  // [in] index is the index of the variable for which increment is required
  // 
  double inline getIncrValue(int index) const {
    return d_incrValue[index];
  }

  //////////////////////////////////////////////////////////////////////
  // getNumDivisions returns the num of divisions at which variable is tabulated
  // Parameters:
  // [in] index is the index of the variable for which numDivisions is required
  // 
  //////////////////////////////////////////////////////////////////////
  // getStoicValue returns the stoichiometric of teh variable
  // Parameters:
  // [in] index is the index of the variable for which stoic value is required
  // 
  double inline getStoicValue(int index) const{
    return d_stoicValue[index];
  }
  
  int inline getNumDivisions(int index) const {
    return d_numDivisions[index];
  }
  

 private:
  int d_numTableDim;
  int *d_numDivisions;
  double *d_maxValue;
  double *d_minValue;
  double *d_incrValue;
  double *d_stoicValue;
    
}; // End Class MixRxnTableInfo

} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
// Revision 1.1  2000/12/18 17:53:10  rawat
// adding mixing model for reacting flows
//
//
