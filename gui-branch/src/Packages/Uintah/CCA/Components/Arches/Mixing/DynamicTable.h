//----- MixRxnTableInfo.h --------------------------------------------------

#ifndef Uintah_Component_Arches_DynamicTable_h
#define Uintah_Component_Arches_DynamicTable_h

/****************************************************************************
CLASS
    DynamicTable
      DynamicTable class provides the interface between the mixing or reaction
      table and the classes that require information from these tables.

GENERAL INFORMATION
    DynamicTable.h - Declaration of DynamicTable Class

    Author: Jennifer Spinti (spinti@crsim.utah.edu)

    Creation Date: 24 April 2001

    C-SAFE

    Copyright U of U 2001

KEYWORDS
    KD_Tree, Table_Interpolation, Mixing_Model, Reaction_Model

DESCRIPTION
    DynamicTable class converts values of mixing and reaction variables into
    the index system used by the mixing and reaction tables so that table entries
    can be read. It also performs interpolation on table entries.

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS:
   1.


  **************************************************************************/

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>

#include <vector>

namespace Uintah {
class MixRxnTableInfo;

class DynamicTable {
 public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Constructs an instance of DynamicTable.
  //
  DynamicTable();

  // GROUP: Destructor:
  ///////////////////////////////////////////////////////////////////////
  //
  // Destructor 
  //
  virtual ~DynamicTable();

  Stream getProps(const std::vector<double> mixRxnVar); //Public or private???
  virtual Stream tableLookUp(int* tableKeyIndex) = 0;//???Public or private???
  // Allocates memory for table
  void tableSetup(int numTableDim, MixRxnTableInfo* tableInfo);
 
 protected:
  // GROUP: Actual Action Methods :
  //////////////////////////////////////////////////////////////////////
  //

 

 private:
  // Recursive function to linearly interpolate from the KDTree
  Stream interpolate(int currentDim, int* lowIndex, int* upIndex,
		     double* lowFactor, double* upFactor);
  int d_tableDim;
  MixRxnTableInfo* d_tableInfo;
  // two dimensional arrays for storing information for linear interpolation
  int **d_tableIndexVec;
  double **d_tableBoundsVec;

    
}; // End Class DynamicTable

} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.4  2001/11/08 19:13:44  spinti
// 1. Corrected minor problems in ILDMReactionModel.cc
// 2. Added tabulation capability to StanjanEquilibriumReactionModel.cc. Now,
//    a reaction table is created dynamically. The limits and spacing in the
//    table are specified in the *.ups file.
// 3. Corrected the mixture temperature computation in Stream::addStream. It
//    now is computed using a Newton search.
// 4. Made other minor corrections to various reaction model files.
//
//
