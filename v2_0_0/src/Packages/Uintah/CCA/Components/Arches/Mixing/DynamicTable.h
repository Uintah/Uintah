//----- DynamicTable.h --------------------------------------------------

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

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

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

  void getProps(const std::vector<double>& mixRxnVar, Stream& outStream); //Public or private???
  virtual void tableLookUp(int* tableKeyIndex, Stream& tableValue) = 0;//???Public or private???
  // Allocates memory for table
  void tableSetup(int numTableDim, MixRxnTableInfo* tableInfo);
 
 protected:
  // GROUP: Actual Action Methods :
  //////////////////////////////////////////////////////////////////////
  //

 

 private:
  // Recursive function to linearly interpolate from the KDTree
  void interpolate(int currentDim, int* lowIndex, int* upIndex,
		     double* lowFactor, double* upFactor, Stream& interpValue);
  int d_tableDim;
  MixRxnTableInfo* d_tableInfo;
  // two dimensional arrays for storing information for linear interpolation
  int **d_tableIndexVec;
  double **d_tableBoundsVec;
  Stream d_upValue; // Used in interpolate
  Stream d_lowValue; // Used in interpolate

    
}; // End Class DynamicTable

} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.9  2003/08/07 00:48:14  sparker
// SGI 64 bit warnings rampage
//
// Revision 1.8  2003/01/22 00:43:04  spinti
// Added improved BetaPDF mixing model and capability to create a betaPDF table a priori. Cleaned up favre averaging and streamlined sections of code.
//
// Revision 1.7  2002/05/31 22:04:44  spinti
// *** empty log message ***
//
// Revision 1.5  2002/03/28 23:14:50  spinti
// 1. Added in capability to save mixing and reaction tables as KDTree or 2DVector
// 2. Tables can be declared either static or dynamic
// 3. Added capability to run using static clipped Gaussian MixingModel table
// 4. Removed mean values mixing model option from PDFMixingModel and made it
//    a separate mixing model, MeanMixingModel.
//
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
