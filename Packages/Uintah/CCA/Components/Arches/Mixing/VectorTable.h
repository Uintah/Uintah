//----- VectorTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_VectorTable_h
#define Uintah_Component_Arches_VectorTable_h

/****************************************************************************
CLASS
    VectorTable
      VectorTable class provides a vector data structure to store table 
      information.

GENERAL INFORMATION
    VectorTable.h - Declaration of VectorTable Class

    Author: Jennifer Spinti (spinti@crsim.utah.edu)

    Creation Date: 17 January 2002

    C-SAFE

    Copyright U of U 2002

KEYWORDS

DESCRIPTION
    VectorTable stores a table in vector of vectors. The dimension of the 
    first vector is the number of state space variables. The sdimension is the total
    number of combinations of independent variables for which the state space
    variables are tabulated. It can be used to stored subgrid scale mixing and/or
    reaction model data for systems with few independent variables. It provides
    search and insert capabilities into the Vector data structure.

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS:
   1.


  **************************************************************************/

#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTable.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class MixRxnTableInfo;
  class Stream;
 
class VectorTable: public MixRxnTable {

 public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Constructs an instance of VectorTable.
  //
  VectorTable(int numIndepVars, MixRxnTableInfo* tableInfo);

  // GROUP: Destructor:
  ///////////////////////////////////////////////////////////////////////
  //
  // Destructor 
  //
  virtual ~VectorTable();

  // GROUP: Manipulate
  //////////////////////////////////////////////////////////////////////
  // Inserts stateSpaceVars vector in table at location based on keyIndex.  
  //virtual bool Insert(int keyIndex[], std::vector<double>& stateSpaceVars); 
  virtual bool Insert(int keyIndex[], Stream& stateSpaceVars);

  // GROUP: Access
  //////////////////////////////////////////////////////////////////////
  // Lookup function looks up state space vector in the table at location 
  // based on keyIndex. If it is found, it stores the state space vars in 
  // stateSpaceVars and returns true, else it just returns false.
  //virtual bool Lookup(int keyIndex[], std::vector<double>& stateSpaceVars);
  virtual bool Lookup(int keyIndex[], Stream& stateSpaceVars);


 private:
 
  int d_numIndepVars;
  //std::vector< std::vector <double> > d_tableVec;
  std::vector<double> d_stateSpaceVec;
  MixRxnTableInfo* d_tableInfo;
  std::vector<Stream> d_tableVec;
    
}; // End Class VectorTable

} // end namespace Uintah

#endif
