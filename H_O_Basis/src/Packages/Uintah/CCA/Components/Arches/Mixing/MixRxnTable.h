//----- MixRxnTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_MixRxnTable_h
#define Uintah_Component_Arches_MixRxnTable_h

/***************************************************************************
CLASS
    MixRxnTable
       The MixRxnTable class sets up the data structure for the tables
       created and accessed by the mixing and reaction models.
       
GENERAL INFORMATION
    MixRxnTable.h - Declaration of MixRxnTable class

    Author: Jennifer Spinti (spinti@crsim.utah.edu)
    
    Creation Date : 01-22-2002

    C-SAFE
    
    Copyright U of U 2002

KEYWORDS
   Vector_Table, KDTree, table_data_structure
    
DESCRIPTION
   MixRxnTable is an abstract class which provides a general interface
   to a table type. These tables are used to store the values of state
   space variables as functions of the independent variables. These tables
   can be created dynamically or a priori and then read in. Both the
   mixing and reaction models require such tables.

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <Packages/Uintah/CCA/Components/Arches/Mixing/ReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class MixRxnTableInfo;

class MixRxnTable {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
  //MixRxnTable(int numIndepVars, MixRxnTableInfo* tableInfo);
      MixRxnTable();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for MixRxnTable 
      //
      virtual ~MixRxnTable();

      // GROUP: Manipulate
      //////////////////////////////////////////////////////////////////////
      // Inserts stateSpaceVars vector in table at location based on keyIndex.  
      //virtual bool Insert(int keyIndex[], std::vector<double>& stateSpaceVars) = 0; 
      virtual bool Insert(int keyIndex[], Stream& stateSpaceVars) = 0;  
      
      // GROUP: Access
      //////////////////////////////////////////////////////////////////////
      // Lookup function looks up state space vector in the table at location 
      // based on keyIndex. If it is found, it stores the state space vars in 
      // stateSpaceVars and returns true, else it just returns false.
      //virtual bool Lookup(int keyIndex[], std::vector<double>& stateSpaceVars) = 0;
      virtual bool Lookup(int keyIndex[], Stream& stateSpaceVars) = 0;

protected :

private:

}; // end class MixRxnTable

} // end namespace Uintah

#endif

