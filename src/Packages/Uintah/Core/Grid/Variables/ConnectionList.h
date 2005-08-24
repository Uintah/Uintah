#ifndef Packages_Uintah_Core_Grid_ConnectionList_h
#define Packages_Uintah_Core_Grid_ConnectionList_h


/*--------------------------------------------------------------------------
CLASS
   ConnectionList
   
   Type of list of unstructured graph connections for implicit AMR ICE
   pressure solve.

GENERAL INFORMATION

   File: ConnectionList.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   Stencil7.

DESCRIPTION   
   ConnectionList Type of list of unstructured graph connections for
   implicit AMR ICE pressure solve. It is currently designed to be used
   as a type of CCVariable, not as a perPatch variable, in case we may need
   to trasmit it across patches/processors in the future.

WARNING
   The MPI trasmit functions (swapbytes, makeMPI_ConnectionList) are not
   yet implemented.
   --------------------------------------------------------------------------*/

#include <vector>
#include <Core/Util/FancyAssert.h>
#include <Core/Geometry/IntVector.h>

namespace Uintah {
  // Forward declarations
  class TypeDescription;
  
  //___________________________________________________________________
  // class GraphConnection~
  // A single graph connection from "_this" cell to cell "_other". In
  // the future, if we only use this inside ConnectionList and call
  // ConnectionList as a type to CCVariable/NCVariable/etc., we can
  // omit the member "_this" as we already know what the current cell is.
  //___________________________________________________________________
  struct GraphConnection {
    SCIRun::IntVector _this;   // This cell's index
    SCIRun::IntVector _other;  // Other cell's index (at next-finer level)
    double            _weight; // Matrix entry A(this,other)
  };

  //___________________________________________________________________
  // class ConnectionList~
  // List of unstructured graph connections.
  //___________________________________________________________________
  // This is what we ultimately want to do, but to have a forward
  // class declaration in Uintah/Core/Disclosure/TypeUtils.h
  // without including this header file, we do it instead in an ugly way:
  //  typedef std::vector<GraphConnection> ConnectionList;

  class ConnectionList {
  public:
    std::vector<GraphConnection> _this; // That's retarded but such is life
  };


} // end namespace Uintah

namespace SCIRun {
  void swapbytes(Uintah::ConnectionList&);
} // end namespace SCIRun

#endif // Packages_Uintah_Core_Grid_ConnectionList_h

