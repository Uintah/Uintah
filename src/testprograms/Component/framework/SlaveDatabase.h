#ifndef SLAVEDATABASE_H
#define SLAVEDATABASE_H

//
// SlaveDatabase.h
//
// This class is used by the MasterController to manage information 
// about which SlaveControllers have been created and what there
// status is.
//

#include <testprograms/Component/dav/Common/MastContInterfaces_sidl.h>

#include <Core/Exceptions/InternalError.h>

#include <Core/CCA/SSIDL/array.h>

#include <string>

namespace dav {

using namespace SCIRun;
using std::string;
using SSIDL::array1;

class SlaveInfo {
  
public:
  SlaveInfo( string machineId, Mc2Sc mc2Sc, int maxProcs );

  string       d_machineId;
  Mc2Sc        d_mc2Sc;
  int          d_maxProcs;  // Number of processors on this SC's machine.
};

class SlaveDatabase {
public:

  SlaveDatabase();
  ~SlaveDatabase();

  int         numActive();
  SlaveInfo * find( string machineId );
  SlaveInfo * leastLoaded();
  void        add( SlaveInfo * si ) throw (InternalError);
  void        remove( SlaveInfo * si );

  void        getMachineIds( array1<string> & ids );

  void        updateGuis();  // Runs through list of SCs and tells
                             // them to send gui info to the gui

  void        shutDown();  // Runs through the list of each of the active
                           // SCs and sends a shutDown message to them.

private:

  array1<SlaveInfo*>::iterator   findPosition( string machineId );

  int                 d_numSlaves;
  array1<SlaveInfo *> d_slaves;
};

} // end namespace dav

#endif
