#ifndef UINTAH_COMPONENTS_SCHEDULERS_DWMPIHANDLER_H
#define UINTAH_COMPONENTS_SCHEDULERS_DWMPIHANDLER_H

#include <SCICore/Thread/Runnable.h>
#include <SCICore/Thread/Thread.h>

using SCICore::Thread::Thread;
using SCICore::Thread::Runnable;

#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/DataWarehouseP.h>

namespace Uintah {

class DWMpiHandler : public Runnable {

public:

  enum DataType { ReductionVar, GridVar, Quit };

  struct MpiDataRequest {
    int fromMpiRank; // Rank of process that wants to send info to this DW.
    int toMpiRank;   // Rank of the process being sent the data.
    int tag;         // Tag to use in the actual send.
    char     varName[ 48 ];
    int      region;
    int      generation; // Generation of DW making request.
                         //   Should match generation of the currently
                         //   registered DW with this DWMpiHandler.
    DataType type;// Type of data that will be sent
  };

  static const int  MAX_BUFFER_SIZE;
  static const int  DATA_REQUEST_TAG;
  static const int  DATA_MESSAGE_TAG;

  DWMpiHandler();
  ~DWMpiHandler();

  // Sends out an MPI message to all of the DWMpiHandlers telling them
  // to shutdown.  This should only be called once by MPI Process of
  // Rank 0.
  static void shutdown( int numMpiProcs );

  void registerDW( DataWarehouseP dw );

  // A DataWarehouseP must be registered before the thread is created...
  void run();

private:

  DataWarehouseP d_dw; // The current DataWarehouse.

};

} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.2  2000/05/11 20:10:22  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.1  2000/05/08 18:31:25  dav
// DWMpiHandler handles MPI requests and sends for the datawarehouses
//
//
