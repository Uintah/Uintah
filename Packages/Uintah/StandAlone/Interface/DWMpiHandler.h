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
  DWMpiHandler();

  void registerDW( DataWarehouseP dw );

  void run();

private:

  enum DataType { ReductionVar, GridVar, Quit };

  struct MpiDataRequest {
    int fromMpiRank; // Rank of process that wants to send info to this DW.
    int toMpiRank;   // Rank of the process being sent the data.
    int tag;         // Tag to use in the actual send.
    DataType type;// Type of data that will be sent
    char     varName[ 50 ];
    int      region;
    int      generation; // Generation of DW making request.
  };                     //   Should match generation of the currently
                         //   registered DW with this DWMpiHandler.
  static const int  MAX_BUFFER_SIZE;
  static const int  MPI_DATA_REQUEST_TAG;

  DataWarehouseP d_dw; // The current DataWarehouse.

};

} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2000/05/08 18:31:25  dav
// DWMpiHandler handles MPI requests and sends for the datawarehouses
//
//
