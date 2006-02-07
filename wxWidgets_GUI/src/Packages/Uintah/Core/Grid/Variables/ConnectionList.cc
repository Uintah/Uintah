/*--------------------------------------------------------------------------
 * File: ConnectionList.cc
 *
 * Implementation of unstructured connection list type. Mostly taken from
 * Stencil7.cc.
 *--------------------------------------------------------------------------*/

#include <Packages/Uintah/Core/Grid/Variables/ConnectionList.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Endian.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <mpi.h>

using namespace Uintah;

namespace Uintah {
  static MPI_Datatype makeMPI_ConnectionList()
  {
    throw SCIRun::InternalError("makeMPI_ConnectionList() not yet implemented",
                                __FILE__, __LINE__);
    // Code from Stencil7.cc:
    //    ASSERTEQ(sizeof(ConnectionList), sizeof(double)*7);
    //    MPI_Datatype mpitype;
    //    MPI_Type_vector(1, 7, 7, MPI_DOUBLE, &mpitype);
    //    MPI_Type_commit(&mpitype);
    //    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(ConnectionList*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "ConnectionList", true, 
				  &makeMPI_ConnectionList);
    }
    return td;
  }
  
} // end namespace Uintah

namespace SCIRun {

  void swapbytes(ConnectionList& /*a*/) {
    throw InternalError("swapbytes(ConnectionList&) not yet implemented",
                        __FILE__, __LINE__);
  }

} // end namespace SCIRun
