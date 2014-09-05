
#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Util/Endian.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <mpi.h>

using namespace Uintah;

namespace Uintah {
  static MPI_Datatype makeMPI_Stencil7()
  {
    ASSERTEQ(sizeof(Stencil7), sizeof(double)*7);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 7, 7, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(Stencil7*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "Stencil7", true, 
				  &makeMPI_Stencil7);
    }
    return td;
  }
  
}

namespace SCIRun {

void swapbytes( Stencil7& a) {
  SWAP_8(a.p);
  SWAP_8(a.e);
  SWAP_8(a.w);
  SWAP_8(a.n);
  SWAP_8(a.s);
  SWAP_8(a.t);
  SWAP_8(a.b);
}

} // namespace SCIRun
