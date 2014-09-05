#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderBase.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/Endian.h>
#include <iostream>

using namespace Uintah;

SecondOrderBase::SecondOrderBase()
{
}
SecondOrderBase::~SecondOrderBase()
{
  // Destructor
}
//______________________________________________________________________
//

//______________________________________________________________________
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif
namespace Uintah {

  static MPI_Datatype makeMPI_fflux()
  {
    ASSERTEQ(sizeof(SecondOrderBase::fflux), sizeof(double)*6);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 6, 6, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(SecondOrderBase::fflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "SecondOrderBase::fflux", true, 
				  &makeMPI_fflux);
    }
    return td;
  }
}

namespace SCIRun {

  void
  swapbytes( Uintah::SecondOrderBase::fflux& f) {
    double *p = f.d_fflux;
    SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  }

} // namespace SCIRun
