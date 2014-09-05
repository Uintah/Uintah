#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Util/FancyAssert.h>
#include <iostream>
using namespace Uintah;

Advector::Advector()
{
    //__________________________________
    //  outflux/influx slabs
    OF_slab[RIGHT] = RIGHT;         IF_slab[RIGHT]  = LEFT;
    OF_slab[LEFT]  = LEFT;          IF_slab[LEFT]   = RIGHT;
    OF_slab[TOP]   = TOP;           IF_slab[TOP]    = BOTTOM;
    OF_slab[BOTTOM]= BOTTOM;        IF_slab[BOTTOM] = TOP;  
    OF_slab[FRONT] = FRONT;         IF_slab[FRONT]  = BACK;
    OF_slab[BACK]  = BACK;          IF_slab[BACK]   = FRONT;   

    // Slab adjacent cell
    S_ac[RIGHT]  =  IntVector( 1, 0, 0);   
    S_ac[LEFT]   =  IntVector(-1, 0, 0);   
    S_ac[TOP]    =  IntVector( 0, 1, 0);   
    S_ac[BOTTOM] =  IntVector( 0,-1, 0);   
    S_ac[FRONT]  =  IntVector( 0, 0, 1);   
    S_ac[BACK]   =  IntVector( 0, 0,-1);
}

Advector::~Advector()
{
}

//______________________________________________________________________
//
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif  

//______________________________________________________________________
//  
namespace Uintah {
  //__________________________________
  void  warning_restartTimestep( vector<IntVector> badCells,
                                 vector<double> badOutFlux,
                                 const double vol,
                                 const int indx,
                                 const Patch* patch,
                                 DataWarehouse* new_dw)
  {
    cout << Parallel::getMPIRank() << " ERROR: ICE Advection operator: "
         << " Influx_outflux error detected, "
         << " patch " << patch->getID()
         << ", Level " << patch->getLevel()->getIndex()
         << ", matl indx "<< indx << " " << badCells.size() << " bad cells "
         << endl;
         
    for (int i = 0; i<(int) badCells.size(); i++) {
      cout << Parallel::getMPIRank() << "  cell " << badCells[i] 
           << " \t\t total_outflux (" << badOutFlux[i]<< ") > cellVol (" 
           << vol << ")" << endl;
      break;
    }
    //cout << " A timestep restart has been requested \n " << endl;
    new_dw->restartTimestep();
  }
  
  //__________________________________
  static MPI_Datatype makeMPI_fflux()
  {
    ASSERTEQ(sizeof(fflux), sizeof(double)*6);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 6, 6, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }

  const TypeDescription* fun_getTypeDescription(fflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "fflux", true, 
				  &makeMPI_fflux);
    }
    return td;
  }
} // namespace Uintah


//______________________________________________________________________
//  
namespace SCIRun {

  void swapbytes( Uintah::fflux& f) {
    double *p = f.d_fflux;
    SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  }
} // namespace SCIRun
