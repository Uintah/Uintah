#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderAdvector.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Exceptions/OutFluxVolume.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <iostream>

using namespace Uintah;
using std::cerr;
using std::endl;


FirstOrderAdvector::FirstOrderAdvector() 
{
  OFS_CCLabel = 0;
}


FirstOrderAdvector::FirstOrderAdvector(DataWarehouse* new_dw, 
                                   const Patch* patch)
{
  OFS_CCLabel = VarLabel::create("OFS_CC",
                             CCVariable<fflux>::getTypeDescription());

  new_dw->allocateTemporary(d_OFS,  patch, Ghost::AroundCells,1);
}


FirstOrderAdvector::~FirstOrderAdvector()
{
  VarLabel::destroy(OFS_CCLabel);
}

FirstOrderAdvector* FirstOrderAdvector::clone(DataWarehouse* new_dw,
                                         const Patch* patch)
{
  return scinew FirstOrderAdvector(new_dw,patch);
}

/* ---------------------------------------------------------------------
 Function~  influxOutfluxVolume--
 Purpose~   calculate the individual outfluxes for each cell.
            This includes the slabs and edge fluxes
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
            
 Steps for each cell:  
 1) calculate the volume for each outflux
 3) set the influx_volume for the appropriate cell = to the q_outflux of the 
    adjacent cell. 

Implementation notes:
The outflux of volume is calculated in each cell in the computational domain
+ one layer of extra cells  surrounding the domain.The face-centered velocity 
needs to be defined on all faces for these cells 

See schematic diagram at bottom of ice.cc for del* definitions
 ---------------------------------------------------------------------  */

void FirstOrderAdvector::inFluxOutFluxVolume(
                        const SFCXVariable<double>& uvel_FC,
                        const SFCYVariable<double>& vvel_FC,
                        const SFCZVariable<double>& wvel_FC,
                        const double& delT, 
                        const Patch* patch,
                        const int& indx)

{
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom,delX_right, delX_left, delZ_front, delZ_back;
  double delX_tmp, delY_tmp, delZ_tmp;
  double delX = dx.x(), delY = dx.y(), delZ = dx.z();

  // Compute outfluxes 
  const IntVector gc(1,1,1);
  double error_test = 0.0;
  int    num_cells = 0;
  
  for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
    IntVector curcell = *iter;
    delY_top    = std::max(0.0, (vvel_FC[curcell+IntVector(0,1,0)] * delT));
    delY_bottom = std::max(0.0,-(vvel_FC[curcell+IntVector(0,0,0)] * delT));
    delX_right  = std::max(0.0, (uvel_FC[curcell+IntVector(1,0,0)] * delT));
    delX_left   = std::max(0.0,-(uvel_FC[curcell+IntVector(0,0,0)] * delT));
    delZ_front  = std::max(0.0, (wvel_FC[curcell+IntVector(0,0,1)] * delT));
    delZ_back   = std::max(0.0,-(wvel_FC[curcell+IntVector(0,0,0)] * delT));
    
    delX_tmp    = delX - delX_right - delX_left;
    delY_tmp    = delY - delY_top   - delY_bottom;
    delZ_tmp    = delZ - delZ_front - delZ_back;     

    //__________________________________
    //   SLAB outfluxes
    double delX_Z_tmp = delX_tmp * delZ_tmp;
    double delX_Y_tmp = delX_tmp * delY_tmp;
    double delY_Z_tmp = delY_tmp * delZ_tmp;
    d_OFS[curcell].d_fflux[TOP]   = delY_top   * delX_Z_tmp;
    d_OFS[curcell].d_fflux[BOTTOM]= delY_bottom* delX_Z_tmp;
    d_OFS[curcell].d_fflux[RIGHT] = delX_right * delY_Z_tmp;
    d_OFS[curcell].d_fflux[LEFT]  = delX_left  * delY_Z_tmp;
    d_OFS[curcell].d_fflux[FRONT] = delZ_front * delX_Y_tmp;
    d_OFS[curcell].d_fflux[BACK]  = delZ_back  * delX_Y_tmp; 

    //__________________________________
    //  Bullet proofing
    double total_fluxout = 0.0;
    total_fluxout  += d_OFS[curcell].d_fflux[TOP];
    total_fluxout  += d_OFS[curcell].d_fflux[BOTTOM];
    total_fluxout  += d_OFS[curcell].d_fflux[RIGHT];
    total_fluxout  += d_OFS[curcell].d_fflux[LEFT];
    total_fluxout  += d_OFS[curcell].d_fflux[FRONT];
    total_fluxout  += d_OFS[curcell].d_fflux[BACK];
    
    num_cells++;
    error_test +=(vol - total_fluxout)/fabs(vol- total_fluxout);
  }  //cell iterator
  
  //__________________________________
  // if total_fluxout > vol then 
  // find the cell and throw an exception.  
  if (fabs(error_test - num_cells) > 1.0e-2) {
    for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
      IntVector curcell = *iter; 
      double total_fluxout = 0.0;
      for(int face = TOP; face <= BACK; face++ )  {
        total_fluxout  += d_OFS[curcell].d_fflux[face];
      }
      if (vol - total_fluxout < 0.0) {
        throw OutFluxVolume(*iter,total_fluxout, vol, indx);
      }
    }  // cell iter
  }  // if total_fluxout > vol
}



/* ---------------------------------------------------------------------
 Function~  ICE::advectQFirst--ADVECTION:
 Purpose~   Calculate the advection of q_CC 
   
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
            
 Steps for each cell:      
- Compute q outflux and q influx for each cell.
- Finally sum the influx and outflux portions
       
 advect_preprocessing MUST be done prior to this function
 ---------------------------------------------------------------------  */

void FirstOrderAdvector::advectQ(const CCVariable<double>& q_CC,
                             const Patch* patch,
                             CCVariable<double>& q_advected,
				 DataWarehouse* /*new_dw*/)
{

  advect<double>(q_CC,patch,q_advected);

}


void FirstOrderAdvector::advectQ(const CCVariable<Vector>& q_CC,
                             const Patch* patch,
                             CCVariable<Vector>& q_advected,
				 DataWarehouse* /*new_dw*/)
{

  advect<Vector>(q_CC,patch,q_advected);

}


template <class T> void FirstOrderAdvector::advect(const CCVariable<T>& q_CC,
                                              const Patch* patch,
                                              CCVariable<T>& q_advected)
  
{
  T  sum_q_outflux, sum_q_influx, zero(0.);
  IntVector adjcell;

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    IntVector curcell = *iter;
    int i = curcell.x();
    int j = curcell.y();
    int k = curcell.z();
   
    sum_q_outflux      = zero;
    sum_q_influx       = zero;
    
    //__________________________________
    //  OUTFLUX: SLAB  
    sum_q_outflux  = (d_OFS[curcell].d_fflux[BOTTOM] + d_OFS[curcell].d_fflux[TOP]
                   +  d_OFS[curcell].d_fflux[LEFT]   + d_OFS[curcell].d_fflux[RIGHT]  
                   +  d_OFS[curcell].d_fflux[BACK]   + d_OFS[curcell].d_fflux[FRONT]) 
                   *  q_CC[curcell];
                   
    //__________________________________
    //  INFLUX: SLABS
    adjcell = IntVector(i, j+1, k);       // TOP
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[BOTTOM];
    adjcell = IntVector(i, j-1, k);       // BOTTOM
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[TOP];
    adjcell = IntVector(i+1, j, k);       // RIGHT
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[LEFT];
    adjcell = IntVector(i-1, j, k);       // LEFT
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[RIGHT];
    adjcell = IntVector(i, j, k+1);       // FRONT
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[BACK];
    adjcell = IntVector(i, j, k-1);       // BACK
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[FRONT];

    //__________________________________
    //  Calculate the advected q at t + delta t
    q_advected[curcell] = sum_q_influx - sum_q_outflux;

  }
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {

  static MPI_Datatype makeMPI_fflux()
  {
    ASSERTEQ(sizeof(FirstOrderAdvector::fflux), sizeof(double)*6);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 6, 6, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(FirstOrderAdvector::fflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                              "FirstOrderAdvector::fflux", true, 
                              &makeMPI_fflux);
    }
    return td;
  }
  
}

namespace SCIRun {

void swapbytes( Uintah::FirstOrderAdvector::fflux& f) {
  double *p = f.d_fflux;
  SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
}

} // namespace SCIRun
