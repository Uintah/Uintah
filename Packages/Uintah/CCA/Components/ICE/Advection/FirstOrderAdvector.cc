#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderAdvector.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Exceptions/OutFluxVolume.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
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

  new_dw->allocate(d_OFS, OFS_CCLabel,0, patch, Ghost::AroundCells,1);
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
			   const Patch* patch)

{
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom,delX_right, delX_left, delZ_front, delZ_back;
  double delX_tmp, delY_tmp,   delZ_tmp;

  // Compute outfluxes 
  const IntVector gc(1,1,1);
  bool err=false;
  IntVector err_cell(0,0,0);
  double err_total_fluxout = 0;
  double err_vol = 0;
  for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
    IntVector curcell = *iter;
    delY_top    = std::max(0.0, (vvel_FC[curcell+IntVector(0,1,0)] * delT));
    delY_bottom = std::max(0.0,-(vvel_FC[curcell+IntVector(0,0,0)] * delT));
    delX_right  = std::max(0.0, (uvel_FC[curcell+IntVector(1,0,0)] * delT));
    delX_left   = std::max(0.0,-(uvel_FC[curcell+IntVector(0,0,0)] * delT));
    delZ_front  = std::max(0.0, (wvel_FC[curcell+IntVector(0,0,1)] * delT));
    delZ_back   = std::max(0.0,-(wvel_FC[curcell+IntVector(0,0,0)] * delT));
    
    delX_tmp    = dx.x() - delX_right - delX_left;
    delY_tmp    = dx.y() - delY_top   - delY_bottom;
    delZ_tmp    = dx.z() - delZ_front - delZ_back;
    
    //__________________________________
    //   SLAB outfluxes
    d_OFS[curcell].d_fflux[TOP]   = delY_top   * delX_tmp * delZ_tmp;
    d_OFS[curcell].d_fflux[BOTTOM]= delY_bottom* delX_tmp * delZ_tmp;
    d_OFS[curcell].d_fflux[RIGHT] = delX_right * delY_tmp * delZ_tmp;
    d_OFS[curcell].d_fflux[LEFT]  = delX_left  * delY_tmp * delZ_tmp;
    d_OFS[curcell].d_fflux[FRONT] = delZ_front * delX_tmp * delY_tmp;
    d_OFS[curcell].d_fflux[BACK]  = delZ_back  * delX_tmp * delY_tmp;

    //__________________________________
    //  Bullet proofing
    double total_fluxout = 0.0;
    for(int face = TOP; face <= BACK; face++ )  {
      total_fluxout  += d_OFS[curcell].d_fflux[face];
    }

    if (total_fluxout > vol) {
      err_cell = *iter;
      err_total_fluxout = total_fluxout;
      err_vol = vol;
      err=true;
    }
  }
  if(err)
    throw OutFluxVolume(err_cell,err_total_fluxout,err_vol);

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
				 CCVariable<double>& q_advected)
{

  advect<double>(q_CC,patch,q_advected);

}


void FirstOrderAdvector::advectQ(const CCVariable<Vector>& q_CC,
				 const Patch* patch,
				 CCVariable<Vector>& q_advected)
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
    for(int face = TOP; face <= BACK; face++ )  {
      sum_q_outflux  += q_CC[curcell] * d_OFS[curcell].d_fflux[face];
    }

    //__________________________________
    //  INFLUX: SLABS
    adjcell = IntVector(i, j+1, k);	// TOP
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[BOTTOM];
    adjcell = IntVector(i, j-1, k);	// BOTTOM
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[TOP];
    adjcell = IntVector(i+1, j, k);	// RIGHT
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[LEFT];
    adjcell = IntVector(i-1, j, k);	// LEFT
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[RIGHT];
    adjcell = IntVector(i, j, k+1);	// FRONT
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[BACK];
    adjcell = IntVector(i, j, k-1);	// BACK
    sum_q_influx  += q_CC[adjcell] * d_OFS[adjcell].d_fflux[FRONT];

    //__________________________________
    //  Calculate the advected q at t + delta t
    q_advected[curcell] = sum_q_influx - sum_q_outflux;

  }

}

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








