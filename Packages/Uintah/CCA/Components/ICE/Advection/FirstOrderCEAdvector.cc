#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderCEAdvector.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Exceptions/OutFluxVolume.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <iostream>

using namespace Uintah;

FirstOrderCEAdvector::FirstOrderCEAdvector()
{
  OFE_CCLabel = 0;
  OFC_CCLabel = 0;
}

FirstOrderCEAdvector::FirstOrderCEAdvector(DataWarehouse* new_dw, 
                                          const Patch* patch)
  :   d_advector(new_dw,patch)
{


  OFE_CCLabel = VarLabel::create("OFE_CC",
                             CCVariable<eflux>::getTypeDescription());
  OFC_CCLabel = VarLabel::create("OFC_CC",
                             CCVariable<cflux>::getTypeDescription());
  new_dw->allocateTemporary(d_OFE,  patch, Ghost::AroundCells,1);
  new_dw->allocateTemporary(d_OFC,  patch, Ghost::AroundCells,1);
}


FirstOrderCEAdvector::~FirstOrderCEAdvector()
{
  VarLabel::destroy(OFE_CCLabel);
  VarLabel::destroy(OFC_CCLabel);
}

FirstOrderCEAdvector* FirstOrderCEAdvector::clone(DataWarehouse* new_dw,
                                   const Patch* patch)
{
  return scinew FirstOrderCEAdvector(new_dw,patch);
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

void FirstOrderCEAdvector::inFluxOutFluxVolume(
                        const SFCXVariable<double>& uvel_FC,
                        const SFCYVariable<double>& vvel_FC,
                        const SFCZVariable<double>& wvel_FC,
                        const double& delT, 
                        const Patch* patch,
                        const int&   indx,
                        const bool& bulletProof_test)

{
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom,delX_right, delX_left, delZ_front, delZ_back;
  double delX_tmp, delY_tmp,   delZ_tmp;

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
    
    delX_tmp    = dx.x() - delX_right - delX_left;
    delY_tmp    = dx.y() - delY_top   - delY_bottom;
    delZ_tmp    = dx.z() - delZ_front - delZ_back;
    
    //__________________________________
    //   SLAB outfluxes
    d_advector.d_OFS[curcell].d_fflux[FOA::TOP]   = delY_top   * delX_tmp * delZ_tmp;
    d_advector.d_OFS[curcell].d_fflux[FOA::BOTTOM]= delY_bottom* delX_tmp * delZ_tmp;
    d_advector.d_OFS[curcell].d_fflux[FOA::RIGHT] = delX_right * delY_tmp * delZ_tmp;
    d_advector.d_OFS[curcell].d_fflux[FOA::LEFT]  = delX_left  * delY_tmp * delZ_tmp;
    d_advector.d_OFS[curcell].d_fflux[FOA::FRONT] = delZ_front * delX_tmp * delY_tmp;
    d_advector.d_OFS[curcell].d_fflux[FOA::BACK]  = delZ_back  * delX_tmp * delY_tmp;

    // Edge flux terms
    d_OFE[curcell].d_eflux[TOP_R]     = delY_top      * delX_right * delZ_tmp;
    d_OFE[curcell].d_eflux[TOP_FR]    = delY_top      * delX_tmp   * delZ_front;
    d_OFE[curcell].d_eflux[TOP_L]     = delY_top      * delX_left  * delZ_tmp;
    d_OFE[curcell].d_eflux[TOP_BK]    = delY_top      * delX_tmp   * delZ_back;
    
    d_OFE[curcell].d_eflux[BOT_R]     = delY_bottom   * delX_right * delZ_tmp;
    d_OFE[curcell].d_eflux[BOT_FR]    = delY_bottom   * delX_tmp   * delZ_front;
    d_OFE[curcell].d_eflux[BOT_L]     = delY_bottom   * delX_left  * delZ_tmp;
    d_OFE[curcell].d_eflux[BOT_BK]    = delY_bottom   * delX_tmp   * delZ_back;
    
    d_OFE[curcell].d_eflux[RIGHT_BK]  = delY_tmp      * delX_right * delZ_back;
    d_OFE[curcell].d_eflux[RIGHT_FR]  = delY_tmp      * delX_right * delZ_front;
    
    d_OFE[curcell].d_eflux[LEFT_BK]   = delY_tmp      * delX_left  * delZ_back;
    d_OFE[curcell].d_eflux[LEFT_FR]   = delY_tmp      * delX_left  * delZ_front;
    
    //__________________________________
    //   Corner flux terms
    d_OFC[curcell].d_cflux[TOP_R_BK]  = delY_top      * delX_right * delZ_back;
    d_OFC[curcell].d_cflux[TOP_R_FR]  = delY_top      * delX_right * delZ_front;
    d_OFC[curcell].d_cflux[TOP_L_BK]  = delY_top      * delX_left  * delZ_back;
    d_OFC[curcell].d_cflux[TOP_L_FR]  = delY_top      * delX_left  * delZ_front;
    
    d_OFC[curcell].d_cflux[BOT_R_BK]  = delY_bottom   * delX_right * delZ_back;
    d_OFC[curcell].d_cflux[BOT_R_FR]  = delY_bottom   * delX_right * delZ_front;
    d_OFC[curcell].d_cflux[BOT_L_BK]  = delY_bottom   * delX_left  * delZ_back;
    d_OFC[curcell].d_cflux[BOT_L_FR]  = delY_bottom   * delX_left  * delZ_front;

    //__________________________________
    //  Bullet proofing
    double total_fluxout = 0.0;
    for(int face = FOA::TOP; face <= FOA::BACK; 
       face++ )  {
      total_fluxout  += d_advector.d_OFS[curcell].d_fflux[face];
    }
    for(int edge = TOP_R; edge <= LEFT_FR; edge++ )  {
      total_fluxout  += d_OFE[curcell].d_eflux[edge];
    }
    for(int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
      total_fluxout  += d_OFC[curcell].d_cflux[corner];
    }
    num_cells++;
    error_test +=(vol - total_fluxout)/fabs(vol- total_fluxout);
  }  // cell iterator
  //__________________________________
  // if total_fluxout > vol then 
  // find the cell and throw an exception.  
  if (fabs(error_test - num_cells) > 1.0e-2 && bulletProof_test) {
    for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
      IntVector curcell = *iter; 
      double total_fluxout = 0.0;
      for(int face = FOA::TOP; face <= FOA::BACK; face++ )  {
        total_fluxout  += d_advector.d_OFS[curcell].d_fflux[face];
      }
      for(int edge = TOP_R; edge <= LEFT_FR; edge++ )  {
        total_fluxout  += d_OFE[curcell].d_eflux[edge];
      }
      for(int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
        total_fluxout  += d_OFC[curcell].d_cflux[corner];
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

void FirstOrderCEAdvector::advectQ(const CCVariable<double>& q_CC,
                             const Patch* patch,
                             CCVariable<double>& q_advected,
				 DataWarehouse* /*new_dw*/)
{

  advect<double>(q_CC,patch,q_advected);

}


void FirstOrderCEAdvector::advectQ(const CCVariable<Vector>& q_CC,
                             const Patch* patch,
                             CCVariable<Vector>& q_advected,
				 DataWarehouse* /*new_dw*/)
{

  advect<Vector>(q_CC,patch,q_advected);

}


template <class T> void FirstOrderCEAdvector::advect(const CCVariable<T>& q_CC,
                                              const Patch* patch,
                                              CCVariable<T>& q_advected)
{
  T sum_q_outflux(0.), sum_q_outflux_EF(0.), sum_q_outflux_CF(0.);
  T sum_q_influx(0.), sum_q_influx_EF(0.), sum_q_influx_CF(0.);
  IntVector adjcell;
  
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    IntVector curcell = *iter;
    int i = curcell.x();
    int j = curcell.y();
    int k = curcell.z();
   
    sum_q_outflux      *= 0.0;
    sum_q_outflux_EF   *= 0.0;
    sum_q_outflux_CF   *= 0.0;
    sum_q_influx       *= 0.0;
    sum_q_influx_EF    *= 0.0;
    sum_q_influx_CF    *= 0.0;
    
    //__________________________________
    //  OUTFLUX: SLAB 
    for(int face = FOA::TOP; face <= FOA::BACK; 
       face++ )  {
      sum_q_outflux  += q_CC[curcell] * d_advector.d_OFS[curcell].d_fflux[face];
    }
    //__________________________________
    //  OUTFLUX: EDGE_FLUX
    for(int edge = TOP_R; edge <= LEFT_FR; edge++ )   {
      sum_q_outflux_EF += q_CC[curcell] * d_OFE[curcell].d_eflux[edge];
    }
    //__________________________________
    //  OUTFLUX: CORNER FLUX
    for(int corner = TOP_R_BK;        corner <= BOT_L_FR; corner++ )  {
      sum_q_outflux_CF +=  q_CC[curcell] * d_OFC[curcell].d_cflux[corner];
    } 

    //__________________________________
    //  INFLUX: SLABS
    adjcell = IntVector(i, j+1, k);       // TOP
    sum_q_influx  += q_CC[adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::BOTTOM];
    adjcell = IntVector(i, j-1, k);       // BOTTOM
    sum_q_influx  += q_CC[adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::TOP];
    adjcell = IntVector(i+1, j, k);       // RIGHT
    sum_q_influx  += q_CC[adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::LEFT];
    adjcell = IntVector(i-1, j, k);       // LEFT
    sum_q_influx  += q_CC[adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::RIGHT];
    adjcell = IntVector(i, j, k+1);       // FRONT
    sum_q_influx  += q_CC[adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::BACK];
    adjcell = IntVector(i, j, k-1);       // BACK
    sum_q_influx  += q_CC[adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::FRONT];
    //__________________________________
    //  INFLUX: EDGES
    adjcell = IntVector(i+1, j+1, k);       // TOP_R
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[BOT_L];
    adjcell = IntVector(i, j+1, k+1);   // TOP_FR
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[BOT_BK];
    adjcell = IntVector(i-1, j+1, k);       // TOP_L
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[BOT_R];
    adjcell = IntVector(i, j+1, k-1);       // TOP_BK
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[BOT_FR];
    adjcell = IntVector(i+1, j-1, k);       // BOT_R
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[TOP_L];
    adjcell = IntVector(i, j-1, k+1);       // BOT_FR
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[TOP_BK];
    adjcell = IntVector(i-1, j-1, k);       // BOT_L
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[TOP_R];
    adjcell = IntVector(i, j-1, k-1);       // BOT_BK
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[TOP_FR];
    adjcell = IntVector(i+1, j, k-1);       // RIGHT_BK
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[LEFT_FR];
    adjcell = IntVector(i+1, j, k+1);       // RIGHT_FR
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[LEFT_BK];
    adjcell = IntVector(i-1, j, k-1);       // LEFT_BK
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[RIGHT_FR];
    adjcell = IntVector(i-1, j, k+1);       // LEFT_FR
    sum_q_influx_EF += q_CC[adjcell] * d_OFE[adjcell].d_eflux[RIGHT_BK];

    //__________________________________
    //   INFLUX: CORNER FLUX
    adjcell = IntVector(i+1, j+1, k-1);       // TOP_R_BK
    sum_q_influx_CF += q_CC[adjcell] * d_OFC[adjcell].d_cflux[BOT_L_FR];
    adjcell = IntVector(i+1, j+1, k+1);       // TOP_R_FR
    sum_q_influx_CF += q_CC[adjcell] * d_OFC[adjcell].d_cflux[BOT_L_BK];
    adjcell = IntVector(i-1, j+1, k-1);       // TOP_L_BK
    sum_q_influx_CF += q_CC[adjcell] * d_OFC[adjcell].d_cflux[BOT_R_FR];
    adjcell = IntVector(i-1, j+1, k+1);       // TOP_L_FR
    sum_q_influx_CF += q_CC[adjcell] * d_OFC[adjcell].d_cflux[BOT_R_BK];
    adjcell = IntVector(i+1, j-1, k-1);       // BOT_R_BK
    sum_q_influx_CF += q_CC[adjcell] * d_OFC[adjcell].d_cflux[TOP_L_FR];
    adjcell = IntVector(i+1, j-1, k+1);       // BOT_R_FR
    sum_q_influx_CF += q_CC[adjcell] * d_OFC[adjcell].d_cflux[TOP_L_BK];
    adjcell = IntVector(i-1, j-1, k-1); // BOT_L_BK
    sum_q_influx_CF += q_CC[adjcell] * d_OFC[adjcell].d_cflux[TOP_R_FR];
    adjcell = IntVector(i-1, j-1, k+1);       // BOT_L_FR
    sum_q_influx_CF += q_CC[adjcell] * d_OFC[adjcell].d_cflux[TOP_R_BK];

    //__________________________________
    //  Calculate the advected q at t + delta t
    q_advected[curcell] = - sum_q_outflux - sum_q_outflux_EF - sum_q_outflux_CF
                        + sum_q_influx  + sum_q_influx_EF  + sum_q_influx_CF;

  }

}
  
namespace Uintah {

  static MPI_Datatype makeMPI_eflux()
  {
    ASSERTEQ(sizeof(FirstOrderCEAdvector::eflux), sizeof(double)*12);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 12, 12, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(FirstOrderCEAdvector::eflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                              "FirstOrderCEAdvector::eflux", true, 
                              &makeMPI_eflux);
    }
    return td;
  }
  
  static MPI_Datatype makeMPI_cflux()
  {
    ASSERTEQ(sizeof(FirstOrderCEAdvector::cflux), sizeof(double)*8);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 8, 8, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(FirstOrderCEAdvector::cflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                              "FirstOrderCEAdvector::cflux", true, 
                              &makeMPI_cflux);
    }
    return td;
  }
  
}

namespace SCIRun {

void swapbytes( Uintah::FirstOrderCEAdvector::eflux& e) {
  double *p = e.d_eflux;
  SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
}
  
void swapbytes( Uintah::FirstOrderCEAdvector::cflux& c) {
  double *p = c.d_cflux;
  SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
}

} // namespace SCIRun
