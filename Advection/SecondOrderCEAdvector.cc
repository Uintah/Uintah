#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderCEAdvector.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Exceptions/OutFluxVolume.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <iostream>

#define d_SMALL_NUM 1.0e-100

using namespace Uintah;

SecondOrderCEAdvector::SecondOrderCEAdvector()
{
  OFE_CCLabel = 0;
  OFC_CCLabel = 0;
}

SecondOrderCEAdvector::SecondOrderCEAdvector(DataWarehouse* new_dw, 
                                          const Patch* patch)
  :   d_advector(new_dw,patch)
{


  OFE_CCLabel = VarLabel::create("OFE_CC",
                             CCVariable<eflux>::getTypeDescription());
  OFC_CCLabel = VarLabel::create("OFC_CC",
                             CCVariable<cflux>::getTypeDescription());
  new_dw->allocateTemporary(d_OFE,  patch, Ghost::AroundCells,1);
  new_dw->allocateTemporary(d_OFC,  patch, Ghost::AroundCells,1);
  
  new_dw->allocateTemporary(r_out_x_EF,   patch, Ghost::AroundCells,1);
  new_dw->allocateTemporary(r_out_y_EF,   patch, Ghost::AroundCells,1);
  new_dw->allocateTemporary(r_out_z_EF,   patch, Ghost::AroundCells,1);
    
}


SecondOrderCEAdvector::~SecondOrderCEAdvector()
{
  VarLabel::destroy(OFE_CCLabel);
  VarLabel::destroy(OFC_CCLabel);
}

SecondOrderCEAdvector* SecondOrderCEAdvector::clone(DataWarehouse* new_dw,
                                   const Patch* patch)
{
  return scinew SecondOrderCEAdvector(new_dw,patch);
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

void SecondOrderCEAdvector::inFluxOutFluxVolume(
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
  double delX_tmp, delY_tmp,   delZ_tmp;
  double delX = dx.x(), delY = dx.y(), delZ = dx.z();
  double r_x, r_y, r_z, rx_R, rx_L, ry_TOP, ry_BOT, rz_FR, rz_BK;

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

    // Edge outflux terms
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
    //   Corner outflux terms
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
    
    //__________________________________
    //   FOR EACH SLAB 
       
      r_x = delX_left/2.0   -  delX_right/2.0;
      r_y = delY_bottom/2.0 - delY_top/2.0;
      r_z = delZ_back/2.0   - delZ_front/2.0;
      
      d_advector.r_out_x[curcell].d_fflux[FOA::RIGHT] = delX/2.0 - delX_right/2.0;
      d_advector.r_out_y[curcell].d_fflux[FOA::RIGHT] = r_y;
      d_advector.r_out_z[curcell].d_fflux[FOA::RIGHT] = r_z;
      
      d_advector.r_out_x[curcell].d_fflux[FOA::LEFT] = delX_left/2.0 - delX/2.0;
      d_advector.r_out_y[curcell].d_fflux[FOA::LEFT] = r_y;
      d_advector.r_out_z[curcell].d_fflux[FOA::LEFT] = r_z;
      
      d_advector.r_out_x[curcell].d_fflux[FOA::TOP] = r_x;
      d_advector.r_out_y[curcell].d_fflux[FOA::TOP] = delY/2.0 - delY_top/2.0;
      d_advector.r_out_z[curcell].d_fflux[FOA::TOP] = r_z;
     
      d_advector.r_out_x[curcell].d_fflux[FOA::BOTTOM] = r_x;
      d_advector.r_out_y[curcell].d_fflux[FOA::BOTTOM] = delY_bottom/2.0 - delY/2.0;
      d_advector.r_out_z[curcell].d_fflux[FOA::BOTTOM] = r_z;
      
      d_advector.r_out_x[curcell].d_fflux[FOA::FRONT] = r_x;
      d_advector.r_out_y[curcell].d_fflux[FOA::FRONT] = r_y;
      d_advector.r_out_z[curcell].d_fflux[FOA::FRONT] = delZ/2.0 - delZ_front/2.0;
     
      d_advector.r_out_x[curcell].d_fflux[FOA::BACK] = r_x;
      d_advector.r_out_y[curcell].d_fflux[FOA::BACK] = r_y;
      d_advector.r_out_z[curcell].d_fflux[FOA::BACK] = delZ_back/2.0 - delZ/2.0;
      
     //__________________________________
     //     FOR EACH EDGE
     
     rx_R   = delX/2.0        - delX_right/2.0;
     rx_L   = delX_left/2.0   - delX/2.0;
     ry_TOP = delY/2.0        - delY_top/2.0;
     ry_BOT = delY_bottom/2.0 - delY/2.0;
     rz_FR  = delZ/2.0        - delZ_front/2.0;
     rz_BK  = delZ_back/2.0   - delZ/2.0;
     
     r_out_x_EF[curcell].d_eflux[TOP_R] = rx_R;
     r_out_y_EF[curcell].d_eflux[TOP_R] = ry_TOP;  
     r_out_z_EF[curcell].d_eflux[TOP_R] = r_z;
     
     r_out_x_EF[curcell].d_eflux[TOP_FR] = r_x;
     r_out_y_EF[curcell].d_eflux[TOP_FR] = ry_TOP;  
     r_out_z_EF[curcell].d_eflux[TOP_FR] = rz_FR;
     
     r_out_x_EF[curcell].d_eflux[TOP_L] = rx_L;
     r_out_y_EF[curcell].d_eflux[TOP_L] = ry_TOP;  
     r_out_z_EF[curcell].d_eflux[TOP_L] = r_z;
     
     r_out_x_EF[curcell].d_eflux[TOP_BK] = r_x;
     r_out_y_EF[curcell].d_eflux[TOP_BK] = ry_TOP;  
     r_out_z_EF[curcell].d_eflux[TOP_BK] = rz_BK;
     
     r_out_x_EF[curcell].d_eflux[BOT_R] = rx_R;
     r_out_y_EF[curcell].d_eflux[BOT_R] = ry_BOT;  
     r_out_z_EF[curcell].d_eflux[BOT_R] = r_z;
     
     r_out_x_EF[curcell].d_eflux[BOT_FR] = r_x;
     r_out_y_EF[curcell].d_eflux[BOT_FR] = ry_BOT;  
     r_out_z_EF[curcell].d_eflux[BOT_FR] = rz_FR;
     
     r_out_x_EF[curcell].d_eflux[BOT_L] = rx_L;
     r_out_y_EF[curcell].d_eflux[BOT_L] = ry_BOT;  
     r_out_z_EF[curcell].d_eflux[BOT_L] = r_z;
     
     r_out_x_EF[curcell].d_eflux[BOT_BK] = r_x;
     r_out_y_EF[curcell].d_eflux[BOT_BK] = ry_BOT;  
     r_out_z_EF[curcell].d_eflux[BOT_BK] = rz_BK;
     
     r_out_x_EF[curcell].d_eflux[RIGHT_BK] = rx_R;
     r_out_y_EF[curcell].d_eflux[RIGHT_BK] = r_y;  
     r_out_z_EF[curcell].d_eflux[RIGHT_BK] = rz_BK;
     
     r_out_x_EF[curcell].d_eflux[RIGHT_FR] = rx_R;
     r_out_y_EF[curcell].d_eflux[RIGHT_FR] = r_y;  
     r_out_z_EF[curcell].d_eflux[RIGHT_FR] = rz_FR;
     
     r_out_x_EF[curcell].d_eflux[LEFT_BK] = rx_L;
     r_out_y_EF[curcell].d_eflux[LEFT_BK] = r_y;  
     r_out_z_EF[curcell].d_eflux[LEFT_BK] = rz_BK;
     
     r_out_x_EF[curcell].d_eflux[LEFT_FR] = rx_L;
     r_out_y_EF[curcell].d_eflux[LEFT_FR] = r_y;  
     r_out_z_EF[curcell].d_eflux[LEFT_FR] = rz_FR;

    num_cells++;
    error_test +=(vol - total_fluxout)/fabs(vol- total_fluxout);
    
  } //cell iterator  
  //__________________________________
  // if total_fluxout > vol then 
  // find the cell and throw an exception.  
  if (fabs(error_test - num_cells) > 1.0e-2) {
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
 Function~  ICE::advectQSecond--ADVECTION:
 Purpose~   Calculate the advection of q_CC 
   
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
            
 Steps for each cell:      
- Compute q outflux and q influx for each cell.
- Finally sum the influx and outflux portions
       
 advect_preprocessing MUST be done prior to this function
 ---------------------------------------------------------------------  */

void SecondOrderCEAdvector::advectQ(const CCVariable<double>& q_CC,
                             const Patch* patch,
                             CCVariable<double>& q_advected,
			        DataWarehouse* new_dw)
{

  CCVariable<double> grad_lim;
  StaticArray<CCVariable<double> > q_OAFS(6), q_OAFE(12), q_OAFC(8);
  double unit = 1.0;
  double SN = d_SMALL_NUM;
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(grad_lim,   patch);
  
  for (int face = FOA::TOP; face <= FOA::BACK; face ++){
    new_dw->allocateTemporary(q_OAFS[face],   patch,gac,1);
  }
  for (int edge= TOP_R; edge <= LEFT_FR; edge++) {
    new_dw->allocateTemporary(q_OAFE[edge],   patch,gac,1);
  }
  for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++) {
    new_dw->allocateTemporary(q_OAFC[corner],   patch,gac,1);
  }

  gradientLimiter(q_CC, patch, grad_lim, unit, SN, new_dw);
  qAverageFlux(q_CC, patch, grad_lim, q_OAFS,  q_OAFE, q_OAFC, new_dw);
  advect(q_OAFS, q_OAFE, q_OAFC, patch, q_advected);

}


void SecondOrderCEAdvector::advectQ(const CCVariable<Vector>& q_CC,
                             const Patch* patch,
                             CCVariable<Vector>& q_advected,
			        DataWarehouse* new_dw)
{
  CCVariable<Vector> grad_lim;
  StaticArray<CCVariable<Vector> > q_OAFS(6), q_OAFE(12), q_OAFC(8);
  Vector unit(1.0,1.0,1.0);
  Vector SN(d_SMALL_NUM,d_SMALL_NUM,d_SMALL_NUM);
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(grad_lim,   patch);
  
  for (int face = FOA::TOP; face <= FOA::BACK; face ++){
    new_dw->allocateTemporary(q_OAFS[face],   patch,gac,1);
  }
  for (int edge= TOP_R; edge <= LEFT_FR; edge++) {
    new_dw->allocateTemporary(q_OAFE[edge],   patch,gac,1);
  }
  for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++) {
    new_dw->allocateTemporary(q_OAFC[corner],   patch,gac,1);
  }

  gradientLimiter(q_CC, patch, grad_lim, unit, SN, new_dw);
  qAverageFlux(q_CC, patch, grad_lim, q_OAFS,  q_OAFE, q_OAFC, new_dw);
  advect(q_OAFS, q_OAFE, q_OAFC, patch, q_advected);

}


template <class T> void SecondOrderCEAdvector::advect(StaticArray<CCVariable<T> >& q_OAFS,
                                              StaticArray<CCVariable<T> >& q_OAFE,
				                  StaticArray<CCVariable<T> >& q_OAFC,
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
    for(int face = FOA::TOP; face <= FOA::BACK; face++ )  {
      sum_q_outflux  += q_OAFS[face][curcell] * d_advector.d_OFS[curcell].d_fflux[face];
    }
    //__________________________________
    //  OUTFLUX: EDGE_FLUX
    for(int edge = TOP_R; edge <= LEFT_FR; edge++ )   {
      sum_q_outflux_EF +=  q_OAFE[edge][curcell]* d_OFE[curcell].d_eflux[edge];
    }
    //__________________________________
    //  OUTFLUX: CORNER FLUX
    for(int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
      sum_q_outflux_CF +=  q_OAFC[corner][curcell] * d_OFC[curcell].d_cflux[corner];
    } 

    //__________________________________
    //  INFLUX: SLABS
    adjcell = IntVector(i, j+1, k);       // TOP
    sum_q_influx  += q_OAFS[FOA::BOTTOM][adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::BOTTOM];
    adjcell = IntVector(i, j-1, k);       // BOTTOM
    sum_q_influx  += q_OAFS[FOA::TOP][adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::TOP];
    adjcell = IntVector(i+1, j, k);       // RIGHT
    sum_q_influx  += q_OAFS[FOA::LEFT][adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::LEFT];
    adjcell = IntVector(i-1, j, k);       // LEFT
    sum_q_influx  += q_OAFS[FOA::RIGHT][adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::RIGHT];
    adjcell = IntVector(i, j, k+1);       // FRONT
    sum_q_influx  += q_OAFS[FOA::BACK][adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::BACK];
    adjcell = IntVector(i, j, k-1);       // BACK
    sum_q_influx  += q_OAFS[FOA::FRONT][adjcell] * d_advector.d_OFS[adjcell].d_fflux[FOA::FRONT];
    //__________________________________
    //  INFLUX: EDGES
    adjcell = IntVector(i+1, j+1, k);       // TOP_R
    sum_q_influx_EF += q_OAFE[BOT_L][adjcell] * d_OFE[adjcell].d_eflux[BOT_L];
    adjcell = IntVector(i, j+1, k+1);   // TOP_FR
    sum_q_influx_EF += q_OAFE[BOT_BK][adjcell] * d_OFE[adjcell].d_eflux[BOT_BK];
    adjcell = IntVector(i-1, j+1, k);       // TOP_L
    sum_q_influx_EF += q_OAFE[BOT_R][adjcell] * d_OFE[adjcell].d_eflux[BOT_R];
    adjcell = IntVector(i, j+1, k-1);       // TOP_BK
    sum_q_influx_EF += q_OAFE[BOT_FR][adjcell] * d_OFE[adjcell].d_eflux[BOT_FR];
    adjcell = IntVector(i+1, j-1, k);       // BOT_R
    sum_q_influx_EF += q_OAFE[TOP_L][adjcell] * d_OFE[adjcell].d_eflux[TOP_L];
    adjcell = IntVector(i, j-1, k+1);       // BOT_FR
    sum_q_influx_EF += q_OAFE[TOP_BK][adjcell] * d_OFE[adjcell].d_eflux[TOP_BK];
    adjcell = IntVector(i-1, j-1, k);       // BOT_L
    sum_q_influx_EF += q_OAFE[TOP_R][adjcell] * d_OFE[adjcell].d_eflux[TOP_R];
    adjcell = IntVector(i, j-1, k-1);       // BOT_BK
    sum_q_influx_EF += q_OAFE[TOP_FR][adjcell] * d_OFE[adjcell].d_eflux[TOP_FR];
    adjcell = IntVector(i+1, j, k-1);       // RIGHT_BK
    sum_q_influx_EF += q_OAFE[LEFT_FR][adjcell] * d_OFE[adjcell].d_eflux[LEFT_FR];
    adjcell = IntVector(i+1, j, k+1);       // RIGHT_FR
    sum_q_influx_EF += q_OAFE[LEFT_BK][adjcell] * d_OFE[adjcell].d_eflux[LEFT_BK];
    adjcell = IntVector(i-1, j, k-1);       // LEFT_BK
    sum_q_influx_EF += q_OAFE[RIGHT_FR][adjcell] * d_OFE[adjcell].d_eflux[RIGHT_FR];
    adjcell = IntVector(i-1, j, k+1);       // LEFT_FR
    sum_q_influx_EF += q_OAFE[RIGHT_BK][adjcell] * d_OFE[adjcell].d_eflux[RIGHT_BK];

    //__________________________________
    //   INFLUX: CORNER FLUX
    adjcell = IntVector(i+1, j+1, k-1);       // TOP_R_BK
    sum_q_influx_CF += q_OAFC[BOT_L_FR][adjcell] * d_OFC[adjcell].d_cflux[BOT_L_FR];
    adjcell = IntVector(i+1, j+1, k+1);       // TOP_R_FR
    sum_q_influx_CF += q_OAFC[BOT_L_BK][adjcell] * d_OFC[adjcell].d_cflux[BOT_L_BK];
    adjcell = IntVector(i-1, j+1, k-1);       // TOP_L_BK
    sum_q_influx_CF += q_OAFC[BOT_R_FR][adjcell] * d_OFC[adjcell].d_cflux[BOT_R_FR];
    adjcell = IntVector(i-1, j+1, k+1);       // TOP_L_FR
    sum_q_influx_CF += q_OAFC[BOT_R_BK][adjcell] * d_OFC[adjcell].d_cflux[BOT_R_BK];
    adjcell = IntVector(i+1, j-1, k-1);       // BOT_R_BK
    sum_q_influx_CF += q_OAFC[TOP_L_FR][adjcell] * d_OFC[adjcell].d_cflux[TOP_L_FR];
    adjcell = IntVector(i+1, j-1, k+1);       // BOT_R_FR
    sum_q_influx_CF += q_OAFC[TOP_L_BK][adjcell] * d_OFC[adjcell].d_cflux[TOP_L_BK];
    adjcell = IntVector(i-1, j-1, k-1); // BOT_L_BK
    sum_q_influx_CF += q_OAFC[TOP_R_FR][adjcell] * d_OFC[adjcell].d_cflux[TOP_R_FR];
    adjcell = IntVector(i-1, j-1, k+1);       // BOT_L_FR
    sum_q_influx_CF += q_OAFC[TOP_R_BK][adjcell] * d_OFC[adjcell].d_cflux[TOP_R_BK];

    //__________________________________
    //  Calculate the advected q at t + delta t
    q_advected[curcell] = - sum_q_outflux - sum_q_outflux_EF - sum_q_outflux_CF
                        + sum_q_influx  + sum_q_influx_EF  + sum_q_influx_CF;
  }

}

template <class T> void SecondOrderCEAdvector::gradientLimiter(const CCVariable<T>& q_CC,
                                              const Patch* patch,
                                              CCVariable<T>& grad_lim,
						    T unit, T SN,
			                         DataWarehouse* new_dw)
  
{
  T  frac,temp, zero(0.);
  T  grad_lim_max, grad_lim_min;

  CCVariable<T> q_vrtx_max;
  CCVariable<T> q_vrtx_min;
  CCVariable<T> q_CC_max;
  CCVariable<T> q_CC_min;
  
  new_dw->allocateTemporary(q_CC_max,   patch);
  new_dw->allocateTemporary(q_CC_min,   patch);
  new_dw->allocateTemporary(q_vrtx_max,   patch);
  new_dw->allocateTemporary(q_vrtx_min,   patch);

  q_CCMaxMin(q_CC, patch, q_CC_max, q_CC_min);
  q_vertexMaxMin(q_CC, patch, q_vrtx_max, q_vrtx_min, new_dw);

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
     IntVector curcell = *iter;

    frac = (q_CC_max[curcell]-q_CC[curcell]+SN)/(q_vrtx_max[curcell]-q_CC[curcell]+SN);
    grad_lim_max = Max(zero, frac);

    frac = (q_CC_min[curcell]-q_CC[curcell]+SN)/(q_vrtx_min[curcell]-q_CC[curcell]+SN);
    grad_lim_min = Max(zero, frac);
    
    temp = Min(unit, grad_lim_max);
    temp = Min(temp, grad_lim_min);
    grad_lim[curcell] = temp;
 
  }
}

template <class T> void SecondOrderCEAdvector::qAverageFlux(const CCVariable<T>& q_CC,
                                              const Patch* patch,
						    CCVariable<T>& grad_lim,
						    StaticArray<CCVariable<T> >& q_OAFS,
				                  StaticArray<CCVariable<T> >& q_OAFE,
				                  StaticArray<CCVariable<T> >& q_OAFC,						    
			                         DataWarehouse* new_dw)
  
{
 CCVariable<T> q_grad_x,q_grad_y,q_grad_z;

 new_dw->allocateTemporary(q_grad_x,   patch);
 new_dw->allocateTemporary(q_grad_y,   patch);
 new_dw->allocateTemporary(q_grad_z,   patch);
 
 gradQ(q_CC, patch, q_grad_x, q_grad_y, q_grad_z);
      
  const IntVector gc(1,1,1);

  for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
    IntVector curcell = *iter;
         for (int face = FOA::TOP; face <= FOA::BACK; face ++){
	    q_OAFS[face][curcell] = q_CC[curcell];
	  }
	  for (int edge = TOP_R; edge <= LEFT_FR; edge++) {
	    q_OAFE[edge][curcell] = q_CC[curcell];
	  }
	  for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
	    q_OAFC[corner][curcell] = q_CC[curcell];
         }
    }

 for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    IntVector curcell = *iter;
    
    q_grad_x[curcell] = q_grad_x[curcell] * grad_lim[curcell];
    q_grad_y[curcell] = q_grad_y[curcell] * grad_lim[curcell];
    q_grad_z[curcell] = q_grad_z[curcell] * grad_lim[curcell];

    
    //__________________________________
    //  OUTAVERAGEFLUX: SLAB
    for (int face = FOA::TOP; face <= FOA::BACK; face ++){ 
      q_OAFS[face][curcell] = q_CC[curcell] + q_grad_x[curcell] * d_advector.r_out_x[curcell].d_fflux[face] + 
                              q_grad_y[curcell] * d_advector.r_out_y[curcell].d_fflux[face] + 
				  q_grad_z[curcell] * d_advector.r_out_z[curcell].d_fflux[face];
    }
  				  
    //__________________________________
    //  OUTAVERAGEFLUX: EDGE
    for (int edge = TOP_R; edge <= LEFT_FR; edge++ )   { 
      q_OAFE[edge][curcell] = q_CC[curcell] + q_grad_x[curcell] * r_out_x_EF[curcell].d_eflux[edge] + 
                              q_grad_y[curcell] * r_out_y_EF[curcell].d_eflux[edge] + 
				  q_grad_z[curcell] * r_out_z_EF[curcell].d_eflux[edge];
    }
    
    //__________________________________
    //  OUTAVERAGEFLUX: CORNER
    for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  { 
      q_OAFC[corner][curcell] = q_CC[curcell];
    }
  }
   
}

template <class T> void SecondOrderCEAdvector::q_CCMaxMin(const CCVariable<T>& q_CC,
                                              const Patch* patch,
						    CCVariable<T>& q_CC_max, 
						    CCVariable<T>& q_CC_min)
{  
     T q_CC_max_tmp, q_CC_min_tmp;
     IntVector adjcell;
     for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
       IntVector curcell = *iter;
	int i = curcell.x();
       int j = curcell.y();
       int k = curcell.z();
       
	adjcell = IntVector(i-1, j, k);
	q_CC_max_tmp = q_CC[adjcell];
	q_CC_min_tmp = q_CC[adjcell];
	adjcell = IntVector(i+1, j, k);
	q_CC_max_tmp = Max(q_CC_max_tmp, q_CC[adjcell]);
       q_CC_min_tmp = Min(q_CC_min_tmp, q_CC[adjcell]);
	adjcell = IntVector(i, j-1, k);
	q_CC_max_tmp = Max(q_CC_max_tmp, q_CC[adjcell]);
       q_CC_min_tmp = Min(q_CC_min_tmp, q_CC[adjcell]);
       adjcell = IntVector(i, j+1, k);
	q_CC_max_tmp = Max(q_CC_max_tmp, q_CC[adjcell]);
       q_CC_min_tmp = Min(q_CC_min_tmp, q_CC[adjcell]);
	adjcell = IntVector(i, j, k-1);
	q_CC_max_tmp = Max(q_CC_max_tmp, q_CC[adjcell]);
       q_CC_min_tmp = Min(q_CC_min_tmp, q_CC[adjcell]);
	adjcell = IntVector(i, j, k+1);
	q_CC_max_tmp = Max(q_CC_max_tmp, q_CC[adjcell]);
       q_CC_min_tmp = Min(q_CC_min_tmp, q_CC[adjcell]);
	
	q_CC_max[curcell] = q_CC_max_tmp;
	q_CC_min[curcell] = q_CC_min_tmp;

      }       
      
}						    


template <class T> void SecondOrderCEAdvector::q_vertexMaxMin(const CCVariable<T>& q_CC,
                                              const Patch* patch,
						    CCVariable<T>& q_vrtx_max, 
						    CCVariable<T>& q_vrtx_min,
			                         DataWarehouse* new_dw)
{

     T q_vrtx1, q_vrtx2,q_vrtx3,q_vrtx4,q_vrtx5,q_vrtx6,q_vrtx7,q_vrtx8;
     T q_vrtx_tmp_max, q_vrtx_tmp_min;
     
     Vector dx = patch->dCell();
     double delX_2 = dx.x()/2.0;
     double delY_2 = dx.y()/2.0;
     double delZ_2 = dx.z()/2.0;
  
     CCVariable<T> q_grad_x,q_grad_y,q_grad_z;
 
     new_dw->allocateTemporary(q_grad_x,   patch);
     new_dw->allocateTemporary(q_grad_y,   patch);
     new_dw->allocateTemporary(q_grad_z,   patch);
     
     gradQ(q_CC, patch, q_grad_x, q_grad_y, q_grad_z);

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector curcell = *iter;
      
      q_vrtx1 = q_CC[curcell]+q_grad_x[curcell]*(-delX_2)+q_grad_y[curcell]*(-delY_2)+q_grad_z[curcell]*(-delZ_2);
      q_vrtx2 = q_CC[curcell]+q_grad_x[curcell]*(delX_2)+q_grad_y[curcell]*(-delY_2)+q_grad_z[curcell]*(-delZ_2);	
      q_vrtx3 = q_CC[curcell]+q_grad_x[curcell]*(delX_2)+q_grad_y[curcell]*(delY_2)+q_grad_z[curcell]*(-delZ_2);      
      q_vrtx4 = q_CC[curcell]+q_grad_x[curcell]*(-delX_2)+q_grad_y[curcell]*(delY_2)+q_grad_z[curcell]*(-delZ_2);	
      q_vrtx5 = q_CC[curcell]+q_grad_x[curcell]*(-delX_2)+q_grad_y[curcell]*(-delY_2)+q_grad_z[curcell]*(delZ_2);	
      q_vrtx6 = q_CC[curcell]+q_grad_x[curcell]*(delX_2)+q_grad_y[curcell]*(-delY_2)+q_grad_z[curcell]*(delZ_2);      
      q_vrtx7 = q_CC[curcell]+q_grad_x[curcell]*(-delX_2)+q_grad_y[curcell]*(delY_2)+q_grad_z[curcell]*(delZ_2);
      q_vrtx8 = q_CC[curcell]+q_grad_x[curcell]*(delX_2)+q_grad_y[curcell]*(delY_2)+q_grad_z[curcell]*(delZ_2);      
            
      q_vrtx_tmp_max = Max(q_vrtx1,q_vrtx2);
      q_vrtx_tmp_min = Min(q_vrtx1,q_vrtx2);
      q_vrtx_tmp_max = Max(q_vrtx3,q_vrtx_tmp_max);
      q_vrtx_tmp_min = Min(q_vrtx3,q_vrtx_tmp_min);
      q_vrtx_tmp_max = Max(q_vrtx4,q_vrtx_tmp_max);
      q_vrtx_tmp_min = Min(q_vrtx4,q_vrtx_tmp_min);
      q_vrtx_tmp_max = Max(q_vrtx5,q_vrtx_tmp_max);
      q_vrtx_tmp_min = Min(q_vrtx5,q_vrtx_tmp_min);
      q_vrtx_tmp_max = Max(q_vrtx6,q_vrtx_tmp_max);
      q_vrtx_tmp_min = Min(q_vrtx6,q_vrtx_tmp_min);
      q_vrtx_tmp_max = Max(q_vrtx7,q_vrtx_tmp_max);
      q_vrtx_tmp_min = Min(q_vrtx7,q_vrtx_tmp_min);
      q_vrtx_tmp_max = Max(q_vrtx8,q_vrtx_tmp_max);
      q_vrtx_tmp_min = Min(q_vrtx8,q_vrtx_tmp_min);
                  
      q_vrtx_max[curcell] = q_vrtx_tmp_max;
      q_vrtx_min[curcell] = q_vrtx_tmp_min; 

    }    
}
template <class T> void SecondOrderCEAdvector::gradQ(const CCVariable<T>& q_CC,
                                              const Patch* patch,
						    CCVariable<T>& q_grad_x,
						    CCVariable<T>& q_grad_y,
						    CCVariable<T>& q_grad_z)
{  
    Vector dx = patch->dCell();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    IntVector adjcell1, adjcell2;
    
   for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    IntVector curcell = *iter;
       int i = curcell.x();
       int j = curcell.y();
       int k = curcell.z();
       
       adjcell1 = IntVector(i+1, j, k);
       adjcell2 = IntVector(i-1, j, k);
	q_grad_x[curcell] = (q_CC[adjcell1] - q_CC[adjcell2])/(2.0*delX);
	
	adjcell1 = IntVector(i, j+1, k);
       adjcell2 = IntVector(i, j-1, k);
	q_grad_y[curcell] = (q_CC[adjcell1] - q_CC[adjcell2])/(2.0*delY);

       adjcell1 = IntVector(i, j, k+1);
       adjcell2 = IntVector(i, j, k-1);
	q_grad_z[curcell] = (q_CC[adjcell1] - q_CC[adjcell2])/(2.0*delZ);

   }
   
}

  
namespace Uintah {

  static MPI_Datatype makeMPI_eflux()
  {
    ASSERTEQ(sizeof(SecondOrderCEAdvector::eflux), sizeof(double)*12);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 12, 12, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(SecondOrderCEAdvector::eflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                              "SecondOrderCEAdvector::eflux", true, 
                              &makeMPI_eflux);
    }
    return td;
  }
  
  static MPI_Datatype makeMPI_cflux()
  {
    ASSERTEQ(sizeof(SecondOrderCEAdvector::cflux), sizeof(double)*8);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 8, 8, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(SecondOrderCEAdvector::cflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                              "SecondOrderCEAdvector::cflux", true, 
                              &makeMPI_cflux);
    }
    return td;
  }
  
}

namespace SCIRun {

void swapbytes( Uintah::SecondOrderCEAdvector::eflux& e) {
  double *p = e.d_eflux;
  SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
}
  
void swapbytes( Uintah::SecondOrderCEAdvector::cflux& c) {
  double *p = c.d_cflux;
  SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
}

} // namespace SCIRun
