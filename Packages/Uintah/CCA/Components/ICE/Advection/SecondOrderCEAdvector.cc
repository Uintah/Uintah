#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderCEAdvector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderBase.h>
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
{
  OFS_CCLabel = VarLabel::create("OFS_CC",
                             CCVariable<fflux>::getTypeDescription());
  OFE_CCLabel = VarLabel::create("OFE_CC",
                             CCVariable<eflux>::getTypeDescription());
  OFC_CCLabel = VarLabel::create("OFC_CC",
                             CCVariable<cflux>::getTypeDescription());
                             
  new_dw->allocateTemporary(d_OFS,  patch, Ghost::AroundCells,1);
  new_dw->allocateTemporary(d_OFE,  patch, Ghost::AroundCells,1);
  new_dw->allocateTemporary(d_OFC,  patch, Ghost::AroundCells,1);
  
  new_dw->allocateTemporary(r_out_x,      patch, Ghost::AroundCells,1);  
  new_dw->allocateTemporary(r_out_y,      patch, Ghost::AroundCells,1);  
  new_dw->allocateTemporary(r_out_z,      patch, Ghost::AroundCells,1);  
  
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
  double error_test = 0.0;
  int    num_cells = 0;
  
  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getExtraCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
  
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    IntVector c = *iter;
    delY_top    = std::max(0.0, (vvel_FC[c+IntVector(0,1,0)] * delT));
    delY_bottom = std::max(0.0,-(vvel_FC[c+IntVector(0,0,0)] * delT));
    delX_right  = std::max(0.0, (uvel_FC[c+IntVector(1,0,0)] * delT));
    delX_left   = std::max(0.0,-(uvel_FC[c+IntVector(0,0,0)] * delT));
    delZ_front  = std::max(0.0, (wvel_FC[c+IntVector(0,0,1)] * delT));
    delZ_back   = std::max(0.0,-(wvel_FC[c+IntVector(0,0,0)] * delT));
    
    delX_tmp    = dx.x() - delX_right - delX_left;
    delY_tmp    = dx.y() - delY_top   - delY_bottom;
    delZ_tmp    = dx.z() - delZ_front - delZ_back;
    
    //__________________________________
    //   SLAB outfluxes
    d_OFS[c].d_fflux[TOP]   = delY_top   * delX_tmp * delZ_tmp;
    d_OFS[c].d_fflux[BOTTOM]= delY_bottom* delX_tmp * delZ_tmp;
    d_OFS[c].d_fflux[RIGHT] = delX_right * delY_tmp * delZ_tmp;
    d_OFS[c].d_fflux[LEFT]  = delX_left  * delY_tmp * delZ_tmp;
    d_OFS[c].d_fflux[FRONT] = delZ_front * delX_tmp * delY_tmp;
    d_OFS[c].d_fflux[BACK]  = delZ_back  * delX_tmp * delY_tmp;

    // Edge outflux terms
    d_OFE[c].d_eflux[TOP_R]     = delY_top      * delX_right * delZ_tmp;
    d_OFE[c].d_eflux[TOP_FR]    = delY_top      * delX_tmp   * delZ_front;
    d_OFE[c].d_eflux[TOP_L]     = delY_top      * delX_left  * delZ_tmp;
    d_OFE[c].d_eflux[TOP_BK]    = delY_top      * delX_tmp   * delZ_back;
    
    d_OFE[c].d_eflux[BOT_R]     = delY_bottom   * delX_right * delZ_tmp;
    d_OFE[c].d_eflux[BOT_FR]    = delY_bottom   * delX_tmp   * delZ_front;
    d_OFE[c].d_eflux[BOT_L]     = delY_bottom   * delX_left  * delZ_tmp;
    d_OFE[c].d_eflux[BOT_BK]    = delY_bottom   * delX_tmp   * delZ_back;
    
    d_OFE[c].d_eflux[RIGHT_BK]  = delY_tmp      * delX_right * delZ_back;
    d_OFE[c].d_eflux[RIGHT_FR]  = delY_tmp      * delX_right * delZ_front;
    
    d_OFE[c].d_eflux[LEFT_BK]   = delY_tmp      * delX_left  * delZ_back;
    d_OFE[c].d_eflux[LEFT_FR]   = delY_tmp      * delX_left  * delZ_front;
    
    //__________________________________
    //   Corner outflux terms
    d_OFC[c].d_cflux[TOP_R_BK]  = delY_top      * delX_right * delZ_back;
    d_OFC[c].d_cflux[TOP_R_FR]  = delY_top      * delX_right * delZ_front;
    d_OFC[c].d_cflux[TOP_L_BK]  = delY_top      * delX_left  * delZ_back;
    d_OFC[c].d_cflux[TOP_L_FR]  = delY_top      * delX_left  * delZ_front;
    
    d_OFC[c].d_cflux[BOT_R_BK]  = delY_bottom   * delX_right * delZ_back;
    d_OFC[c].d_cflux[BOT_R_FR]  = delY_bottom   * delX_right * delZ_front;
    d_OFC[c].d_cflux[BOT_L_BK]  = delY_bottom   * delX_left  * delZ_back;
    d_OFC[c].d_cflux[BOT_L_FR]  = delY_bottom   * delX_left  * delZ_front;

    //__________________________________
    //  Bullet proofing
    double total_fluxout = 0.0;
    for(int face = TOP; face <= BACK; 
       face++ )  {
      total_fluxout  += d_OFS[c].d_fflux[face];
    }
    for(int edge = TOP_R; edge <= LEFT_FR; edge++ )  {
      total_fluxout  += d_OFE[c].d_eflux[edge];
    }
    for(int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
      total_fluxout  += d_OFC[c].d_cflux[corner];
    }
    
    //__________________________________
    //   FOR EACH SLAB 
      r_x = delX_left/2.0   -  delX_right/2.0;
      r_y = delY_bottom/2.0 - delY_top/2.0;
      r_z = delZ_back/2.0   - delZ_front/2.0;
      
      r_out_x[c].d_fflux[RIGHT] = delX/2.0 - delX_right/2.0;
      r_out_y[c].d_fflux[RIGHT] = r_y;
      r_out_z[c].d_fflux[RIGHT] = r_z;
      
      r_out_x[c].d_fflux[LEFT] = delX_left/2.0 - delX/2.0;
      r_out_y[c].d_fflux[LEFT] = r_y;
      r_out_z[c].d_fflux[LEFT] = r_z;
      
      r_out_x[c].d_fflux[TOP] = r_x;
      r_out_y[c].d_fflux[TOP] = delY/2.0 - delY_top/2.0;
      r_out_z[c].d_fflux[TOP] = r_z;
     
      r_out_x[c].d_fflux[BOTTOM] = r_x;
      r_out_y[c].d_fflux[BOTTOM] = delY_bottom/2.0 - delY/2.0;
      r_out_z[c].d_fflux[BOTTOM] = r_z;
      
      r_out_x[c].d_fflux[FRONT] = r_x;
      r_out_y[c].d_fflux[FRONT] = r_y;
      r_out_z[c].d_fflux[FRONT] = delZ/2.0 - delZ_front/2.0;
     
      r_out_x[c].d_fflux[BACK] = r_x;
      r_out_y[c].d_fflux[BACK] = r_y;
      r_out_z[c].d_fflux[BACK] = delZ_back/2.0 - delZ/2.0;
      
     //__________________________________
     //     FOR EACH EDGE 
     rx_R   = delX/2.0        - delX_right/2.0;
     rx_L   = delX_left/2.0   - delX/2.0;
     ry_TOP = delY/2.0        - delY_top/2.0;
     ry_BOT = delY_bottom/2.0 - delY/2.0;
     rz_FR  = delZ/2.0        - delZ_front/2.0;
     rz_BK  = delZ_back/2.0   - delZ/2.0;
     
     r_out_x_EF[c].d_eflux[TOP_R] = rx_R;
     r_out_y_EF[c].d_eflux[TOP_R] = ry_TOP;  
     r_out_z_EF[c].d_eflux[TOP_R] = r_z;
     
     r_out_x_EF[c].d_eflux[TOP_FR] = r_x;
     r_out_y_EF[c].d_eflux[TOP_FR] = ry_TOP;  
     r_out_z_EF[c].d_eflux[TOP_FR] = rz_FR;
     
     r_out_x_EF[c].d_eflux[TOP_L] = rx_L;
     r_out_y_EF[c].d_eflux[TOP_L] = ry_TOP;  
     r_out_z_EF[c].d_eflux[TOP_L] = r_z;
     
     r_out_x_EF[c].d_eflux[TOP_BK] = r_x;
     r_out_y_EF[c].d_eflux[TOP_BK] = ry_TOP;  
     r_out_z_EF[c].d_eflux[TOP_BK] = rz_BK;
     
     r_out_x_EF[c].d_eflux[BOT_R] = rx_R;
     r_out_y_EF[c].d_eflux[BOT_R] = ry_BOT;  
     r_out_z_EF[c].d_eflux[BOT_R] = r_z;
     
     r_out_x_EF[c].d_eflux[BOT_FR] = r_x;
     r_out_y_EF[c].d_eflux[BOT_FR] = ry_BOT;  
     r_out_z_EF[c].d_eflux[BOT_FR] = rz_FR;
     
     r_out_x_EF[c].d_eflux[BOT_L] = rx_L;
     r_out_y_EF[c].d_eflux[BOT_L] = ry_BOT;  
     r_out_z_EF[c].d_eflux[BOT_L] = r_z;
     
     r_out_x_EF[c].d_eflux[BOT_BK] = r_x;
     r_out_y_EF[c].d_eflux[BOT_BK] = ry_BOT;  
     r_out_z_EF[c].d_eflux[BOT_BK] = rz_BK;
     
     r_out_x_EF[c].d_eflux[RIGHT_BK] = rx_R;
     r_out_y_EF[c].d_eflux[RIGHT_BK] = r_y;  
     r_out_z_EF[c].d_eflux[RIGHT_BK] = rz_BK;
     
     r_out_x_EF[c].d_eflux[RIGHT_FR] = rx_R;
     r_out_y_EF[c].d_eflux[RIGHT_FR] = r_y;  
     r_out_z_EF[c].d_eflux[RIGHT_FR] = rz_FR;
     
     r_out_x_EF[c].d_eflux[LEFT_BK] = rx_L;
     r_out_y_EF[c].d_eflux[LEFT_BK] = r_y;  
     r_out_z_EF[c].d_eflux[LEFT_BK] = rz_BK;
     
     r_out_x_EF[c].d_eflux[LEFT_FR] = rx_L;
     r_out_y_EF[c].d_eflux[LEFT_FR] = r_y;  
     r_out_z_EF[c].d_eflux[LEFT_FR] = rz_FR;

    num_cells++;
    error_test +=(vol - total_fluxout)/fabs(vol- total_fluxout);
    
  } //cell iterator  
  //__________________________________
  // if total_fluxout > vol then 
  // find the cell and throw an exception.  
  if (fabs(error_test - num_cells) > 1.0e-2) {
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      IntVector c = *iter; 
      double total_fluxout = 0.0;
      for(int face = TOP; face <= BACK; face++ )  {
        total_fluxout  += d_OFS[c].d_fflux[face];
      }
      for(int edge = TOP_R; edge <= LEFT_FR; edge++ )  {
        total_fluxout  += d_OFE[c].d_eflux[edge];
      }
      for(int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
        total_fluxout  += d_OFC[c].d_cflux[corner];
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
  new_dw->allocateTemporary(grad_lim,         patch, gac, 1);
  
  for (int face = TOP; face <= BACK; face ++){
    new_dw->allocateTemporary(q_OAFS[face],   patch, gac,1);
  }
  for (int edge= TOP_R; edge <= LEFT_FR; edge++) {
    new_dw->allocateTemporary(q_OAFE[edge],    patch, gac,1);
  }
  for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++) {
    new_dw->allocateTemporary(q_OAFC[corner],  patch, gac,1);
  }
  
  CCVariable<double> q_grad_x,q_grad_y,q_grad_z;
  new_dw->allocateTemporary(q_grad_x, patch, gac, 1);  
  new_dw->allocateTemporary(q_grad_y, patch, gac, 1);  
  new_dw->allocateTemporary(q_grad_z, patch, gac, 1);  
  
  //__________________________________
  gradQ(q_CC, patch, q_grad_x, q_grad_y, q_grad_z);
    
  gradientLimiter(q_CC, patch, grad_lim, q_grad_x, q_grad_y, q_grad_z,
                  unit, SN, new_dw);
  qAverageFlux(q_CC, patch, grad_lim,  q_grad_x, q_grad_y, q_grad_z,
               q_OAFS,  q_OAFE, q_OAFC, new_dw);
              
  advect(q_OAFS, q_OAFE, q_OAFC, patch, q_advected);

}

//______________________________________________________________________
//       V E C T O R   V E R S I O N
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
  new_dw->allocateTemporary(grad_lim,         patch,gac,1);
  
  for (int face = TOP; face <= BACK; face ++){
    new_dw->allocateTemporary(q_OAFS[face],   patch,gac,1);
  }
  for (int edge= TOP_R; edge <= LEFT_FR; edge++) {
    new_dw->allocateTemporary(q_OAFE[edge],   patch,gac,1);
  }
  for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++) {
    new_dw->allocateTemporary(q_OAFC[corner], patch,gac,1);
  }
  
  CCVariable<Vector> q_grad_x,q_grad_y,q_grad_z;
  new_dw->allocateTemporary(q_grad_x, patch, gac, 1);  
  new_dw->allocateTemporary(q_grad_y, patch, gac, 1);  
  new_dw->allocateTemporary(q_grad_z, patch, gac, 1);  

  //__________________________________
  gradQ(q_CC, patch, q_grad_x, q_grad_y, q_grad_z);  
  
  gradientLimiter(q_CC, patch, grad_lim, q_grad_x, q_grad_y, q_grad_z,
                  unit, SN, new_dw);
  qAverageFlux(q_CC, patch, grad_lim,  q_grad_x, q_grad_y, q_grad_z,
               q_OAFS,  q_OAFE, q_OAFC, new_dw);
              
  advect(q_OAFS, q_OAFE, q_OAFC, patch, q_advected);

}

//______________________________________________________________________
//
template <class T> void SecondOrderCEAdvector::advect(
                                            StaticArray<CCVariable<T> >& q_OAFS,
                                            StaticArray<CCVariable<T> >& q_OAFE,
				                StaticArray<CCVariable<T> >& q_OAFC,
                                            const Patch* patch,
                                            CCVariable<T>& q_advected)
{
  T sum_q_outflux(0.), sum_q_outflux_EF(0.), sum_q_outflux_CF(0.);
  T sum_q_influx(0.),  sum_q_influx_EF(0.), sum_q_influx_CF(0.);
  IntVector adjcell;
  
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    IntVector c = *iter;
    int i = c.x();
    int j = c.y();
    int k = c.z();
   
    sum_q_outflux      *= 0.0;
    sum_q_outflux_EF   *= 0.0;
    sum_q_outflux_CF   *= 0.0;
    sum_q_influx       *= 0.0;
    sum_q_influx_EF    *= 0.0;
    sum_q_influx_CF    *= 0.0;
    
    //__________________________________
    //  OUTFLUX: SLAB 
    for(int face = TOP; face <= BACK; face++ )  {
      sum_q_outflux  += q_OAFS[face][c] * d_OFS[c].d_fflux[face];
    }
    //__________________________________
    //  OUTFLUX: EDGE_FLUX
    for(int edge = TOP_R; edge <= LEFT_FR; edge++ )   {
      sum_q_outflux_EF +=  q_OAFE[edge][c]* d_OFE[c].d_eflux[edge];
    }
    //__________________________________
    //  OUTFLUX: CORNER FLUX
    for(int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
      sum_q_outflux_CF +=  q_OAFC[corner][c] * d_OFC[c].d_cflux[corner];
    } 

    //__________________________________
    //  INFLUX: SLABS
    adjcell = IntVector(i,   j+1, k);        // TOP
    sum_q_influx  += q_OAFS[BOTTOM][adjcell] * d_OFS[adjcell].d_fflux[BOTTOM];
    adjcell = IntVector(i,   j-1, k);        // BOTTOM
    sum_q_influx  += q_OAFS[TOP][adjcell]    * d_OFS[adjcell].d_fflux[TOP];
    adjcell = IntVector(i+1, j,   k);        // RIGHT
    sum_q_influx  += q_OAFS[LEFT][adjcell]   * d_OFS[adjcell].d_fflux[LEFT];
    adjcell = IntVector(i-1, j,   k);        // LEFT
    sum_q_influx  += q_OAFS[RIGHT][adjcell]  * d_OFS[adjcell].d_fflux[RIGHT];
    adjcell = IntVector(i,   j,   k+1);      // FRONT
    sum_q_influx  += q_OAFS[BACK][adjcell]   * d_OFS[adjcell].d_fflux[BACK];
    adjcell = IntVector(i,   j,   k-1);      // BACK
    sum_q_influx  += q_OAFS[FRONT][adjcell]  * d_OFS[adjcell].d_fflux[FRONT];
    
    //__________________________________
    //  INFLUX: EDGES
    adjcell = IntVector(i+1, j+1, k);       // TOP_R
    sum_q_influx_EF += q_OAFE[BOT_L][adjcell]    * d_OFE[adjcell].d_eflux[BOT_L];
    adjcell = IntVector(i,   j+1, k+1);     // TOP_FR
    sum_q_influx_EF += q_OAFE[BOT_BK][adjcell]   * d_OFE[adjcell].d_eflux[BOT_BK];
    adjcell = IntVector(i-1, j+1, k);       // TOP_L
    sum_q_influx_EF += q_OAFE[BOT_R][adjcell]    * d_OFE[adjcell].d_eflux[BOT_R];
    adjcell = IntVector(i,   j+1, k-1);     // TOP_BK
    sum_q_influx_EF += q_OAFE[BOT_FR][adjcell]   * d_OFE[adjcell].d_eflux[BOT_FR];
    adjcell = IntVector(i+1, j-1, k);       // BOT_R
    sum_q_influx_EF += q_OAFE[TOP_L][adjcell]    * d_OFE[adjcell].d_eflux[TOP_L];
    adjcell = IntVector(i,   j-1, k+1);     // BOT_FR
    sum_q_influx_EF += q_OAFE[TOP_BK][adjcell]   * d_OFE[adjcell].d_eflux[TOP_BK];
    adjcell = IntVector(i-1, j-1, k);       // BOT_L
    sum_q_influx_EF += q_OAFE[TOP_R][adjcell]    * d_OFE[adjcell].d_eflux[TOP_R];
    adjcell = IntVector(i,   j-1, k-1);     // BOT_BK
    sum_q_influx_EF += q_OAFE[TOP_FR][adjcell]   * d_OFE[adjcell].d_eflux[TOP_FR];
    adjcell = IntVector(i+1, j,   k-1);     // RIGHT_BK
    sum_q_influx_EF += q_OAFE[LEFT_FR][adjcell]  * d_OFE[adjcell].d_eflux[LEFT_FR];
    adjcell = IntVector(i+1, j,   k+1);     // RIGHT_FR
    sum_q_influx_EF += q_OAFE[LEFT_BK][adjcell]  * d_OFE[adjcell].d_eflux[LEFT_BK];
    adjcell = IntVector(i-1, j,   k-1);     // LEFT_BK
    sum_q_influx_EF += q_OAFE[RIGHT_FR][adjcell] * d_OFE[adjcell].d_eflux[RIGHT_FR];
    adjcell = IntVector(i-1, j,   k+1);     // LEFT_FR
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
    q_advected[c] = - sum_q_outflux - sum_q_outflux_EF - sum_q_outflux_CF
                    + sum_q_influx  + sum_q_influx_EF  + sum_q_influx_CF;                        
  }
}


//______________________________________________________________________
//
template <class T> void SecondOrderCEAdvector::qAverageFlux(const CCVariable<T>& q_CC,
                                              const Patch* patch,
						    CCVariable<T>& grad_lim,
                                              const CCVariable<T>& q_grad_x,
                                              const CCVariable<T>& q_grad_y,
                                              const CCVariable<T>& q_grad_z,
						    StaticArray<CCVariable<T> >& q_OAFS,
				                  StaticArray<CCVariable<T> >& q_OAFE,
				                  StaticArray<CCVariable<T> >& q_OAFC,						    
			                         DataWarehouse* new_dw)
  
{
#if 0
  //__________________________________
  // loop over each face of each patch. 
  // if the face DOESN'T have a neighbor
  // then set q_OAFS = q_CC
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){  
  
    if ( patch->getBCType(face) != Patch::Neighbor ){
    
      for(CellIterator iter = patch->getFaceCellIterator(face); 
                                                 !iter.done(); iter++) {
        IntVector c = *iter;  // hit only those cells along that face
        T Q_CC = q_CC[c];
        
        for (int face = TOP; face <= BACK; face ++){
          q_OAFS[face][c] = Q_CC;
        }
        for (int edge = TOP_R; edge <= LEFT_FR; edge++) {
          q_OAFE[edge][c] = Q_CC;
        }
        for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
          q_OAFC[corner][c] = Q_CC;
        }
        
      }
    }
  }
#endif  
/*`==========TESTING==========*/
// THERE GOT TO BE A BETTER WAY TO INITIALIZE.
  const IntVector gc(1,1,1);
  for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++) {
    IntVector c = *iter;  // hit only those cells along that face
    T Q_CC = q_CC[c];

    for (int face = TOP; face <= BACK; face ++){
      q_OAFS[face][c] = Q_CC;
    }
    for (int edge = TOP_R; edge <= LEFT_FR; edge++) {
      q_OAFE[edge][c] = Q_CC;
    }
    for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
      q_OAFC[corner][c] = Q_CC;
    }
  }  
/*===========TESTING==========`*/
  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
    
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    IntVector c = *iter;
    
    T q_grad_X = grad_lim[c] * q_grad_x[c];
    T q_grad_Y = grad_lim[c] * q_grad_y[c];
    T q_grad_Z = grad_lim[c] * q_grad_z[c];

    T Q_CC = q_CC[c];
    //__________________________________
    //  OUTAVERAGEFLUX: SLAB
    for (int face = TOP; face <= BACK; face ++){ 
      q_OAFS[face][c] = Q_CC + q_grad_X * r_out_x[c].d_fflux[face] + 
                               q_grad_Y * r_out_y[c].d_fflux[face] + 
				   q_grad_Z * r_out_z[c].d_fflux[face];
    }
  				  
    //__________________________________
    //  OUTAVERAGEFLUX: EDGE
    for (int edge = TOP_R; edge <= LEFT_FR; edge++ )   { 
      q_OAFE[edge][c] = Q_CC + q_grad_X * r_out_x_EF[c].d_eflux[edge] + 
                               q_grad_Y * r_out_y_EF[c].d_eflux[edge] + 
				   q_grad_Z * r_out_z_EF[c].d_eflux[edge];
    }
    
    //__________________________________
    //  OUTAVERAGEFLUX: CORNER
    for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  { 
      q_OAFC[corner][c] = Q_CC;
    }
  }
}

//______________________________________________________________________

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif
  
namespace Uintah {

  static MPI_Datatype makeMPI_fflux()
  {
    ASSERTEQ(sizeof(SecondOrderCEAdvector::fflux), sizeof(double)*6);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 6, 6, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(SecondOrderCEAdvector::fflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                              "SecondOrderCEAdvector::fflux", true, 
                              &makeMPI_fflux);
    }
    return td;
  }
  
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

void swapbytes( Uintah::SecondOrderCEAdvector::fflux& f) {
  double *p = f.d_fflux;
  SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
}

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
