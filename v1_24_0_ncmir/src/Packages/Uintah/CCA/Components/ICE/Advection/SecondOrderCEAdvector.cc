#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderCEAdvector.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <iostream>

using namespace Uintah;
/* ---------------------------------------------------------------------
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
 ---------------------------------------------------------------------  */
SecondOrderCEAdvector::SecondOrderCEAdvector()
{
}

SecondOrderCEAdvector::SecondOrderCEAdvector(DataWarehouse* new_dw, 
                                             const Patch* patch)
{
  Ghost::GhostType  gac = Ghost::AroundCells;                           
  new_dw->allocateTemporary(d_OFS,      patch, gac,1);
  new_dw->allocateTemporary(d_OFE,      patch, gac,1);
  new_dw->allocateTemporary(d_OFC,      patch, gac,1);
  
  new_dw->allocateTemporary(r_out_x,    patch, gac,1);    
  new_dw->allocateTemporary(r_out_y,    patch, gac,1);    
  new_dw->allocateTemporary(r_out_z,    patch, gac,1);    
  
  new_dw->allocateTemporary(r_out_x_EF, patch, gac,1);    
  new_dw->allocateTemporary(r_out_y_EF, patch, gac,1);    
  new_dw->allocateTemporary(r_out_z_EF, patch, gac,1);
  
  new_dw->allocateTemporary(d_mass_massVertex, patch, gac, 1);
  new_dw->allocateTemporary(d_mass_slabs,      patch, gac, 1);
}


SecondOrderCEAdvector::~SecondOrderCEAdvector()
{
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
			      const int& indx,
                           const bool& bulletProof_test,
                           DataWarehouse* new_dw)
{
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom,delX_right, delX_left, delZ_front, delZ_back;
  double delX_tmp, delY_tmp,   delZ_tmp;
  double delX = dx.x(), delY = dx.y(), delZ = dx.z();
  double r_x, r_y, r_z, rx_R, rx_L, ry_TOP, ry_BOT, rz_FR, rz_BK;

  // Compute outfluxes 
  bool error = false;
  
  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getExtraCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
  
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    const IntVector& c = *iter;
    delY_top    = std::max(0.0, (vvel_FC[c+IntVector(0,1,0)] * delT));
    delY_bottom = std::max(0.0,-(vvel_FC[c                 ] * delT));
    delX_right  = std::max(0.0, (uvel_FC[c+IntVector(1,0,0)] * delT));
    delX_left   = std::max(0.0,-(uvel_FC[c                 ] * delT));
    delZ_front  = std::max(0.0, (wvel_FC[c+IntVector(0,0,1)] * delT));
    delZ_back   = std::max(0.0,-(wvel_FC[c                 ] * delT));
    
    delX_tmp    = dx.x() - delX_right - delX_left;
    delY_tmp    = dx.y() - delY_top   - delY_bottom;
    delZ_tmp    = dx.z() - delZ_front - delZ_back;
    
    //__________________________________
    //   SLAB outfluxes
    fflux& ofs = d_OFS[c];
    ofs.d_fflux[TOP]   = delY_top   * delX_tmp * delZ_tmp;
    ofs.d_fflux[BOTTOM]= delY_bottom* delX_tmp * delZ_tmp;
    ofs.d_fflux[RIGHT] = delX_right * delY_tmp * delZ_tmp;
    ofs.d_fflux[LEFT]  = delX_left  * delY_tmp * delZ_tmp;
    ofs.d_fflux[FRONT] = delZ_front * delX_tmp * delY_tmp;
    ofs.d_fflux[BACK]  = delZ_back  * delX_tmp * delY_tmp;

    // Edge outflux terms
    eflux& ofe = d_OFE[c];
    ofe.d_eflux[TOP_R]     = delY_top      * delX_right * delZ_tmp;
    ofe.d_eflux[TOP_FR]    = delY_top      * delX_tmp   * delZ_front;
    ofe.d_eflux[TOP_L]     = delY_top      * delX_left  * delZ_tmp;
    ofe.d_eflux[TOP_BK]    = delY_top      * delX_tmp   * delZ_back;
    
    ofe.d_eflux[BOT_R]     = delY_bottom   * delX_right * delZ_tmp;
    ofe.d_eflux[BOT_FR]    = delY_bottom   * delX_tmp   * delZ_front;
    ofe.d_eflux[BOT_L]     = delY_bottom   * delX_left  * delZ_tmp;
    ofe.d_eflux[BOT_BK]    = delY_bottom   * delX_tmp   * delZ_back;
    
    ofe.d_eflux[RIGHT_BK]  = delY_tmp      * delX_right * delZ_back;
    ofe.d_eflux[RIGHT_FR]  = delY_tmp      * delX_right * delZ_front;
    
    ofe.d_eflux[LEFT_BK]   = delY_tmp      * delX_left  * delZ_back;
    ofe.d_eflux[LEFT_FR]   = delY_tmp      * delX_left  * delZ_front;
    
    //__________________________________
    //   Corner outflux terms
    cflux& ofc = d_OFC[c];
    ofc.d_cflux[TOP_R_BK]  = delY_top      * delX_right * delZ_back;
    ofc.d_cflux[TOP_R_FR]  = delY_top      * delX_right * delZ_front;
    ofc.d_cflux[TOP_L_BK]  = delY_top      * delX_left  * delZ_back;
    ofc.d_cflux[TOP_L_FR]  = delY_top      * delX_left  * delZ_front;
    
    ofc.d_cflux[BOT_R_BK]  = delY_bottom   * delX_right * delZ_back;
    ofc.d_cflux[BOT_R_FR]  = delY_bottom   * delX_right * delZ_front;
    ofc.d_cflux[BOT_L_BK]  = delY_bottom   * delX_left  * delZ_back;
    ofc.d_cflux[BOT_L_FR]  = delY_bottom   * delX_left  * delZ_front;

    //__________________________________
    //  Bullet proofing
    double total_fluxout = 0.0;
    for(int face = TOP; face <= BACK; face++ )  {
      total_fluxout  += ofs.d_fflux[face];
    }
    for(int edge = TOP_R; edge <= LEFT_FR; edge++ )  {
      total_fluxout  += ofe.d_eflux[edge];
    }
    for(int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
      total_fluxout  += ofc.d_cflux[corner];
    }
    if(total_fluxout > vol){
      error = true;
    }    
    //__________________________________
    //   FOR EACH SLAB 
    r_x = delX_left/2.0   -  delX_right/2.0;
    r_y = delY_bottom/2.0 - delY_top/2.0;
    r_z = delZ_back/2.0   - delZ_front/2.0;
    
    fflux& rx = r_out_x[c];
    fflux& ry = r_out_y[c];
    fflux& rz = r_out_z[c];
    
    rx.d_fflux[RIGHT] = delX/2.0 - delX_right/2.0;
    ry.d_fflux[RIGHT] = r_y;
    rz.d_fflux[RIGHT] = r_z;

    rx.d_fflux[LEFT] = delX_left/2.0 - delX/2.0;
    ry.d_fflux[LEFT] = r_y;
    rz.d_fflux[LEFT] = r_z;

    rx.d_fflux[TOP] = r_x;
    ry.d_fflux[TOP] = delY/2.0 - delY_top/2.0;
    rz.d_fflux[TOP] = r_z;

    rx.d_fflux[BOTTOM] = r_x;
    ry.d_fflux[BOTTOM] = delY_bottom/2.0 - delY/2.0;
    rz.d_fflux[BOTTOM] = r_z;

    rx.d_fflux[FRONT] = r_x;
    ry.d_fflux[FRONT] = r_y;
    rz.d_fflux[FRONT] = delZ/2.0 - delZ_front/2.0;

    rx.d_fflux[BACK] = r_x;
    ry.d_fflux[BACK] = r_y;
    rz.d_fflux[BACK] = delZ_back/2.0 - delZ/2.0;

    //__________________________________
    //     FOR EACH EDGE 
    rx_R   = delX/2.0        - delX_right/2.0;
    rx_L   = delX_left/2.0   - delX/2.0;
    ry_TOP = delY/2.0        - delY_top/2.0;
    ry_BOT = delY_bottom/2.0 - delY/2.0;
    rz_FR  = delZ/2.0        - delZ_front/2.0;
    rz_BK  = delZ_back/2.0   - delZ/2.0;

    eflux& rx_EF = r_out_x_EF[c];
    eflux& ry_EF = r_out_y_EF[c];
    eflux& rz_EF = r_out_z_EF[c];
    
    rx_EF.d_eflux[TOP_R] = rx_R;
    ry_EF.d_eflux[TOP_R] = ry_TOP;  
    rz_EF.d_eflux[TOP_R] = r_z;

    rx_EF.d_eflux[TOP_FR] = r_x;
    ry_EF.d_eflux[TOP_FR] = ry_TOP;  
    rz_EF.d_eflux[TOP_FR] = rz_FR;

    rx_EF.d_eflux[TOP_L] = rx_L;
    ry_EF.d_eflux[TOP_L] = ry_TOP;  
    rz_EF.d_eflux[TOP_L] = r_z;

    rx_EF.d_eflux[TOP_BK] = r_x;
    ry_EF.d_eflux[TOP_BK] = ry_TOP;  
    rz_EF.d_eflux[TOP_BK] = rz_BK;

    rx_EF.d_eflux[BOT_R] = rx_R;
    ry_EF.d_eflux[BOT_R] = ry_BOT;  
    rz_EF.d_eflux[BOT_R] = r_z;

    rx_EF.d_eflux[BOT_FR] = r_x;
    ry_EF.d_eflux[BOT_FR] = ry_BOT;  
    rz_EF.d_eflux[BOT_FR] = rz_FR;

    rx_EF.d_eflux[BOT_L] = rx_L;
    ry_EF.d_eflux[BOT_L] = ry_BOT;  
    rz_EF.d_eflux[BOT_L] = r_z;

    rx_EF.d_eflux[BOT_BK] = r_x;
    ry_EF.d_eflux[BOT_BK] = ry_BOT;  
    rz_EF.d_eflux[BOT_BK] = rz_BK;

    rx_EF.d_eflux[RIGHT_BK] = rx_R;
    ry_EF.d_eflux[RIGHT_BK] = r_y;  
    rz_EF.d_eflux[RIGHT_BK] = rz_BK;

    rx_EF.d_eflux[RIGHT_FR] = rx_R;
    ry_EF.d_eflux[RIGHT_FR] = r_y;  
    rz_EF.d_eflux[RIGHT_FR] = rz_FR;

    rx_EF.d_eflux[LEFT_BK] = rx_L;
    ry_EF.d_eflux[LEFT_BK] = r_y;  
    rz_EF.d_eflux[LEFT_BK] = rz_BK;

    rx_EF.d_eflux[LEFT_FR] = rx_L;
    ry_EF.d_eflux[LEFT_FR] = r_y;  
    rz_EF.d_eflux[LEFT_FR] = rz_FR;
  } //cell iterator  
  //__________________________________
  // if total_fluxout > vol then 
  // -find the cell, 
  // -set the outflux vols in all cells = 0.0,
  // -request that the timestep be restarted.   
  if (error && bulletProof_test) {
    vector<IntVector> badCells;
    vector<double>  badOutflux;
    
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      const IntVector& c = *iter; 
      double total_fluxout = 0.0;
      for(int face = TOP; face <= BACK; face++ )  {
        total_fluxout  += d_OFS[c].d_fflux[face];
        d_OFS[c].d_fflux[face] = 0.0;
      }
      for(int edge = TOP_R; edge <= LEFT_FR; edge++ )  {
        total_fluxout  += d_OFE[c].d_eflux[edge];
        d_OFE[c].d_eflux[edge] = 0.0;
      }
      for(int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
        total_fluxout  += d_OFC[c].d_cflux[corner];
        d_OFC[c].d_cflux[corner] = 0.0;
      }
      // keep track of which cells are bad
      if (vol - total_fluxout < 0.0) {
        badCells.push_back(c);
        badOutflux.push_back(total_fluxout);
      }
    }  // cell iter
    warning_restartTimestep( badCells,badOutflux, vol, indx, patch, new_dw);
  }  // if total_fluxout > vol
  if (error && !bulletProof_test) {
    cout <<  " WARNING: ICE Advection operator "
         << " influx outflux volume error.  The bulletproofing that usually"
         << " catches this has been disabled "<< endl;
  }   
}

/* ---------------------------------------------------------------------
 Function~ advectQ
_____________________________________________________________________*/
void SecondOrderCEAdvector::advectMass( const CCVariable<double>& mass,
                                        const Patch* patch,
                                        CCVariable<double>& mass_advected,
                                        DataWarehouse* new_dw)
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  StaticArray<CCVariable<double> > q_OAFE(12), q_OAFC(8);
  CCVariable<vertex<double> > q_vertex;
  CCVariable<double> mass_grad_x, mass_grad_y, mass_grad_z;
  
  new_dw->allocateTemporary(mass_grad_x, patch,gac,1);
  new_dw->allocateTemporary(mass_grad_y, patch,gac,1);
  new_dw->allocateTemporary(mass_grad_z, patch,gac,1);
  new_dw->allocateTemporary(q_vertex,   patch,gac,1);
  for (int edge= TOP_R; edge <= LEFT_FR; edge++) {
    new_dw->allocateTemporary(q_OAFE[edge],    patch, gac,1);
  }
  for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++) {
    new_dw->allocateTemporary(q_OAFC[corner],  patch, gac,1);
  }
  
  bool compatible = false;
  
  // compute the limited gradients of mass 
  gradQ<double>(mass, patch, mass_grad_x, mass_grad_y, mass_grad_z);
  
  Q_vertex<double>(compatible, mass, q_vertex, patch,
	            mass_grad_x, mass_grad_y,  mass_grad_z);
  
  limitedGradient<double>(mass, patch, q_vertex, 
                          mass_grad_x, mass_grad_y, mass_grad_z);
                          
  // compute the value of q for each slab, corner, edge
  // The other versions of advectQ need d_mass_slabs if compatible                                                
  qAverageFlux<double>( compatible, mass, d_notUsed_D, patch, 
                        d_mass_slabs, q_OAFE, q_OAFC, 
                        mass_grad_x, mass_grad_y, mass_grad_z);

  advect<double>(d_mass_slabs, q_OAFE, q_OAFC, 
                 patch,mass, mass_advected, 
		   d_notUsedX, d_notUsedY, d_notUsedZ, 
		   ignoreFaceFluxesD());
                      
  // compute mass_CC/mass_vertex for each cell node                    
  mass_massVertex_ratio(mass, patch, mass_grad_x, mass_grad_y, mass_grad_z); 
}

//__________________________________
//     D O U B L E  
void SecondOrderCEAdvector::advectQ(const bool useCompatibleFluxes,
                                   const bool is_Q_massSpecific,
                                   const CCVariable<double>& A_CC,
                                   const CCVariable<double>& mass,
                                   const Patch* patch,
                                   CCVariable<double>& q_advected,
                                   DataWarehouse* new_dw)
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  CCVariable<facedata<double> > q_OAFS;
  CCVariable<vertex<double> > q_vertex;
  CCVariable<double> q_grad_x, q_grad_y, q_grad_z, q_CC;
  StaticArray<CCVariable<double> > q_OAFE(12), q_OAFC(8);
    
  new_dw->allocateTemporary(q_CC,     patch,gac,2);  
  new_dw->allocateTemporary(q_OAFS,   patch,gac,1);
  new_dw->allocateTemporary(q_grad_x, patch,gac,1);
  new_dw->allocateTemporary(q_grad_y, patch,gac,1);
  new_dw->allocateTemporary(q_grad_z, patch,gac,1);
  new_dw->allocateTemporary(q_vertex, patch,gac,2);
  
  for (int edge= TOP_R; edge <= LEFT_FR; edge++) {
    new_dw->allocateTemporary(q_OAFE[edge],    patch, gac,1);
  }
  for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++) {
    new_dw->allocateTemporary(q_OAFC[corner],  patch, gac,1);
  }
  
  // convert from flux to primitive var. if using compatible fluxes
  bool compatible = 
   flux_to_primitive<double>(useCompatibleFluxes,is_Q_massSpecific,
                             patch, A_CC, mass, q_CC); 
  
  // compute the limited gradients of q_CC
  gradQ<double>(q_CC, patch, q_grad_x, q_grad_y, q_grad_z); 
  
  Q_vertex<double>(compatible, q_CC, q_vertex, patch,
	            q_grad_x, q_grad_y,  q_grad_z);  
  
  limitedGradient<double>(q_CC, patch, q_vertex, 
                          q_grad_x, q_grad_y,  q_grad_z);


  // compute the value of q at slab, edge,corner
  qAverageFlux<double>( compatible, q_CC, mass, patch, 
                        q_OAFS, q_OAFE, q_OAFC, 
                        q_grad_x, q_grad_y, q_grad_z); 
            
  advect(q_OAFS, q_OAFE, q_OAFC, patch, q_CC,q_advected,
          d_notUsedX, d_notUsedY, d_notUsedZ, ignoreFaceFluxesD());

}
//__________________________________
//  S P E C I A L I Z E D   D O U B L E 
//  needed by implicit solve
void SecondOrderCEAdvector::advectQ(const CCVariable<double>& q_CC,
                                    const Patch* patch,
                                    CCVariable<double>& q_advected,
                                    SFCXVariable<double>& q_XFC,
                                    SFCYVariable<double>& q_YFC,
                                    SFCZVariable<double>& q_ZFC,
				        DataWarehouse* new_dw)
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  CCVariable<facedata<double> > q_OAFS;
  CCVariable<vertex<double> > q_vertex;
  CCVariable<double> q_grad_x, q_grad_y, q_grad_z;
  StaticArray<CCVariable<double> > q_OAFE(12), q_OAFC(8);
    
  new_dw->allocateTemporary(q_OAFS,   patch,gac,1);
  new_dw->allocateTemporary(q_grad_x, patch,gac,1);
  new_dw->allocateTemporary(q_grad_y, patch,gac,1);
  new_dw->allocateTemporary(q_grad_z, patch,gac,1);
  new_dw->allocateTemporary(q_vertex, patch,gac,2);
  
  for (int edge= TOP_R; edge <= LEFT_FR; edge++) {
    new_dw->allocateTemporary(q_OAFE[edge],    patch, gac,1);
  }
  for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++) {
    new_dw->allocateTemporary(q_OAFC[corner],  patch, gac,1);
  }
  bool compatible = false; 
  
  // compute the limited gradients of q_CC
  gradQ<double>(q_CC, patch, q_grad_x, q_grad_y, q_grad_z); 
  
  Q_vertex<double>(compatible, q_CC, q_vertex, patch,
	            q_grad_x, q_grad_y,  q_grad_z);  
  
  limitedGradient<double>(q_CC, patch, q_vertex, 
                          q_grad_x, q_grad_y,  q_grad_z);


  // compute the value of q at slab, edge,corner
  qAverageFlux<double>( compatible, q_CC, d_notUsed_D, patch, 
                        q_OAFS, q_OAFE, q_OAFC, 
                        q_grad_x, q_grad_y, q_grad_z);
                
  advect(q_OAFS, q_OAFE, q_OAFC, patch, q_CC, q_advected,
          q_XFC, q_YFC, q_ZFC, saveFaceFluxes());

}

//__________________________________
//     V E C T O R
void SecondOrderCEAdvector::advectQ(const bool useCompatibleFluxes,
                                    const bool is_Q_massSpecific,
                                    const CCVariable<Vector>& A_CC,
                                    const CCVariable<double>& mass,
                                    const Patch* patch,
                                    CCVariable<Vector>& q_advected,
                                    DataWarehouse* new_dw)
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  CCVariable<facedata<Vector> > q_OAFS;
  CCVariable<vertex<Vector> > q_vertex;
  CCVariable<Vector> q_grad_x, q_grad_y, q_grad_z, q_CC;
  StaticArray<CCVariable<Vector> > q_OAFE(12), q_OAFC(8);
    
  new_dw->allocateTemporary(q_CC,     patch,gac,2);  
  new_dw->allocateTemporary(q_OAFS,   patch,gac,1);
  new_dw->allocateTemporary(q_grad_x, patch,gac,1);
  new_dw->allocateTemporary(q_grad_y, patch,gac,1);
  new_dw->allocateTemporary(q_grad_z, patch,gac,1);
  new_dw->allocateTemporary(q_vertex, patch,gac,2);
  
  for (int edge= TOP_R; edge <= LEFT_FR; edge++) {
    new_dw->allocateTemporary(q_OAFE[edge],    patch, gac,1);
  }
  for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++) {
    new_dw->allocateTemporary(q_OAFC[corner],  patch, gac,1);
  }
  
  // convert from flux to primitive var. if using compatible fluxes
  bool compatible = 
   flux_to_primitive<Vector>(useCompatibleFluxes,is_Q_massSpecific,
                             patch, A_CC, mass, q_CC); 
  
  // compute the limited gradients of q_CC
  gradQ<Vector>(q_CC, patch, q_grad_x, q_grad_y, q_grad_z); 
  
  Q_vertex<Vector>(compatible, q_CC, q_vertex, patch,
	            q_grad_x, q_grad_y,  q_grad_z);  
  
  limitedGradient<Vector>(q_CC, patch, q_vertex, 
                          q_grad_x, q_grad_y,  q_grad_z);


  // compute the value of q at slab, edge,corner
  qAverageFlux<Vector>( compatible, q_CC, mass, patch, 
                        q_OAFS, q_OAFE, q_OAFC, 
                        q_grad_x, q_grad_y, q_grad_z);
         
  advect(q_OAFS, q_OAFE, q_OAFC, patch, q_CC, q_advected,
          d_notUsedX, d_notUsedY, d_notUsedZ, ignoreFaceFluxesV()); 
}

/* ---------------------------------------------------------------------
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template <class T, typename F> 
  void SecondOrderCEAdvector::advect(  CCVariable<facedata<T> >& q_OAFS,
                                       StaticArray<CCVariable<T> >& q_OAFE,
				           StaticArray<CCVariable<T> >& q_OAFC,
                                       const Patch* patch,
                                       const CCVariable<T>& q_CC,
                                       CCVariable<T>& q_advected,
                                       SFCXVariable<double>& q_XFC,
                                       SFCYVariable<double>& q_YFC,
                                       SFCZVariable<double>& q_ZFC,
                                       F save_q_FC)  // function is passed in
{
                                  //  W A R N I N G
  Vector dx = patch->dCell();    // assumes equal cell spacing             
  double invvol = 1.0/(dx.x() * dx.y() * dx.z());     
  double oneThird = 1.0/3.0;
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    const IntVector& c = *iter;
    T q_face_flux[6];
    double faceVol[6];
         
    for(int f = TOP; f <= BACK; f++ )  {    
      double slab_vol = 0.0;
      T q_slab_flux = T(0.0);
      //__________________________________
      //   S L A B S
      IntVector ac = c + S_ac[f];     // slab adjacent cell
      double outfluxVol = d_OFS[c ].d_fflux[OF_slab[f]];
      double influxVol  = d_OFS[ac].d_fflux[IF_slab[f]];

      q_slab_flux  =  q_OAFS[ac].d_data[IF_slab[f]] * influxVol
                    - q_OAFS[c].d_data[OF_slab[f]]  * outfluxVol;   
                               
      slab_vol     =  outfluxVol +  influxVol;

      //__________________________________
      //   E D G E S  
      T q_edge_flux = T(0.0);
      double edge_vol = 0.0;

      for(int e = 0; e < 4; e++ ) {
        int OF = OF_edge[f][e];    // cleans up the equations
        int IF = IF_edge[f][e];

        IntVector ac = c + E_ac[f][e]; // adjacent cell
        outfluxVol = 0.5 * d_OFE[c ].d_eflux[OF];
        influxVol  = 0.5 * d_OFE[ac].d_eflux[IF];

        q_edge_flux += -q_OAFE[OF][c]  * outfluxVol
                    +   q_OAFE[IF][ac] * influxVol;
        edge_vol    += outfluxVol + influxVol;
      }               

      //__________________________________
      //   C O R N E R S
      T q_corner_flux = T(0.0);
      double corner_vol = 0.0;

      for(int crner = 0; crner < 4; crner++ ) {
        int OF = OF_corner[f][crner];    // cleans up the equations
        int IF = IF_corner[f][crner];

        IntVector ac = c + C_ac[f][crner]; // adjacent cell
        outfluxVol = oneThird * d_OFC[c ].d_cflux[OF];
        influxVol  = oneThird * d_OFC[ac].d_cflux[IF];

        q_corner_flux += -q_OAFC[OF][c]  * outfluxVol 
                      +   q_OAFC[IF][ac] * influxVol; 
        corner_vol    += outfluxVol + influxVol;
      }  //  corner loop
      
      q_face_flux[f] = q_slab_flux + q_edge_flux + q_corner_flux;
      faceVol[f]     = slab_vol + edge_vol + corner_vol;
    }  // face loop 
       
    //__________________________________
    //  sum up all the contributions
    q_advected[c] = T(0.0);
    for(int f = TOP; f <= BACK; f++ )  {
      q_advected[c] += q_face_flux[f] * invvol;
    }

    //__________________________________
    //  inline function to compute q_FC 
    save_q_FC(c, q_XFC, q_YFC, q_ZFC, faceVol, q_face_flux, q_CC);                                         
  }  //cell iterator
}


//______________________________________________________________________
template <class T> 
void SecondOrderCEAdvector::qAverageFlux(const bool useCompatibleFluxes,
                                         const CCVariable<T>& q_CC,
                                         const CCVariable<double>& mass_CC,
                                         const Patch* patch,
					      CCVariable<facedata<T> >& q_OAFS,
				             StaticArray<CCVariable<T> >& q_OAFE,
				             StaticArray<CCVariable<T> >& q_OAFC,
                                         const CCVariable<T>& grad_x,
                                         const CCVariable<T>& grad_y,
                                         const CCVariable<T>& grad_z)
  
{
  //__________________________________
  // on Boundary faces set q_OAFS = q_CC
  vector<Patch::FaceType>::const_iterator itr;

  for (itr  = patch->getBoundaryFaces()->begin(); 
       itr != patch->getBoundaryFaces()->end(); ++itr){
    Patch::FaceType face = *itr;

    for(CellIterator iter = patch->getFaceCellIterator(face); 
        !iter.done(); iter++) {
      const IntVector& c = *iter;  // hit only those cells along that face
      T Q_CC = q_CC[c];
      if (useCompatibleFluxes){         //  PULL THIS OUT AND MAKE SEPARATE LOOPS
        Q_CC *= mass_CC[c];
      }
      facedata<T>& oafs = q_OAFS[c];
      for (int face = TOP; face <= BACK; face ++){
        oafs.d_data[face] = Q_CC;
      }
      for (int edge = TOP_R; edge <= LEFT_FR; edge++) {
        q_OAFE[edge][c] = Q_CC;
      }
      for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  {
        q_OAFC[corner][c] = Q_CC;
      }
    }
  }

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
  
  
  //__________________________________
  //   S L A B S 
  if (!useCompatibleFluxes) {  // non-compatible advection 
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      const IntVector& c = *iter;

      T Q_CC = q_CC[c];
      T q_grad_X = grad_x[c];
      T q_grad_Y = grad_y[c];
      T q_grad_Z = grad_z[c];
      facedata<T>& q_slab = q_OAFS[c];
      const fflux& rx = r_out_x[c];
      const fflux& ry = r_out_y[c];
      const fflux& rz = r_out_z[c];

      for (int face = TOP; face <= BACK; face ++){
        q_slab.d_data[face] = q_grad_X * rx.d_fflux[face] +         
                              q_grad_Y * ry.d_fflux[face] +               
                              q_grad_Z * rz.d_fflux[face] + Q_CC;   
      } 
    }
  }
  
  if (useCompatibleFluxes) {   // compatible fluxes  
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      const IntVector& c = *iter;

      T Q_CC = q_CC[c];
      T q_grad_X = grad_x[c];
      T q_grad_Y = grad_y[c];
      T q_grad_Z = grad_z[c];
      
      facedata<T>& q_slab = q_OAFS[c];
      const fflux& rx = r_out_x[c];
      const fflux& ry = r_out_y[c];
      const fflux& rz = r_out_z[c];
      const facedata<double> mass_slab = d_mass_slabs[c];
      const double mass = mass_CC[c];
      
      for (int face = TOP; face <= BACK; face ++){
        q_slab.d_data[face] = mass_slab.d_data[face] * Q_CC 
            + mass * ( q_grad_X * rx.d_fflux[face] +         
                       q_grad_Y * ry.d_fflux[face] +               
                       q_grad_Z * rz.d_fflux[face]);   
      } 
    }
  }
  
  
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    const IntVector& c = *iter;   				  
    //__________________________________
    //   E D G E S 
    T Q_CC = q_CC[c];
    T q_grad_X = grad_x[c];
    T q_grad_Y = grad_y[c];
    T q_grad_Z = grad_z[c];
    eflux& rx_EF = r_out_x_EF[c];
    eflux& ry_EF = r_out_y_EF[c];
    eflux& rz_EF = r_out_z_EF[c];
    for (int edge = TOP_R; edge <= LEFT_FR; edge++ )   { 
      q_OAFE[edge][c] = Q_CC + q_grad_X * rx_EF.d_eflux[edge] + 
                               q_grad_Y * ry_EF.d_eflux[edge] + 
				   q_grad_Z * rz_EF.d_eflux[edge];
    }
    
    //__________________________________
    //   C O R N E R S
    for (int corner = TOP_R_BK; corner <= BOT_L_FR; corner++ )  { 
      q_OAFC[corner][c] = Q_CC;
    }
  }
}
