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
                           const bool& bulletProof_test)
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
    for(int face = TOP; face <= BACK; face++ )  {
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
  if (fabs(error_test - num_cells) > 1.0e-2 && bulletProof_test) {
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
 Function~ allocateAndCompute_Q_ave
 Purpose   allocates temporary space and computes q for each slab,
           edge and corner.  
_____________________________________________________________________*/
template <class T>
void SecondOrderCEAdvector::allocateAndCompute_Q_ave(
                              const CCVariable<T>& q_CC,
                              const Patch* patch,
                              DataWarehouse* new_dw,
                              StaticArray<CCVariable<T> >& q_OAFS,
                              StaticArray<CCVariable<T> >& q_OAFE,
                              StaticArray<CCVariable<T> >& q_OAFC )
{

  CCVariable<T> grad_lim, q_grad_x,q_grad_y,q_grad_z;
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
  
  new_dw->allocateTemporary(q_grad_x,   patch, gac, 1); 
  new_dw->allocateTemporary(q_grad_y,   patch, gac, 1); 
  new_dw->allocateTemporary(q_grad_z,   patch, gac, 1);

  //__________________________________
  gradQ(q_CC, patch, q_grad_x, q_grad_y, q_grad_z);
    
  gradientLimiter(q_CC, patch, grad_lim, q_grad_x, q_grad_y, q_grad_z,new_dw);
  qAverageFlux(q_CC, patch, grad_lim,  q_grad_x, q_grad_y, q_grad_z,
               q_OAFS,  q_OAFE, q_OAFC);
}

/* ---------------------------------------------------------------------
 Function~ advectQ
_____________________________________________________________________*/
//     D O U B L E  
void SecondOrderCEAdvector::advectQ(const CCVariable<double>& q_CC,
                             const Patch* patch,
                             CCVariable<double>& q_advected,
			        DataWarehouse* new_dw)
{
 
  StaticArray<CCVariable<double> > q_OAFS(6), q_OAFE(12), q_OAFC(8);
  
  allocateAndCompute_Q_ave<double>(q_CC, patch, new_dw, q_OAFS, q_OAFE, q_OAFC);
              
  advect(q_OAFS, q_OAFE, q_OAFC, patch, q_CC,q_advected,
          d_notUsedX, d_notUsedY, d_notUsedZ, ignoreFaceFluxesD);

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
  StaticArray<CCVariable<double> > q_OAFS(6), q_OAFE(12), q_OAFC(8);
  
  allocateAndCompute_Q_ave<double>(q_CC, patch, new_dw, q_OAFS, q_OAFE, q_OAFC);

              
  advect(q_OAFS, q_OAFE, q_OAFC, patch, q_CC, q_advected,
          q_XFC, q_YFC, q_ZFC, saveFaceFluxes);

}

//__________________________________
//     V E C T O R
void SecondOrderCEAdvector::advectQ(const CCVariable<Vector>& q_CC,
                             const Patch* patch,
                             CCVariable<Vector>& q_advected,
			        DataWarehouse* new_dw)
{
  StaticArray<CCVariable<Vector> > q_OAFS(6), q_OAFE(12), q_OAFC(8);
  
  allocateAndCompute_Q_ave<Vector>(q_CC, patch, new_dw, q_OAFS, q_OAFE, q_OAFC);
         
  advect(q_OAFS, q_OAFE, q_OAFC, patch, q_CC, q_advected,
          d_notUsedX, d_notUsedY, d_notUsedZ, ignoreFaceFluxesV); 
}

/* ---------------------------------------------------------------------
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template <class T, typename F> 
  void SecondOrderCEAdvector::advect(  StaticArray<CCVariable<T> >& q_OAFS,
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
  
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    IntVector c = *iter;
    double oneThird = 1.0/3.0;
     
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

      q_slab_flux  =  q_OAFS[IF_slab[f]][ac] * influxVol
                    - q_OAFS[OF_slab[f]][c]  * outfluxVol;   
                               
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
//
template <class T> void SecondOrderCEAdvector::qAverageFlux(const CCVariable<T>& q_CC,
                                              const Patch* patch,
						    CCVariable<T>& grad_lim,
                                              const CCVariable<T>& q_grad_x,
                                              const CCVariable<T>& q_grad_y,
                                              const CCVariable<T>& q_grad_z,
						    StaticArray<CCVariable<T> >& q_OAFS,
				                  StaticArray<CCVariable<T> >& q_OAFE,
				                  StaticArray<CCVariable<T> >& q_OAFC)
  
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
