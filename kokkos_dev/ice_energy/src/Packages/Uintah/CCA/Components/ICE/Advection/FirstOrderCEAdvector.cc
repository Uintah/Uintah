#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderCEAdvector.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
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
FirstOrderCEAdvector::FirstOrderCEAdvector()
{
}

FirstOrderCEAdvector::FirstOrderCEAdvector(DataWarehouse* new_dw, 
                                          const Patch* patch)
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(d_OFS,  patch, gac,1);
  new_dw->allocateTemporary(d_OFE,  patch, gac,1);
  new_dw->allocateTemporary(d_OFC,  patch, gac,1);
}


FirstOrderCEAdvector::~FirstOrderCEAdvector()
{
}

FirstOrderCEAdvector* FirstOrderCEAdvector::clone(DataWarehouse* new_dw,
                                   const Patch* patch)
{
  return scinew FirstOrderCEAdvector(new_dw,patch);
}

/* ---------------------------------------------------------------------
 Function~  influxOutfluxVolume--
 Purpose~   calculate the individual outfluxes for each cell.
            This includes the slabs, edges and corners
 Steps for each cell:  
 1) calculate the volume for each outflux
 2) test to see if the total outflux > cell volume

Implementation notes:
The outflux of volume is calculated in each cell in the computational domain
+ one layer of extra cells  surrounding the domain.The face-centered velocity c
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
                        const bool& bulletProof_test,
                        DataWarehouse* new_dw)

{
  if(patch->getLevel()->getID() > 0){
    cout << " WARNING: FirstOrderCE doesn't work with multiple levels" << endl;
    cout << " Todd:  you need to set boundary conditions on the transverse vel_FC" << endl;
    throw InternalError(" ERROR", __FILE__, __LINE__);
  }
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom,delX_right, delX_left, delZ_front, delZ_back;
  double delX_tmp, delY_tmp,   delZ_tmp;

  // Compute outfluxes 
  const IntVector gc(1,1,1);
  bool error = false;
  
  for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
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

    // Edge flux terms
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
    //   Corner flux terms
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
  }  // cell iterator
  //__________________________________
  // if total_fluxout > vol then 
  // -find the cell, 
  // -set the outflux vols in all cells = 0.0,
  // -request that the timestep be restarted.  
  if (error && bulletProof_test) {
    vector<IntVector> badCells;
    vector<double>  badOutflux;
    
    for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
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
    string mesg = " WARNING: ICE Advection operator "
         " influx outflux volume error.  The bulletproofing that usually"
        " catches this has been disabled ";
    static SCIRun::ProgressiveWarning warn(mesg,10); 
    warn.invoke();
  }
}

/* ---------------------------------------------------------------------
 Function~ advectQ
_____________________________________________________________________*/
//     M A S S
void FirstOrderCEAdvector::advectMass(const CCVariable<double>& q_CC,
                                      CCVariable<double>& q_advected,
                                      advectVarBasket* varBasket)
{
        
  advectCE<double>(q_CC,varBasket->patch,q_advected, 
                   d_notUsedX, d_notUsedY, d_notUsedZ, 
                   ignore_q_FC_calc_D());
                   
  q_FC_fluxes<double>(q_CC, "mass", varBasket);  
}

//__________________________________
//     D O U B L E
void FirstOrderCEAdvector::advectQ(const CCVariable<double>& q_CC,
                                   const CCVariable<double>& /*mass*/,
                                   CCVariable<double>& q_advected,
                                   advectVarBasket* varBasket)
{
        
  advectCE<double>(q_CC,varBasket->patch,q_advected, 
                   d_notUsedX, d_notUsedY, d_notUsedZ, 
                   ignore_q_FC_calc_D());

  q_FC_fluxes<double>(q_CC, varBasket->desc, varBasket);  
}

//__________________________________
//  S P E C I A L I Z E D   D O U B L E 
void FirstOrderCEAdvector::advectQ(const CCVariable<double>& q_CC,
                                   const Patch* patch,
                                   CCVariable<double>& q_advected,
                                   SFCXVariable<double>& q_XFC,
                                   SFCYVariable<double>& q_YFC,
                                   SFCZVariable<double>& q_ZFC,
                                   DataWarehouse* /*new_dw*/)
{
  advectCE<double>(q_CC,patch,q_advected,  
                   q_XFC, q_YFC, q_ZFC, save_q_FC());
  
  q_FC_PlusFaces( q_CC, patch, q_XFC, q_YFC, q_ZFC);
}
//__________________________________
//     V E C T O R
void FirstOrderCEAdvector::advectQ(const CCVariable<Vector>& q_CC,
                                   const CCVariable<double>& /*mass*/,
                                   CCVariable<Vector>& q_advected,
                                   advectVarBasket* varBasket)
{
  advectCE<Vector>(q_CC,varBasket->patch,q_advected, 
                   d_notUsedX, d_notUsedY, d_notUsedZ, 
                   ignore_q_FC_calc_V());
                   
  q_FC_fluxes<Vector>(q_CC, varBasket->desc, varBasket);  
} 


/* ---------------------------------------------------------------------
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template <class T, typename F> 
  void FirstOrderCEAdvector::advectCE(const CCVariable<T>& q_CC,
                                  const Patch* patch,                   
                                  CCVariable<T>& q_advected,
                                  SFCXVariable<double>& q_XFC,
                                  SFCYVariable<double>& q_YFC,
                                  SFCZVariable<double>& q_ZFC,
                                  F save_q_FC)  // function is passed in
{
                                
  Vector dx = patch->dCell();           
  double invvol = 1.0/(dx.x() * dx.y() * dx.z());
  double oneThird = 1.0/3.0;
  
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    const IntVector& c = *iter;
    //__________________________________
    //   all faces
    T Q_CC = q_CC[c];
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

      q_slab_flux  = - Q_CC  * outfluxVol + q_CC[ac] * influxVol;             
      slab_vol     =  outfluxVol +  influxVol;                 

      //__________________________________
      //   E D G E S  
      T q_edge_flux = T(0.0);
      double edge_vol = 0.0;

      for(int e = 0; e < 4; e++ ) {
        int OF = OF_edge[f][e];        // cleans up the equations
        int IF = IF_edge[f][e];

        IntVector ac = c + E_ac[f][e]; // adjcent cell
        outfluxVol = 0.5 * d_OFE[c ].d_eflux[OF];
        influxVol  = 0.5 * d_OFE[ac].d_eflux[IF];

        q_edge_flux += -Q_CC    * outfluxVol
                    +  q_CC[ac] * influxVol;
        edge_vol    += outfluxVol + influxVol;
      }                

      //__________________________________
      //   C O R N E R S
      T q_corner_flux = T(0.0);
      double corner_vol = 0.0;

      for(int crner = 0; crner < 4; crner++ ) {
        int OF = OF_corner[f][crner];      // cleans up the equations
        int IF = IF_corner[f][crner];

        IntVector ac = c + C_ac[f][crner]; // adjacent cell
        outfluxVol = oneThird * d_OFC[c ].d_cflux[OF];
        influxVol  = oneThird * d_OFC[ac].d_cflux[IF];

        q_corner_flux += -Q_CC * outfluxVol 
                      +  q_CC[ac] * influxVol; 
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
  }
}

/*_____________________________________________________________________
 Function~ q_FC_operator
 This takes care of the q_FC values  on the x+, y+, z+ patch faces
_____________________________________________________________________*/
template<class T>
void FirstOrderCEAdvector::q_FC_operator(CellIterator iter, 
                                         IntVector adj_offset,
                                         const int face,
                                         const CCVariable<double>& q_CC,
                                         T& q_FC)
{
  double oneThird = 1.0/3.0;

  for(;!iter.done(); iter++){
    IntVector R = *iter;      
    IntVector L = R + adj_offset; 
    //__________________________________
    //   S L A B S
    // face:           LEFT,   BOTTOM,   BACK  
    // IF_slab[face]:  RIGHT,  TOP,      FRONT
    double outfluxVol = d_OFS[R].d_fflux[face];
    double influxVol  = d_OFS[L].d_fflux[IF_slab[face]];

    double q_slab_flux = q_CC[L] * influxVol - q_CC[R] * outfluxVol;             
    double slab_vol    =  outfluxVol +  influxVol;                 

    //__________________________________
    //   E D G E S  
    double q_edge_flux = 0.0;
    double edge_vol = 0.0;

    for(int e = 0; e < 4; e++ ) {
      int OF = OF_edge[face][e];        // cleans up the equations
      int IF = IF_edge[face][e];

      IntVector L = R + E_ac[face][e]; // adjcent cell
      outfluxVol = 0.5 * d_OFE[R].d_eflux[OF];
      influxVol  = 0.5 * d_OFE[L].d_eflux[IF];

      q_edge_flux += -q_CC[R] * outfluxVol
                  +  q_CC[L]  * influxVol;
      edge_vol    += outfluxVol + influxVol;
    }                

    //__________________________________
    //   C O R N E R S
    double q_corner_flux = 0.0;
    double corner_vol = 0.0;

    for(int crner = 0; crner < 4; crner++ ) {
      int OF = OF_corner[face][crner];      // cleans up the equations
      int IF = IF_corner[face][crner];

      IntVector L = R + C_ac[face][crner]; // adjacent cell
      outfluxVol = oneThird * d_OFC[R].d_cflux[OF];
      influxVol  = oneThird * d_OFC[L].d_cflux[IF];

      q_corner_flux += -q_CC[R] * outfluxVol 
                    +  q_CC[L]  * influxVol; 
      corner_vol    += outfluxVol + influxVol;
    }  //  corner loop

    double q_faceFlux = q_slab_flux + q_edge_flux + q_corner_flux;
    double faceVol    = slab_vol + edge_vol + corner_vol;
    
    double q_tmp_FC = fabs(q_faceFlux)/(faceVol + 1.0e-100);

      // if q_FC = 0.0 then set it equal to q_CC[c]
    q_FC[R] = equalZero(q_faceFlux, q_CC[R], q_tmp_FC);
  }
}

/*_____________________________________________________________________
 Function~  q_FC_PlusFaces
 Compute q_FC values on the faces between the extra cells
 and the interior domain only on the x+, y+, z+ patch faces 
_____________________________________________________________________*/
void FirstOrderCEAdvector::q_FC_PlusFaces(const CCVariable<double>& q_CC,
                                          const Patch* patch,
                                          SFCXVariable<double>& q_XFC,
                                          SFCYVariable<double>& q_YFC,
                                          SFCZVariable<double>& q_ZFC)
{                                                  
  vector<IntVector> adj_offset(3);
  adj_offset[0] = IntVector(-1, 0, 0);    // X faces
  adj_offset[1] = IntVector(0, -1, 0);    // Y faces
  adj_offset[2] = IntVector(0,  0, -1);   // Z faces

  CellIterator Xiter=patch->getFaceCellIterator(Patch::xplus,"minusEdgeCells");
  CellIterator Yiter=patch->getFaceCellIterator(Patch::yplus,"minusEdgeCells");
  CellIterator Ziter=patch->getFaceCellIterator(Patch::zplus,"minusEdgeCells");
  
  IntVector patchOnBoundary = patch->neighborsHigh();
  // only work on patches that are at the edge of the computational domain
  
  if (patchOnBoundary.x() == 1 ){
    q_FC_operator<SFCXVariable<double> >(Xiter, adj_offset[0], LEFT,  
                                         q_CC,q_XFC);
  } 
  if (patchOnBoundary.y() == 1 ){
    q_FC_operator<SFCYVariable<double> >(Yiter, adj_offset[1], BOTTOM,
                                         q_CC,q_YFC); 
  }
  if (patchOnBoundary.z() == 1 ){  
    q_FC_operator<SFCZVariable<double> >(Ziter, adj_offset[2], BACK,  
                                         q_CC,q_ZFC);  
  }
}

/*_____________________________________________________________________
 Function~ q_FC_flux operator
 computes the flux of q at the cell face
_____________________________________________________________________*/
template<class T, class V>
void FirstOrderCEAdvector::q_FC_flux_operator(CellIterator, 
                                              IntVector,
                                              const int,
                                              const CCVariable<V>&,
                                              T&)
{
  // implement when we have FO and SO working
}
/*_____________________________________________________________________
 Function~  q_FC_fluxes
_____________________________________________________________________*/
template<class T>
void FirstOrderCEAdvector::q_FC_fluxes( const CCVariable<T>&,
                                        const string& ,
                                        advectVarBasket* vb)
{
  if(vb->doAMR){
    // implement when we have FO and SO working
  }  // doAMR   
}
