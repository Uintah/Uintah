#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderAdvector.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/IntVector.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <iostream>

using namespace Uintah;
using std::cerr;
using std::endl;

/* ---------------------------------------------------------------------
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
 ---------------------------------------------------------------------  */
FirstOrderAdvector::FirstOrderAdvector() 
{
}


FirstOrderAdvector::FirstOrderAdvector(DataWarehouse* new_dw, 
                                   const Patch* patch)
{
  new_dw->allocateTemporary(d_OFS,     patch, Ghost::AroundCells,1);
}


FirstOrderAdvector::~FirstOrderAdvector()
{
}

FirstOrderAdvector* FirstOrderAdvector::clone(DataWarehouse* new_dw,
                                         const Patch* patch)
{
  return scinew FirstOrderAdvector(new_dw,patch);
}

/* ---------------------------------------------------------------------
 Function~  influxOutfluxVolume--
 Purpose~   calculate the individual outfluxes for each cell.
            This includes the slabs
 Steps for each cell:  
 1) calculate the volume for each outflux
 2) test to see if the total outflux > cell volume

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
                        const int& indx,
                        const bool& bulletProof_test,
                        DataWarehouse* new_dw)

{
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom,delX_right, delX_left, delZ_front, delZ_back;
  double delX_tmp, delY_tmp, delZ_tmp;
  double delX = dx.x(), delY = dx.y(), delZ = dx.z();

  // Compute outfluxes 
  const IntVector gc(1,1,1);
  bool error = false;
  for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
    const IntVector& c = *iter; 
    delY_top    = std::max(0.0, (vvel_FC[c+IntVector(0,1,0)] * delT));
    delY_bottom = std::max(0.0,-(vvel_FC[c+IntVector(0,0,0)] * delT));
    delX_right  = std::max(0.0, (uvel_FC[c+IntVector(1,0,0)] * delT));
    delX_left   = std::max(0.0,-(uvel_FC[c+IntVector(0,0,0)] * delT));
    delZ_front  = std::max(0.0, (wvel_FC[c+IntVector(0,0,1)] * delT));
    delZ_back   = std::max(0.0,-(wvel_FC[c+IntVector(0,0,0)] * delT));
    
    delX_tmp    = delX - delX_right - delX_left;
    delY_tmp    = delY - delY_top   - delY_bottom;
    delZ_tmp    = delZ - delZ_front - delZ_back;     

    //__________________________________
    //   SLAB outfluxes
    double delX_Z_tmp = delX_tmp * delZ_tmp;
    double delX_Y_tmp = delX_tmp * delY_tmp;
    double delY_Z_tmp = delY_tmp * delZ_tmp;
    fflux& ofs = d_OFS[c];
    ofs.d_fflux[TOP]   = delY_top   * delX_Z_tmp;
    ofs.d_fflux[BOTTOM]= delY_bottom* delX_Z_tmp;
    ofs.d_fflux[RIGHT] = delX_right * delY_Z_tmp;
    ofs.d_fflux[LEFT]  = delX_left  * delY_Z_tmp;
    ofs.d_fflux[FRONT] = delZ_front * delX_Y_tmp;
    ofs.d_fflux[BACK]  = delZ_back  * delX_Y_tmp; 

    //__________________________________
    //  Bullet proofing
    double total_fluxout = 0.0;
    for(int face = TOP; face <= BACK; face++ )  {
      total_fluxout  += ofs.d_fflux[face];
    }
    if(total_fluxout > vol){
      error = true;
    }
  }  //cell iterator
  
  //__________________________________
  // if total_fluxout > vol then 
  // -find the cell, 
  // -set the outflux slab vol in all cells = 0.0,
  // -request that the timestep be restarted.
  if (error && bulletProof_test) {
    vector<IntVector> badCells;
    vector<double>  badOutflux;
    
    for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
      IntVector c = *iter; 
      double total_fluxout = 0.0;
      for(int face = TOP; face <= BACK; face++ )  {
        total_fluxout  += d_OFS[c].d_fflux[face];
        d_OFS[c].d_fflux[face] = 0.0;
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

/*_____________________________________________________________________
 Function~ advectQ
_____________________________________________________________________*/
//     M A S S
void FirstOrderAdvector::advectMass(const CCVariable<double>& q_CC,
                                    const Patch* patch,
                                    CCVariable<double>& q_advected,
			               DataWarehouse* /*new_dw*/)
{
        
  advectSlabs<double>(q_CC,patch,q_advected, 
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignoreFaceFluxesD());
}

//__________________________________
//     D O U B L E
void FirstOrderAdvector::advectQ(const bool /*useCompatibleFluxes*/,
                                 const bool /*is_Q_massSpecific*/,
                                 const CCVariable<double>& q_CC,
                                 const CCVariable<double>& /*mass*/,
                                 const Patch* patch,
                                 CCVariable<double>& q_advected,
                                 DataWarehouse* /*new_dw*/)
{
        
  advectSlabs<double>(q_CC,patch,q_advected, 
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignoreFaceFluxesD());
}

//__________________________________
//  S P E C I A L I Z E D   D O U B L E 
//  needed by implicit ICE
void FirstOrderAdvector::advectQ(const CCVariable<double>& q_CC,
                                 const Patch* patch,
                                 CCVariable<double>& q_advected,
                                 SFCXVariable<double>& q_XFC,
                                 SFCYVariable<double>& q_YFC,
                                 SFCZVariable<double>& q_ZFC,
				     DataWarehouse* /*new_dw*/)
{
  advectSlabs<double>(q_CC,patch,q_advected,  
                      q_XFC, q_YFC, q_ZFC, saveFaceFluxes());
		      
  compute_q_FC_PlusFaces( q_CC, patch, q_XFC, q_YFC, q_ZFC); 
}
//__________________________________
//     V E C T O R
void FirstOrderAdvector::advectQ(const bool /*useCompatibleFluxes*/,
                                 const bool /*is_Q_massSpecific*/,
                                 const CCVariable<Vector>& q_CC,
                                 const CCVariable<double>& /*mass*/,
                                 const Patch* patch,
                                 CCVariable<Vector>& q_advected,
                                 DataWarehouse* /*new_dw*/)
{
  advectSlabs<Vector>(q_CC,patch,q_advected, 
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignoreFaceFluxesV());
		      
} 

/*_____________________________________________________________________
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template <class T, typename F> 
  void FirstOrderAdvector::advectSlabs(const CCVariable<T>& q_CC,
                                       const Patch* patch,                   
                                       CCVariable<T>& q_advected,
                                       SFCXVariable<double>& q_XFC,
                                       SFCYVariable<double>& q_YFC,
                                       SFCZVariable<double>& q_ZFC,
                                       F save_q_FC) // function is passed in
{                                
                                  //  W A R N I N G
  Vector dx = patch->dCell();    // assumes equal cell spacing             
  double invvol = 1.0/(dx.x() * dx.y() * dx.z());                     

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    const IntVector& c = *iter;  
    
    T q_face_flux[6];
    double faceVol[6];
    
    T sum_q_face_flux(0.0);   
    for(int f = TOP; f <= BACK; f++ )  {    
      //__________________________________
      //   S L A B S
      IntVector ac = c + S_ac[f];     // slab adjacent cell
      double outfluxVol = d_OFS[c ].d_fflux[OF_slab[f]];
      double influxVol  = d_OFS[ac].d_fflux[IF_slab[f]];

      T q_faceFlux_tmp  =   q_CC[ac] * influxVol - q_CC[c] * outfluxVol;
        
      faceVol[f]       =  outfluxVol +  influxVol;
      q_face_flux[f]   = q_faceFlux_tmp; 
      sum_q_face_flux += q_faceFlux_tmp;
    }  
    q_advected[c] = sum_q_face_flux*invvol;
    
    //__________________________________
    //  inline function to compute q_FC
    save_q_FC(c, q_XFC, q_YFC, q_ZFC, faceVol, q_face_flux, q_CC); 
  }
}
/*_____________________________________________________________________
 Function~ compute_q_FC
 This takes care of the q_FC values  on the x+, y+, z+ patch faces
_____________________________________________________________________*/
template<class T>
void FirstOrderAdvector::compute_q_FC(CellIterator iter, 
                		      IntVector adj_offset,
                		      const int face,
                		      const CCVariable<double>& q_CC,
                		      T& q_FC)
{
  for(;!iter.done(); iter++){
    IntVector R = *iter;      
    IntVector L = R + adj_offset; 
     
     // face:           LEFT,   BOTTOM,   BACK  
     // IF_slab[face]:  RIGHT,  TOP,      FRONT
    double outfluxVol = d_OFS[R].d_fflux[face];
    double influxVol  = d_OFS[L].d_fflux[IF_slab[face]];
    
    double q_faceFlux = q_CC[L] * influxVol - q_CC[R] * outfluxVol;
    double faceVol    = outfluxVol + influxVol;
    
    double tmp_FC     = fabs(q_faceFlux)/(faceVol + 1.0e-100);

    // if q_FC = 0.0 then set it equal to q_CC[c]
    q_FC[R] = equalZero(q_faceFlux, q_CC[R], tmp_FC);
  }
}

/*_____________________________________________________________________
 Function~  compute_q_FC_PlusFaces
 Compute q_FC values on the faces between the extra cells
 and the interior domain only on the x+, y+, z+ patch faces 
_____________________________________________________________________*/
void FirstOrderAdvector::compute_q_FC_PlusFaces(
      	      	      	      	       const CCVariable<double>& q_CC,
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
    compute_q_FC<SFCXVariable<double> >(Xiter, adj_offset[0], LEFT,  
                                        q_CC,q_XFC);
  } 
  if (patchOnBoundary.y() == 1 ){
    compute_q_FC<SFCYVariable<double> >(Yiter, adj_offset[1], BOTTOM,
                                        q_CC,q_YFC); 
  }
  if (patchOnBoundary.z() == 1 ){  
    compute_q_FC<SFCZVariable<double> >(Ziter, adj_offset[2], BACK,  
                                        q_CC,q_ZFC);  
  }
}

