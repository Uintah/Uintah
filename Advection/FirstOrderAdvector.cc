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
                        const bool& bulletProof_test)

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
    IntVector c = *iter;
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
    d_OFS[c].d_fflux[TOP]   = delY_top   * delX_Z_tmp;
    d_OFS[c].d_fflux[BOTTOM]= delY_bottom* delX_Z_tmp;
    d_OFS[c].d_fflux[RIGHT] = delX_right * delY_Z_tmp;
    d_OFS[c].d_fflux[LEFT]  = delX_left  * delY_Z_tmp;
    d_OFS[c].d_fflux[FRONT] = delZ_front * delX_Y_tmp;
    d_OFS[c].d_fflux[BACK]  = delZ_back  * delX_Y_tmp; 

    //__________________________________
    //  Bullet proofing
    double total_fluxout = 0.0;
    for(int face = TOP; face <= BACK; face++ )  {
      total_fluxout  += d_OFS[c].d_fflux[face];
    }
    
    num_cells++;
    error_test +=(vol - total_fluxout)/fabs(vol- total_fluxout);
  }  //cell iterator
  
  //__________________________________
  // if total_fluxout > vol then 
  // find the cell and throw an exception.
  if(fabs(error_test - num_cells) > 1.0e-2){
    cout << " outfluxVol > vol " << endl;
  }
  if (fabs(error_test - num_cells) > 1.0e-2 && bulletProof_test) {
    for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
      IntVector c = *iter; 
      double total_fluxout = 0.0;
      for(int face = TOP; face <= BACK; face++ )  {
        total_fluxout  += d_OFS[c].d_fflux[face];
      }
      if (vol - total_fluxout < 0.0) {
        throw OutFluxVolume(*iter,total_fluxout, vol, indx);
      }
    }  // cell iter
  }  // if total_fluxout > vol
}

/* ---------------------------------------------------------------------
 Function~ advectQ
_____________________________________________________________________*/
//     D O U B L E
void FirstOrderAdvector::advectQ(const CCVariable<double>& q_CC,
                             const Patch* patch,
                             CCVariable<double>& q_advected,
				 DataWarehouse* /*new_dw*/)
{
        
  advectSlabs<double>(q_CC,patch,q_advected, 
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignoreFaceFluxesD);
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
                      q_XFC, q_YFC, q_ZFC, saveFaceFluxes);
}
//__________________________________
//     V E C T O R
void FirstOrderAdvector::advectQ(const CCVariable<Vector>& q_CC,
                             const Patch* patch,
                             CCVariable<Vector>& q_advected,
				 DataWarehouse* /*new_dw*/)
{
  advectSlabs<Vector>(q_CC,patch,q_advected, 
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignoreFaceFluxesV);
} 

/* ---------------------------------------------------------------------
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
    IntVector c = *iter; 
    //__________________________________
    //   all faces
    T q_face_flux[6];
    double faceVol[6];
         
    for(int f = TOP; f <= BACK; f++ )  {    
      //__________________________________
      //   S L A B S
      IntVector ac = c + S_ac[f];     // slab adjacent cell
      double outfluxVol = d_OFS[c ].d_fflux[OF_slab[f]];
      double influxVol  = d_OFS[ac].d_fflux[IF_slab[f]];

      q_face_flux[f] =   q_CC[ac] * influxVol - q_CC[c] * outfluxVol;
      faceVol[f]     =   outfluxVol +  influxVol;
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

