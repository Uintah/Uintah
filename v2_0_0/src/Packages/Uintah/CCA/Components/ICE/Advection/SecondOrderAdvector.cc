#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderAdvector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Exceptions/OutFluxVolume.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <iostream>

using namespace Uintah;
using namespace std;

SecondOrderAdvector::SecondOrderAdvector()
{
  OFS_CCLabel = 0;
}


SecondOrderAdvector::SecondOrderAdvector(DataWarehouse* new_dw, 
                                         const Patch*  patch) 
{
  OFS_CCLabel = VarLabel::create("OFS_CC",
                                 CCVariable<fflux>::getTypeDescription());
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(d_OFS,    patch, gac, 1);
  new_dw->allocateTemporary(r_out_x,  patch, gac, 1); 
  new_dw->allocateTemporary(r_out_y,  patch, gac, 1); 
  new_dw->allocateTemporary(r_out_z,  patch, gac, 1);      
}


SecondOrderAdvector::~SecondOrderAdvector()
{
  VarLabel::destroy(OFS_CCLabel);
}
SecondOrderAdvector* SecondOrderAdvector::clone(DataWarehouse* new_dw,
                                                const Patch* patch)
{
  return scinew SecondOrderAdvector(new_dw,patch);
}

/* ---------------------------------------------------------------------
 Function~  influxOutfluxVolume--
 Purpose~   calculate the individual outfluxes for each cell.
            This includes the only the slabs.
            
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

void
SecondOrderAdvector::inFluxOutFluxVolume( const SFCXVariable<double>& uvel_FC,
                                          const SFCYVariable<double>& vvel_FC,
                                          const SFCZVariable<double>& wvel_FC,
                                          const double& delT, 
                                          const Patch* patch,
                                          const int& indx,
                                          const bool& bulletProof_test)

{
  Vector dx = patch->dCell();

  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom, delX_right, delX_left, delZ_front, delZ_back;
  double delX_tmp, delY_tmp, delZ_tmp;
  double delX = dx.x(), delY = dx.y(), delZ = dx.z();
  double r_x, r_y, r_z;

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
        
    //__________________________________
    //  centroid
    r_x = delX_left/2.0   - delX_right/2.0;
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
  }//cell iterator

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
void SecondOrderAdvector::allocateAndCompute_Q_ave(
                              const CCVariable<T>& q_CC,
                              const Patch* patch,
                              DataWarehouse* new_dw,
                              StaticArray<CCVariable<T> >& q_OAFS )
{

  CCVariable<T> grad_lim, q_grad_x,q_grad_y,q_grad_z;
  Ghost::GhostType  gac = Ghost::AroundCells;
  
  for (int face = TOP; face <= BACK; face++) {
    new_dw->allocateTemporary(q_OAFS[face], patch,gac,1);
  }
  
  new_dw->allocateTemporary(grad_lim, patch, gac, 1);
  new_dw->allocateTemporary(q_grad_x, patch, gac, 1);  
  new_dw->allocateTemporary(q_grad_y, patch, gac, 1);  
  new_dw->allocateTemporary(q_grad_z, patch, gac, 1); 
  
  //__________________________________
  gradQ(q_CC, patch, q_grad_x, q_grad_y, q_grad_z);
    
  gradientLimiter(q_CC, patch, grad_lim, q_grad_x, q_grad_y, q_grad_z, new_dw);
                  
  qAverageFlux(   q_CC, patch, grad_lim, q_grad_x, q_grad_y, q_grad_z, 
                  q_OAFS);
}
/* ---------------------------------------------------------------------
 Function~ advectQ
_____________________________________________________________________*/
//     D O U B L E
void SecondOrderAdvector::advectQ( const CCVariable<double>& q_CC,
                                   const Patch* patch,
                                   CCVariable<double>& q_advected,
                                   DataWarehouse* new_dw)
{

  StaticArray<CCVariable<double> > q_OAFS(6);
  allocateAndCompute_Q_ave<double>(q_CC, patch, new_dw, q_OAFS);
        
  advectSlabs<double>(q_OAFS,patch,q_CC, q_advected, 
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignoreFaceFluxesD);
}
//__________________________________
//  S P E C I A L I Z E D   D O U B L E 
//  needed by implicit solve
void SecondOrderAdvector::advectQ( const CCVariable<double>& q_CC,
                                   const Patch* patch,
                                   CCVariable<double>& q_advected,
                                   SFCXVariable<double>& q_XFC,
                                   SFCYVariable<double>& q_YFC,
                                   SFCZVariable<double>& q_ZFC,
				       DataWarehouse* new_dw)
{
  StaticArray<CCVariable<double> > q_OAFS(6);
  allocateAndCompute_Q_ave<double>(q_CC, patch, new_dw, q_OAFS);
        
  advectSlabs<double>(q_OAFS,patch, q_CC, q_advected, 
                      q_XFC, q_YFC, q_ZFC, saveFaceFluxes);
}
//__________________________________
//     V E C T O R
void SecondOrderAdvector::advectQ(const CCVariable<Vector>& q_CC,
                                  const Patch* patch,
                                  CCVariable<Vector>& q_advected,
                                  DataWarehouse* new_dw)
{
  StaticArray<CCVariable<Vector> > q_OAFS(6);
  allocateAndCompute_Q_ave<Vector>(q_CC, patch, new_dw, q_OAFS);

  advectSlabs<Vector>(q_OAFS,patch, q_CC, q_advected, 
                    d_notUsedX, d_notUsedY, d_notUsedZ, 
                    ignoreFaceFluxesV);
}

/* ---------------------------------------------------------------------
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template < class T, typename F>
void SecondOrderAdvector::advectSlabs( StaticArray<CCVariable<T> >& q_OAFS,
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

      q_face_flux[f]  =  q_OAFS[IF_slab[f]][ac] * influxVol
                       - q_OAFS[OF_slab[f]][c]  * outfluxVol;
                                
      faceVol[f]      =  outfluxVol +  influxVol;
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

//______________________________________________________________________
//
template <class T>
void
SecondOrderAdvector::qAverageFlux( const CCVariable<T>& q_CC,
                                   const Patch* patch,
                                   CCVariable<T>& grad_lim,
                                   const CCVariable<T>& q_grad_x,
                                   const CCVariable<T>& q_grad_y,
                                   const CCVariable<T>& q_grad_z,
                                   StaticArray<CCVariable<T> >& q_OAFS )
  
{
  //__________________________________
  // loop over each face of each patch 
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
        
      }   
    }
  }
   
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
    //  with limiter.    
    q_OAFS[BACK][c] =   q_grad_X * r_out_x[c].d_fflux[BACK] + 
                        q_grad_Y * r_out_y[c].d_fflux[BACK] +               
                        q_grad_Z * r_out_z[c].d_fflux[BACK] + Q_CC;     
                                  
    q_OAFS[FRONT][c] =  q_grad_X * r_out_x[c].d_fflux[FRONT] + 
                        q_grad_Y * r_out_y[c].d_fflux[FRONT] + 
                        q_grad_Z * r_out_z[c].d_fflux[FRONT] + Q_CC;

    q_OAFS[BOTTOM][c] = q_grad_X * r_out_x[c].d_fflux[BOTTOM] + 
                        q_grad_Y * r_out_y[c].d_fflux[BOTTOM] + 
                        q_grad_Z * r_out_z[c].d_fflux[BOTTOM] + Q_CC;
                                  
    q_OAFS[TOP][c] =    q_grad_X * r_out_x[c].d_fflux[TOP] + 
                        q_grad_Y * r_out_y[c].d_fflux[TOP] + 
                        q_grad_Z * r_out_z[c].d_fflux[TOP] +  Q_CC;
                                  
    q_OAFS[LEFT][c] =   q_grad_X * r_out_x[c].d_fflux[LEFT] + 
                        q_grad_Y * r_out_y[c].d_fflux[LEFT] + 
                        q_grad_Z * r_out_z[c].d_fflux[LEFT] + Q_CC;

    q_OAFS[RIGHT][c] =  q_grad_X * r_out_x[c].d_fflux[RIGHT] + 
                        q_grad_Y * r_out_y[c].d_fflux[RIGHT] + 
                        q_grad_Z * r_out_z[c].d_fflux[RIGHT] + Q_CC;
  }
}
