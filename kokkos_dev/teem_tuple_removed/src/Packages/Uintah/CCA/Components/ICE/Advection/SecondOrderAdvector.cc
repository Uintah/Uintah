#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderAdvector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
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
}


SecondOrderAdvector::SecondOrderAdvector(DataWarehouse* new_dw, 
                                         const Patch*  patch) 
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(d_OFS,    patch, gac, 1);
  new_dw->allocateTemporary(r_out_x,  patch, gac, 1); 
  new_dw->allocateTemporary(r_out_y,  patch, gac, 1); 
  new_dw->allocateTemporary(r_out_z,  patch, gac, 1);      
}


SecondOrderAdvector::~SecondOrderAdvector()
{
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
                                          const bool& bulletProof_test,
                                          DataWarehouse* new_dw)

{

  Vector dx = patch->dCell();

  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom, delX_right, delX_left, delZ_front, delZ_back;
  double delX_tmp, delY_tmp, delZ_tmp;
  double delX = dx.x(), delY = dx.y(), delZ = dx.z();
  double r_x, r_y, r_z;

  // Compute outfluxes 

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getExtraCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
  
  bool error = false;
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    const IntVector& c = *iter;
 
    //!
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
    fflux& ofs = d_OFS[c];
    ofs.d_fflux[TOP]   = delY_top   * delX_Z_tmp;
    ofs.d_fflux[BOTTOM]= delY_bottom* delX_Z_tmp;
    ofs.d_fflux[RIGHT] = delX_right * delY_Z_tmp;
    ofs.d_fflux[LEFT]  = delX_left  * delY_Z_tmp;
    ofs.d_fflux[FRONT] = delZ_front * delX_Y_tmp;
    ofs.d_fflux[BACK]  = delZ_back  * delX_Y_tmp; 
    
    //__________________________________
    //  Bullet proofing
    //!
    double total_fluxout = 0.0;
    for(int face = TOP; face <= BACK; face++ )  {
      total_fluxout  += ofs.d_fflux[face];
    }
    if(total_fluxout > vol){
      error = true;
    }
        
    //__________________________________
    //  centroid
    r_x = delX_left/2.0   - delX_right/2.0;
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
  }//cell iterator
  
  //__________________________________
  // if total_fluxout > vol then 
  // find the cell and throw an exception.  
  if (error && bulletProof_test) {
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      const IntVector& c = *iter; 
      double total_fluxout = 0.0;
      for(int face = TOP; face <= BACK; face++ )  {
        total_fluxout  += d_OFS[c].d_fflux[face];
      }
      if (vol - total_fluxout < 0.0) {
        warning_restartTimestep( c, total_fluxout, vol, indx, new_dw);
        return;
        //throw OutFluxVolume(*iter,total_fluxout, vol, indx);
      }
    }  // cell iter
  }  // if total_fluxout > vol  
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

  CCVariable<facedata<double> > q_OAFS;
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(q_OAFS, patch,gac,1);
  
  qAverageFlux( q_CC, patch, q_OAFS);
        
  advectSlabs<double>(q_OAFS,patch,q_CC, q_advected, 
		      d_notUsedX, d_notUsedY, d_notUsedZ, 
		      ignoreFaceFluxesD());
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
  CCVariable<facedata<double> > q_OAFS;
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(q_OAFS, patch,gac,1);

  qAverageFlux(q_CC, patch, q_OAFS);
        
  advectSlabs<double>(q_OAFS,patch, q_CC, q_advected, 
                      q_XFC, q_YFC, q_ZFC, saveFaceFluxes());
}
//__________________________________
//     V E C T O R
void SecondOrderAdvector::advectQ(const CCVariable<Vector>& q_CC,
                                  const Patch* patch,
                                  CCVariable<Vector>& q_advected,
                                  DataWarehouse* new_dw)
{
  CCVariable<facedata<Vector> > q_OAFS;
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(q_OAFS, patch,gac,1);
  
  qAverageFlux(q_CC, patch, q_OAFS);

  advectSlabs<Vector>(q_OAFS,patch, q_CC, q_advected, 
                    d_notUsedX, d_notUsedY, d_notUsedZ, 
                    ignoreFaceFluxesV());
}

/* ---------------------------------------------------------------------
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template < class T, typename F>
void SecondOrderAdvector::advectSlabs( CCVariable<facedata<T> >& q_OAFS,
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

      T q_faceFlux_tmp = q_OAFS[ac].d_data[IF_slab[f]] * influxVol
	                 - q_OAFS[c].d_data[OF_slab[f]] * outfluxVol;
                                
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
//______________________________________________________________________
template <class T>
void
SecondOrderAdvector::qAverageFlux( const CCVariable<T>& q_CC,
                                   const Patch* patch,
                                   CCVariable<facedata<T> >& q_OAFS )
  
{
  const Level* level=patch->getLevel();
  Vector dx = patch->dCell();
  Vector inv_2del;
  inv_2del.x(1.0/(2.0 * dx.x()) );
  inv_2del.y(1.0/(2.0 * dx.y()) );
  inv_2del.z(1.0/(2.0 * dx.z()) );
  Vector dx_2 = dx/2.0;
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

      facedata<T>& oafs = q_OAFS[c];
      for (int face = TOP; face <= BACK; face ++){
        oafs.d_data[face] = Q_CC;
      }
    } 
  }
  //__________________________________
  // On Fine level patches set q_OAFS = q_CC in the 
  // extra cells of that patch
  // IS THIS THE RIGHT THING TO DO????
  if (level->getIndex() > 0 ) {
    for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
      
      for(CellIterator iter = patch->getFaceCellIterator(face); 
          !iter.done(); iter++) {
        const IntVector& c = *iter; 
        T Q_CC = q_CC[c];

        facedata<T>& oafs = q_OAFS[c];
        for (int face = TOP; face <= BACK; face ++){
          oafs.d_data[face] = Q_CC;
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
    const IntVector& c = *iter;

    T grad_x, grad_y, grad_z, gradlim;
    gradQ( q_CC, c, inv_2del, grad_x, grad_y, grad_z);
    gradientLimiter( q_CC, c, dx_2, gradlim, grad_x, grad_y, grad_z); 

    T q_grad_X = gradlim * grad_x;
    T q_grad_Y = gradlim * grad_y;
    T q_grad_Z = gradlim * grad_z;    
    
    T Q_CC = q_CC[c];
    //__________________________________
    //  OUTAVERAGEFLUX: SLAB
    //  with limiter.
    facedata<T>& oafs = q_OAFS[c];
    const fflux& rx = r_out_x[c];
    const fflux& ry = r_out_y[c];
    const fflux& rz = r_out_z[c];
    oafs.d_data[BACK] = q_grad_X * rx.d_fflux[BACK] + 
                        q_grad_Y * ry.d_fflux[BACK] +               
                        q_grad_Z * rz.d_fflux[BACK] + Q_CC;     
                                  
    oafs.d_data[FRONT] =q_grad_X * rx.d_fflux[FRONT] + 
                        q_grad_Y * ry.d_fflux[FRONT] + 
                        q_grad_Z * rz.d_fflux[FRONT] + Q_CC;

    oafs.d_data[BOTTOM] = q_grad_X * rx.d_fflux[BOTTOM] + 
                        q_grad_Y * ry.d_fflux[BOTTOM] + 
                        q_grad_Z * rz.d_fflux[BOTTOM] + Q_CC;
                                  
    oafs.d_data[TOP] =    q_grad_X * rx.d_fflux[TOP] + 
                        q_grad_Y * ry.d_fflux[TOP] + 
                        q_grad_Z * rz.d_fflux[TOP] +  Q_CC;
                                  
    oafs.d_data[LEFT] =   q_grad_X * rx.d_fflux[LEFT] + 
                        q_grad_Y * ry.d_fflux[LEFT] + 
                        q_grad_Z * rz.d_fflux[LEFT] + Q_CC;

    oafs.d_data[RIGHT] =  q_grad_X * rx.d_fflux[RIGHT] + 
                        q_grad_Y * ry.d_fflux[RIGHT] + 
                        q_grad_Z * rz.d_fflux[RIGHT] + Q_CC;
  }
}
