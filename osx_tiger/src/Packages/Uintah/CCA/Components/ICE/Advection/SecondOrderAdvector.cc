#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderAdvector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
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
  new_dw->allocateTemporary(d_OFS,         patch, gac, 1);
  new_dw->allocateTemporary(r_out_x,       patch, gac, 1); 
  new_dw->allocateTemporary(r_out_y,       patch, gac, 1); 
  new_dw->allocateTemporary(r_out_z,       patch, gac, 1);
  new_dw->allocateTemporary(d_mass_massVertex, patch, gac, 1);
  new_dw->allocateTemporary(d_mass_slabs,      patch, gac, 1); 
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
    
    //__________________________________
    //   SLAB outfluxes
    double delX_Z = delX * delZ;
    double delX_Y = delX * delY;
    double delY_Z = delY * delZ;
    fflux& ofs = d_OFS[c];
    ofs.d_fflux[TOP]   = delY_top   * delX_Z;
    ofs.d_fflux[BOTTOM]= delY_bottom* delX_Z;
    ofs.d_fflux[RIGHT] = delX_right * delY_Z;
    ofs.d_fflux[LEFT]  = delX_left  * delY_Z;
    ofs.d_fflux[FRONT] = delZ_front * delX_Y;
    ofs.d_fflux[BACK]  = delZ_back  * delX_Y; 
    
    //__________________________________
    //  Bullet proofing
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
  // -find the cell, 
  // -set the outflux slab vol in all cells = 0.0,
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
//       M A S S   ( non-compatible)
void SecondOrderAdvector::advectMass( const CCVariable<double>& mass,
                                      CCVariable<double>& mass_advected,
                                      advectVarBasket* varBasket)
{
  const Patch* patch = varBasket->patch;
  DataWarehouse* new_dw = varBasket->new_dw;
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  CCVariable<vertex<double> > q_vertex;
  CCVariable<double> mass_grad_x, mass_grad_y, mass_grad_z;
  
  new_dw->allocateTemporary(mass_grad_x, patch,gac,1);
  new_dw->allocateTemporary(mass_grad_y, patch,gac,1);
  new_dw->allocateTemporary(mass_grad_z, patch,gac,1);
  new_dw->allocateTemporary(q_vertex,   patch,gac,1);
  bool compatible = false;
  
  // compute the limited gradients of mass 
  gradQ<double>(mass, patch, mass_grad_x, mass_grad_y, mass_grad_z);
  
  Q_vertex<double>(compatible, mass, q_vertex, patch,
                    mass_grad_x, mass_grad_y,  mass_grad_z);
  
  limitedGradient<double>(mass, patch, q_vertex, 
                          mass_grad_x, mass_grad_y, mass_grad_z);
                          
  // compute the value of q at the slab d_mass_slabs 
  // The other versions of advectQ need d_mass_slabs if compatible                                                
  qAverageFlux<double>( compatible, mass, d_notUsed_D, patch, d_mass_slabs, 
                        mass_grad_x, mass_grad_y, mass_grad_z);

  advectSlabs<double>(d_mass_slabs, patch,mass, mass_advected, 
                        d_notUsedX, d_notUsedY, d_notUsedZ, 
                        ignore_q_FC_calc_D());
                      
  // compute mass_CC/mass_vertex for each cell node                    
  mass_massVertex_ratio(mass, patch, mass_grad_x, mass_grad_y, mass_grad_z);
  
  q_FC_fluxes<double>(mass, d_mass_slabs,"mass", varBasket);
}


//__________________________________
//     D O U B L E
void SecondOrderAdvector::advectQ( const CCVariable<double>& A_CC,
                                   const CCVariable<double>& mass,
                                   CCVariable<double>& q_advected,
                                   advectVarBasket* varBasket)
{
  // pull variables out of the basket
  const Patch* patch = varBasket->patch;
  DataWarehouse* new_dw = varBasket->new_dw;
  bool useCompatibleFluxes = varBasket->useCompatibleFluxes;
  bool is_Q_massSpecific = varBasket ->is_Q_massSpecific;

  Ghost::GhostType  gac = Ghost::AroundCells;
  CCVariable<facedata<double> > q_OAFS;
  CCVariable<vertex<double> > q_vertex;
  CCVariable<double> q_grad_x, q_grad_y, q_grad_z, q_CC;
    
  new_dw->allocateTemporary(q_CC,     patch,gac,2);  
  new_dw->allocateTemporary(q_OAFS,   patch,gac,1);
  new_dw->allocateTemporary(q_grad_x, patch,gac,1);
  new_dw->allocateTemporary(q_grad_y, patch,gac,1);
  new_dw->allocateTemporary(q_grad_z, patch,gac,1);
  new_dw->allocateTemporary(q_vertex, patch,gac,2);
   
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
                            
  // compute the value of q at the slab q_OAFS
  qAverageFlux<double>( compatible, q_CC, mass, patch, q_OAFS, 
                        q_grad_x, q_grad_y, q_grad_z);
        
  advectSlabs<double>(q_OAFS,patch,q_CC, q_advected, 
                        d_notUsedX, d_notUsedY, d_notUsedZ, 
                        ignore_q_FC_calc_D());
                      
  q_FC_fluxes<double>(q_CC, q_OAFS, varBasket->desc, varBasket);  
}
//__________________________________
//  S P E C I A L I Z E D   D O U B L E 
//  needed by implicit solve  ( Non-compatible)
void SecondOrderAdvector::advectQ( const CCVariable<double>& q_CC,
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

  new_dw->allocateTemporary(q_OAFS,   patch,gac,1);
  new_dw->allocateTemporary(q_grad_x, patch,gac,1);
  new_dw->allocateTemporary(q_grad_y, patch,gac,1);
  new_dw->allocateTemporary(q_grad_z, patch,gac,1);
  new_dw->allocateTemporary(q_vertex, patch,gac,2);
  bool compatible = false;
  
  // compute the limited gradients    
  gradQ<double>(q_CC, patch, 
                q_grad_x, q_grad_y, q_grad_z); 
  
  Q_vertex<double>(compatible, q_CC, q_vertex, patch,
                    q_grad_x, q_grad_y,  q_grad_z); 
                   
  limitedGradient<double>(q_CC,patch,q_vertex,
                          q_grad_x,q_grad_y, q_grad_z);

  // compute the value of q_CC at the slab q_OAFS 
  qAverageFlux<double>( compatible, q_CC, d_notUsed_D, patch, q_OAFS, 
                        q_grad_x, q_grad_y, q_grad_z);
        
  advectSlabs<double>(q_OAFS,patch, q_CC, q_advected, 
                      q_XFC, q_YFC, q_ZFC, save_q_FC());
                      
  q_FC_PlusFaces( q_OAFS, q_CC, patch, q_XFC, q_YFC, q_ZFC);
}
//__________________________________
//     V E C T O R
void SecondOrderAdvector::advectQ(const CCVariable<Vector>& A_CC,
                                  const CCVariable<double>& mass,
                                  CCVariable<Vector>& q_advected,
                                  advectVarBasket* varBasket)
{

  // pull variables out of the basket
  const Patch* patch = varBasket->patch;
  DataWarehouse* new_dw = varBasket->new_dw;
  bool useCompatibleFluxes = varBasket->useCompatibleFluxes;
  bool is_Q_massSpecific = varBasket->is_Q_massSpecific;

  Ghost::GhostType  gac = Ghost::AroundCells;
  CCVariable<facedata<Vector> > q_OAFS;
  CCVariable<vertex<Vector> > q_vertex;
  CCVariable<Vector> q_grad_x, q_grad_y, q_grad_z, q_CC;
  
  new_dw->allocateTemporary(q_CC,     patch,gac,2);
  new_dw->allocateTemporary(q_OAFS,   patch,gac,1);
  new_dw->allocateTemporary(q_grad_x, patch,gac,1);
  new_dw->allocateTemporary(q_grad_y, patch,gac,1);
  new_dw->allocateTemporary(q_grad_z, patch,gac,1);
  new_dw->allocateTemporary(q_vertex, patch,gac,2);
  
  // convert from flux to primitive var. if using compatible fluxes
  bool compatible = 
   flux_to_primitive<Vector>(useCompatibleFluxes,is_Q_massSpecific,
                             patch, A_CC, mass, q_CC); 

  // compute the limited gradients of q_CC   
  gradQ<Vector>(q_CC, patch, q_grad_x, q_grad_y, q_grad_z); 
  
  Q_vertex<Vector>(compatible, q_CC, q_vertex, patch,
                    q_grad_x, q_grad_y,  q_grad_z); 
             
  limitedGradient<Vector>(q_CC,patch,q_vertex,
                          q_grad_x,q_grad_y,q_grad_z);

  // compute the value of q_CC at the slab q_OAFS 
  qAverageFlux<Vector>( compatible, q_CC, mass, patch, q_OAFS, 
                        q_grad_x, q_grad_y, q_grad_z);

  advectSlabs<Vector>(q_OAFS,patch, q_CC, q_advected, 
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignore_q_FC_calc_V());

  q_FC_fluxes<Vector>(q_CC, q_OAFS, varBasket->desc, varBasket);  
}

/*_____________________________________________________________________
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
                                
      faceVol[f]       = outfluxVol +  influxVol;
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
SecondOrderAdvector::qAverageFlux( const bool useCompatibleFluxes,
                                   const CCVariable<T>& q_CC,
                                   const CCVariable<double>& mass_CC,
                                   const Patch* patch,
                                   CCVariable<facedata<T> >& q_OAFS,
                                   const CCVariable<T>& grad_x,
                                   const CCVariable<T>& grad_y,
                                   const CCVariable<T>& grad_z)
  
{
  const Level* level=patch->getLevel();

  //__________________________________
  // on Boundary faces set q_OAFS
  // compatiblefluxes q_oafs = q_CC * mass_CC
  // non-compatible   q_oafs = q_CC
  vector<Patch::FaceType>::const_iterator itr; 
  
  for (itr  = patch->getBoundaryFaces()->begin(); 
       itr != patch->getBoundaryFaces()->end(); ++itr){
    Patch::FaceType face = *itr;
    
    for(CellIterator iter = patch->getFaceCellIterator(face); 
                                                    !iter.done(); iter++) {
      const IntVector& c = *iter;
      T Q_CC = q_CC[c];
      if (useCompatibleFluxes){         //  PULL THIS OUT AND MAKE SEPARATE LOOPS
        Q_CC *= mass_CC[c];
      }

      facedata<T>& oafs = q_OAFS[c];
      for (int face = TOP; face <= BACK; face ++){
        oafs.d_data[face] = Q_CC ;
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
        if (useCompatibleFluxes){
          Q_CC *= mass_CC[c];
        }
      
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
}
/*_____________________________________________________________________
 Function~ q_FC_operator
  Compute q at the face center of each cell
_____________________________________________________________________*/
template<class T>
void SecondOrderAdvector::q_FC_operator(CellIterator iter, 
                                           IntVector adj_offset,
                                           const int face,
                                           const CCVariable<facedata<double> >& q_OAFS,
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
                       
    double q_faceFlux = q_OAFS[L].d_data[IF_slab[face]] * influxVol 
                 - q_OAFS[R].d_data[face] * outfluxVol;
                      
    double faceVol = outfluxVol + influxVol;
    
    double q_tmp_FC = fabs(q_faceFlux)/(faceVol + 1.0e-100);
    
    // if q_tmp_FC = 0.0 then set it equal to q_CC[c]
    q_FC[R] = equalZero(q_faceFlux, q_CC[R], q_tmp_FC);
  }
}

/*_____________________________________________________________________
 Function~  q_FC_PlusFaces
 Compute q_FC values on the faces between the extra cells
 and the interior domain only on the x+, y+, z+ patch faces 
_____________________________________________________________________*/
void SecondOrderAdvector::q_FC_PlusFaces(
                                   const CCVariable<facedata<double> >& q_OAFS,
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
    q_FC_operator<SFCXVariable<double> >(Xiter, adj_offset[0],LEFT,  
                                        q_OAFS,q_CC,q_XFC);  
  }
  if (patchOnBoundary.y() == 1){
    q_FC_operator<SFCYVariable<double> >(Yiter, adj_offset[1],BOTTOM,
                                        q_OAFS,q_CC,q_YFC);
  }
  if (patchOnBoundary.z() == 1){  
    q_FC_operator<SFCZVariable<double> >(Ziter, adj_offset[2],BACK,  
                                        q_OAFS,q_CC,q_ZFC);   
  }
}

/*_____________________________________________________________________
 Function~ q_FC_flux_operator
  Compute q at the face center of each cell
_____________________________________________________________________*/
template<class T, class V>
void SecondOrderAdvector::q_FC_flux_operator(CellIterator iter, 
                                                 IntVector adj_offset,
                                                 const int face,
                                                 const CCVariable<facedata<V> >& q_OAFS,
                                             const CCVariable<V>& q_CC,
                                                 T& q_FC_flux)
{ 
  int out_indx = OF_slab[face];
  int in_indx  = IF_slab[face];
  
  for(;!iter.done(); iter++){
    IntVector c = *iter;      
    IntVector ac = c + adj_offset; 
     
     // face:           LEFT,   BOTTOM,   BACK  
     // IF_slab[face]:  RIGHT,  TOP,      FRONT
    double outfluxVol = d_OFS[c].d_fflux[out_indx];
    double influxVol  = d_OFS[ac].d_fflux[in_indx];
                       
    q_FC_flux[c] = q_OAFS[ac].d_data[in_indx] * influxVol 
                 - q_OAFS[c].d_data[out_indx] * outfluxVol;
  }
}
/*_____________________________________________________________________
 Function~  q_FC_fluxes
 Computes the sum(flux of q at the face center) over all subcycle timesteps
 on the fine level.  We only need to hit the cell that are on a coarse-fine 
 interface, ignoring the extraCells.
_____________________________________________________________________*/
template<class T>
void SecondOrderAdvector::q_FC_fluxes(const CCVariable<T>& q_CC,
                                      const CCVariable<facedata<T> >& q_OAFS,
                                      const string& desc,
                                      advectVarBasket* vb)
{
  if(vb->doAMR){
    // pull variables from the basket
    const int indx = vb->indx;
    const Patch* patch = vb->patch;
    DataWarehouse* new_dw = vb->new_dw;
    DataWarehouse* old_dw = vb->old_dw;
    const double AMR_subCycleProgressVar = vb->AMR_subCycleProgressVar;  

    // form the label names
    string x_name = desc + "_X_FC_flux";
    string y_name = desc + "_Y_FC_flux";
    string z_name = desc + "_Z_FC_flux";

    // get the varLabels
    VarLabel* xlabel = VarLabel::find(x_name);
    VarLabel* ylabel = VarLabel::find(y_name);
    VarLabel* zlabel = VarLabel::find(z_name);  
    if (xlabel == NULL || ylabel == NULL || zlabel == NULL){
      throw InternalError( "Advector: q_FC_fluxes: variable label not found: " 
                            + x_name + " or " + y_name + " or " + z_name);
    }
    Ghost::GhostType  gn  = Ghost::None;
    SFCXVariable<T> q_X_FC_flux;
    SFCYVariable<T> q_Y_FC_flux;
    SFCZVariable<T> q_Z_FC_flux;

    new_dw->allocateAndPut(q_X_FC_flux, xlabel,indx, patch);
    new_dw->allocateAndPut(q_Y_FC_flux, ylabel,indx, patch);
    new_dw->allocateAndPut(q_Z_FC_flux, zlabel,indx, patch); 

    if(AMR_subCycleProgressVar == 0){
      q_X_FC_flux.initialize(T(0.0));
      q_Y_FC_flux.initialize(T(0.0));
      q_Z_FC_flux.initialize(T(0.0));
    }else{
      constSFCXVariable<T> q_X_FC_flux_old;
      constSFCYVariable<T> q_Y_FC_flux_old;
      constSFCZVariable<T> q_Z_FC_flux_old;
      old_dw->get(q_X_FC_flux_old, xlabel, indx, patch, gn,0);
      old_dw->get(q_Y_FC_flux_old, ylabel, indx, patch, gn,0);
      old_dw->get(q_Z_FC_flux_old, zlabel, indx, patch, gn,0);
      q_X_FC_flux.copyData(q_X_FC_flux_old);
      q_Y_FC_flux.copyData(q_Y_FC_flux_old);
      q_Z_FC_flux.copyData(q_Z_FC_flux_old);
    }

    //__________________________________
    // Iterate over coarsefine interface faces
    vector<Patch::FaceType>::const_iterator iter;  
    for (iter  = patch->getCoarseFineInterfaceFaces()->begin(); 
         iter != patch->getCoarseFineInterfaceFaces()->end(); ++iter){
      Patch::FaceType patchFace = *iter;

      cout << "Patch " << patch->getID()<< " Level " << patch->getLevel()->getID()<<" patchFace " << patchFace;
      //__________________________________
      // 
      CellIterator iter=patch->getFaceCellIterator(patchFace, "alongInteriorFaceCells");
      IntVector adj_offset = patch->faceDirection(patchFace); // adj cell offset
      int cellFace = patchFaceToCellFace(patchFace);
  /*`==========TESTING==========*/
        IntVector begin = iter.begin();
        IntVector end = iter.end();
        IntVector half = (end - begin)/IntVector(2,2,2) + begin; 
  /*===========TESTING==========`*/

                            // X+ X-
      if(patchFace == Patch::xminus || patchFace == Patch::xplus){ 

        q_FC_flux_operator<SFCXVariable<T>, T>(iter, adj_offset,cellFace,
                                               q_OAFS, q_CC,q_X_FC_flux); 

        cout << half << " /t difference: q " << q_X_FC_flux[half] <<  endl;  
      }
                            // Y+ Y-
      if(patchFace == Patch::yminus || patchFace == Patch::yplus){

        q_FC_flux_operator<SFCYVariable<T>, T>(iter, adj_offset,cellFace,
                                               q_OAFS, q_CC,q_Y_FC_flux); 

        cout << half << " /t difference: q " << q_Y_FC_flux[half]  << endl;  
      }
                            // Z+ Z-
      if(patchFace == Patch::zminus || patchFace == Patch::zplus){

        q_FC_flux_operator<SFCZVariable<T>, T>(iter, adj_offset,cellFace,
                                               q_OAFS, q_CC,q_Z_FC_flux);

        cout << half << " /t difference: q " << q_Z_FC_flux[half] << endl;
      }
    }  // coarseFineInterface faces
  }   
}
