#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderAdvector.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <iostream>

//#define SPEW
#undef SPEW
#define is_rightFace_variable(face,var) ( ((face == "xminus" || face == "xplus") && var == "scalar-f") ?1:0  )

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
                                       const Patch* patch,
                                       const bool isNewGrid)
{
  new_dw->allocateTemporary(d_OFS,patch, Ghost::AroundCells,1);
  
  // Initialize temporary variables when the grid changes
  if(isNewGrid){   
    double EVILNUM = -9.99666999e30;
    CellIterator iter = patch->getCellIterator();
    CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      const IntVector& c = *iter;
      for(int face = TOP; face <= BACK; face++ )  {
        d_OFS[c].d_fflux[face]= EVILNUM;
      }
    }
  }
}


FirstOrderAdvector::~FirstOrderAdvector()
{
}

FirstOrderAdvector* FirstOrderAdvector::clone(DataWarehouse* new_dw,
                                              const Patch* patch,
                                              const bool isNewGrid)
{
  return scinew FirstOrderAdvector(new_dw,patch,isNewGrid);
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
#if 0
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom,delX_right, delX_left, delZ_front, delZ_back;
  double delX = dx.x(), delY = dx.y(), delZ = dx.z();

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
  }  //cell iterator
  
  //__________________________________
  // if total_fluxout > vol then 
  // -find the cell, 
  // -set the outflux slab vol in all cells = 0.0,
  // -request that the timestep be restarted.
  // -ignore if a timestep restart has already been requested
  bool tsr = new_dw->timestepRestarted();
  
  if (error && bulletProof_test && !tsr) {
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
    ostringstream mesg;
    mesg << " WARNING: ICE Advection operator Influx/Outflux volume error:"
         << " Patch " << patch->getID()
         << ", Level " << patch->getLevel()->getIndex();
    static SCIRun::ProgressiveWarning warn(mesg.str(),10); 
    warn.invoke();
  }
#endif
}

/*_____________________________________________________________________
 Function~ advectQ
_____________________________________________________________________*/
//     M A S S
void FirstOrderAdvector::advectMass(const CCVariable<double>& q_CC,
                                    CCVariable<double>& q_advected,
                                    constSFCXVariable<double>& uvel_FC,
                                    constSFCYVariable<double>& vvel_FC,
                                    constSFCZVariable<double>& wvel_FC,
                                    advectVarBasket* varBasket)
{
        
  advectSlabs<double>(q_CC ,q_advected, varBasket,
                      uvel_FC, vvel_FC, wvel_FC,
                      d_notUsedX, d_notUsedY, d_notUsedZ,
                      ignore_q_FC_D());                
}

//__________________________________
//     D O U B L E 
// (int_eng, sp_vol * mass, transported Variables)
void FirstOrderAdvector::advectQ(const CCVariable<double>& q_CC,
                                 const CCVariable<double>& /*mass*/,
                                 CCVariable<double>& q_advected,
                                 constSFCXVariable<double>& uvel_FC,
                                 constSFCYVariable<double>& vvel_FC,
                                 constSFCZVariable<double>& wvel_FC,
                                 advectVarBasket* varBasket)
{                                 
  advectSlabs<double>(q_CC, q_advected, varBasket,
                      uvel_FC, vvel_FC, wvel_FC,
                      d_notUsedX, d_notUsedY, d_notUsedZ,
                      ignore_q_FC_D());
}

//__________________________________
//  S P E C I A L I Z E D   D O U B L E 
//  needed by implicit ICE  q_CC = volFrac
void FirstOrderAdvector::advectQ(const CCVariable<double>& q_CC,
                                 CCVariable<double>& q_advected,
                                 advectVarBasket* varBasket,
                                 constSFCXVariable<double>& uvel_FC,
                                 constSFCYVariable<double>& vvel_FC,
                                 constSFCZVariable<double>& wvel_FC,
                                 SFCXVariable<double>& q_XFC,
                                 SFCYVariable<double>& q_YFC,
                                 SFCZVariable<double>& q_ZFC)
{
  advectSlabs<double>(q_CC,q_advected, varBasket,
                      uvel_FC, vvel_FC, wvel_FC,
                      q_XFC, q_YFC, q_ZFC, 
                      save_q_FC());
}
//__________________________________
//     V E C T O R  (momentum)
void FirstOrderAdvector::advectQ(const CCVariable<Vector>& q_CC,
                                 const CCVariable<double>& /*mass*/,
                                 CCVariable<Vector>& q_advected,
                                 constSFCXVariable<double>& uvel_FC,
                                 constSFCYVariable<double>& vvel_FC,
                                 constSFCZVariable<double>& wvel_FC,
                                 advectVarBasket* varBasket)
{
  advectSlabs<Vector>(q_CC, q_advected, varBasket, 
                      uvel_FC, vvel_FC, wvel_FC,
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignore_q_FC_V());
}

/*_____________________________________________________________________
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template <class T, typename F> 
  void FirstOrderAdvector::advectSlabs(const CCVariable<T>& q_CC,             
                                       CCVariable<T>& q_advected,
                                       advectVarBasket* vb,
                                       constSFCXVariable<double>& uvel_FC,
                                       constSFCYVariable<double>& vvel_FC,
                                       constSFCZVariable<double>& wvel_FC,
                                       SFCXVariable<double>& q_XFC,
                                       SFCYVariable<double>& q_YFC,
                                       SFCZVariable<double>& q_ZFC,
                                       F save_q_FC) // function is passed in
{
  const Patch* patch = vb->patch;
  Vector cell_dx = patch->dCell();
  
  T zero(0.0);
  q_advected.initialize(zero);
  const double delT = vb->delT;

  double dx = cell_dx[0];
  double dy = cell_dx[1];
  double dz = cell_dx[2];

  // To hit all x-,y-,z- faces increase the
  // cell iterator by one cell in the x+,y+,z+ directions
  CellIterator itr = patch->getCellIterator();
  IntVector l = itr.begin();
  IntVector h = itr.end()  + IntVector(1,1,1);
  CellIterator iter(l,h);
  
  for(;!iter.done(); iter++){
      
    IntVector c = *iter;
    
    IntVector donor;
    IntVector receive;
    int i = c.x();
    int j = c.y(); 
    int k = c.z();
    
    //__________________________________
    // X- face
    double dx_slab = uvel_FC[c] * delT;
    double fluxVol = dx_slab * dy * dz;     // volume fluxed through the face
   
    if(uvel_FC[c] < 0){
      donor   = IntVector(i-1, j, k);    // donor cell
      receive = c;
    }else{
      donor   = c;
      receive = IntVector(i-1, j, k);    // receiving cell
    }

    // compute the flux of q and update the receiver/donor cells
    T q_faceFlux         = q_CC[donor] * fluxVol;
    q_advected[donor]   -= q_faceFlux;
    q_advected[receive] += q_faceFlux;
      
    //__________________________________
    // Y- face
    double dy_slab = vvel_FC[c] * delT;
    fluxVol = dx * dy_slab * dz;       
   
    if(vvel_FC[c] < 0){
      donor   = IntVector(i, j-1, k);  
      receive = c;
    }else{
      donor   = c;
      receive = IntVector(i, j-1, k);  
    }

    q_faceFlux           = q_CC[donor] * fluxVol;
    q_advected[donor]   -= q_faceFlux;
    q_advected[receive] += q_faceFlux;
    
    //__________________________________
    // Z- face
    double dz_slab = wvel_FC[c] * delT;
    fluxVol = dx * dy * dz_slab;   
   
    if(wvel_FC[c] < 0){
      donor   = IntVector(i,j,k-1);
      receive = c;
    }else{
      donor   = c;
      receive = IntVector(i,j,k-1);
    }

    q_faceFlux           = q_CC[donor] * fluxVol;
    q_advected[donor]   -= q_faceFlux;
    q_advected[receive] += q_faceFlux;      
       
    //__________________________________
    //  inline function to save crossing the cell face
    //save_q_FC(q_FC[c], q_CC[up]);
  } 
}
/*`==========TESTING==========*/
#if 0 
/*_____________________________________________________________________
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template <class T, typename F> 
  void FirstOrderAdvector::advectSlabs(const CCVariable<T>& q_CC,             
                                       CCVariable<T>& q_advected,
                                       advectVarBasket* vb,
                                       constSFCXVariable<double>& uvel_FC,
                                       constSFCYVariable<double>& vvel_FC,
                                       constSFCZVariable<double>& wvel_FC,
                                       SFCXVariable<double>& q_XFC,
                                       SFCYVariable<double>& q_YFC,
                                       SFCZVariable<double>& q_ZFC,
                                       F save_q_FC) // function is passed in
{
  const Patch* patch = vb->patch;
  Vector cell_dx = patch->dCell();
  
  T zero(0.0);
  q_advected.initialize(zero);
  const double delT = vb->delT;
  
  //__________________________________
  // precompute  dy * dz 
  // pdir        dy_dz
  // -----------------
  //  x          dy * dz
  //  y          dx * dz
  //  z          dx * dy
  Vector dy_dz;
  dy_dz[0]= cell_dx[1] * cell_dx[2];
  dy_dz[1]= cell_dx[0] * cell_dx[2];
  dy_dz[2]= cell_dx[0] * cell_dx[1];

  // To hit all x-,y-,z- faces increase the
  // cell iterator by one cell in the x+,y+,z+ directions
  CellIterator itr = patch->getCellIterator();
  IntVector l = itr.begin();
  IntVector h = itr.end()  + IntVector(1,1,1);
  CellIterator iter(l,h);
  
  for(;!iter.done(); iter++){
      
    IntVector c = *iter;
    Vector velFC(uvel_FC[c],vvel_FC[c], wvel_FC[c]);
    
    // loop over the x-, y-, z- cell faces
    for (int p_dir = 0; p_dir<3; p_dir ++){
      double dx      = velFC[p_dir] * delT;   // vel_FC on the bottom, left, back cell face
      double fluxVol = dx * dy_dz[p_dir];     // volume fluxed through the face
      
      // Find the donor/receiving cell
      IntVector donor   = c;
      IntVector receive = c;
      if(velFC[p_dir] < 0){
        donor[p_dir]   = c[p_dir] - 1;  // donor cell
      }else{
        receive[p_dir] = c[p_dir] - 1;    // receiving cell
      }

      // compute the flux of q and update the receiver/donor cells
      T q_faceFlux         = q_CC[donor] * fluxVol;
      q_advected[donor]   -= q_faceFlux;
      q_advected[receive] += q_faceFlux;
    }
    //__________________________________
    //  inline function to save crossing the cell face
    //save_q_FC(q_FC[c], q_CC[up]);
  } 
}


/*_____________________________________________________________________
 Function~ advect_operator
________________________________________________*/
template<class T, class V, typename F>
void FirstOrderAdvector::advect_operator(CellIterator iter,
                		            const IntVector dir,
                                        advectVarBasket* vb,
                                        T& vel_FC,
                		            const CCVariable<V>& q_CC,
                                        CCVariable<V>& q_advected,
                                        F save_q_FC)
{
  const double delT = vb->delT;
 
  Vector cell_dx = vb->patch->dCell();
  double dy = cell_dx[dir[1]];
  double dz = cell_dx[dir[2]];
  double dy_dz = dy * dz;
  
  int p_dir = dir[0];  // principal axes.

  double d_SMALL_NUM = 1e-100;
  
  
  
  for(;!iter.done(); iter++){
    //__________________________________
    //   #0          
    IntVector c = *iter;
    IntVector donor   = c;
    IntVector receive = c;
  
    double velFC   = vel_FC[c];
    double dx      = velFC * delT;   // vel_FC on the bottom, left, back cell face
    double fluxVol = dx * dy_dz;
    //__________________________________
    //    #1
    // Find the donor/receiving cell indicies
    double plus_minus_half  = 0.5 * (velFC + d_SMALL_NUM)/fabs(velFC + d_SMALL_NUM);
    int minus_one_or_zero   = int(-0.5 - plus_minus_half);
    int one_or_zero         = abs(minus_one_or_zero);
    
    donor[p_dir]   = c[p_dir] + minus_one_or_zero;  // donor cell
    receive[p_dir] = c[p_dir] + one_or_zero - 1;    // receiving cell
    //__________________________________
    //   #2
    V q_faceFlux          = q_CC[donor] * fluxVol;
    q_advected[donor]    -= q_faceFlux;
    q_advected[receive]  += q_faceFlux;
    //__________________________________
    //  inline function to save crossing the cell face
    //save_q_FC(q_FC[c], q_CC[up]);
  } 

}
/*_____________________________________________________________________
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template <class T, typename F> 
  void FirstOrderAdvector::advectSlabs(const CCVariable<T>& q_CC,             
                                       CCVariable<T>& q_advected,
                                       advectVarBasket* vb,
                                       constSFCXVariable<double>& uvel_FC,
                                       constSFCYVariable<double>& vvel_FC,
                                       constSFCZVariable<double>& wvel_FC,
                                       SFCXVariable<double>& q_XFC,
                                       SFCYVariable<double>& q_YFC,
                                       SFCZVariable<double>& q_ZFC,
                                       F save_q_FC) // function is passed in
{
  const Patch* patch = vb->patch;
  CellIterator XFC_iter = patch->getSFCXIterator();
  CellIterator YFC_iter = patch->getSFCYIterator();
  CellIterator ZFC_iter = patch->getSFCZIterator();
  
  vector<IntVector> dir(3);
  dir[0] = IntVector(0, 1, 2);   // x,y,z faces
  dir[1] = IntVector(1, 2, 0);   // y,z,x faces
  dir[2] = IntVector(2, 0, 1);   // z,x,y faces
  
  T zero(0.0);
  q_advected.initialize(zero);

  
  advect_operator<constSFCXVariable<double >, T>(XFC_iter, dir[0], vb, uvel_FC,
                                      q_CC, q_advected,
                                      save_q_FC); 

  advect_operator<constSFCYVariable<double>, T>(YFC_iter, dir[1], vb, vvel_FC,
                                      q_CC, q_advected,
                                      save_q_FC); 

  advect_operator<constSFCZVariable<double>, T>(ZFC_iter, dir[2], vb, wvel_FC,
                                      q_CC, q_advected,
                                      save_q_FC);
}

#endif 
/*===========TESTING==========`*/
/*_____________________________________________________________________
 Function~  q_FC_fluxes
 Computes the sum(flux of q at the face center) over all subcycle timesteps
 on the fine level.  We only *need* to hit the cell that are on a coarse-fine 
 interface, ignoring the extraCells.  However, this routine computes 
 the fluxes over the entire computational domain, which could be slow.
 Version r29970 has the fluxes computed on the fine level at the coarse
 fine interfaces.  You need to add the same computation on the coarse 
 level. Note that on the coarse level you don't know where the coarse fine
 interfaces are and need to look up one level to find the interfaces.
_____________________________________________________________________*/
template<class T>
void FirstOrderAdvector::q_FC_fluxes( const CCVariable<T>& q_CC,
                                      const string& desc,
                                      advectVarBasket* vb)
{
   
}
