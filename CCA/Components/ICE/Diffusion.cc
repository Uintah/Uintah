/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/ICE/Diffusion.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Exceptions/InternalError.h>

#define SMALL_NUM 1.0e-100

#include <typeinfo>
#include <Core/Util/DebugStream.h>
using namespace Uintah;
namespace Uintah {
/* ---------------------------------------------------------------------
 Function~  scalarDiffusionOperator--
 Purpose~   compute diffusion of scalar q
     Note: the bool flag includeVolFrac dictates whether or not
     the volume fraction is include inside the gradient calculation.
 ---------------------------------------------------------------------  */
void scalarDiffusionOperator(DataWarehouse* new_dw,
                                  const Patch* patch,
                                  const bool use_vol_frac,
                                  const CCVariable<double>& q_CC,  
                                  const CCVariable<double>& vol_frac_CC,
                                  CCVariable<double>& q_diffusion_src,
                                  const CCVariable<double>& diff_coeff,
                                  const double delT)
{
  //__________________________________
  //  bullet proofing against AMR
  const Level* level = patch->getLevel();
  if (level->getIndex() > 0) {
    throw InternalError("AMRICE:scalarDiffusionOperator, computational footprint"
                        " has not been tested ", __FILE__, __LINE__ );
  }
  
  SFCXVariable<double> q_X_FC;
  SFCYVariable<double> q_Y_FC;
  SFCZVariable<double> q_Z_FC;
  IntVector right, left, top, bottom, front, back;
  Vector dx = patch->dCell();
  double areaX = dx.y() * dx.z();
  double areaY = dx.x() * dx.z();
  double areaZ = dx.x() * dx.y();

  q_flux_allFaces( new_dw, patch, use_vol_frac, q_CC, diff_coeff,
                   vol_frac_CC, q_X_FC, q_Y_FC, q_Z_FC);
                   
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    right  = c + IntVector(1,0,0);    left   = c ;    
    top    = c + IntVector(0,1,0);    bottom = c ;    
    front  = c + IntVector(0,0,1);    back   = c ; 

    q_diffusion_src[c]=-((q_X_FC[right] - q_X_FC[left])  *areaX + 
                         (q_Y_FC[top]   - q_Y_FC[bottom])*areaY +
                         (q_Z_FC[front] - q_Z_FC[back])  *areaZ )*delT;                  
  }
}

/* ---------------------------------------------------------------------
 Function~  q_flux_FC--
 Purpose~   compute diffusion flux of q at a face
            if use_vol_frac is true then include the volume fraction in
            the flux calculation.
 ---------------------------------------------------------------------  */
template <class T> 
  void q_flux_FC(CellIterator iter, 
                 IntVector adj_offset,
                 const CCVariable<double>& diff_coeff,
                 const double dx,
                 const CCVariable<double>& vol_frac_CC,
                 const CCVariable<double>& q_CC,
                 T& q_fluxFC,
                 const bool use_vol_frac)
{
  //__________________________________
  //  For variable diff_coeff use
  //  diff_coeff_FC = 2 * k[L] * k[R]/ ( k[R] + k[L])
  
  if(use_vol_frac) {
    for(;!iter.done(); iter++){
      IntVector R = *iter;
      IntVector L = R + adj_offset;
      
      double d_c_L = (diff_coeff[L]*vol_frac_CC[L]);
      double d_c_R = (diff_coeff[R]*vol_frac_CC[R]);
      double diff_coeff_FC = (2.0 * d_c_L * d_c_R )/( d_c_L + d_c_R + SMALL_NUM);

      q_fluxFC[R] = -diff_coeff_FC* (q_CC[R] - q_CC[L])/dx;
    }
  }else
   for(;!iter.done(); iter++){
      IntVector R = *iter;
      IntVector L = R + adj_offset;
      
      double diff_coeff_FC = (2.0 * diff_coeff[L] * diff_coeff[R] )/
                                   (diff_coeff[L] + diff_coeff[R] + SMALL_NUM);
                                   
      q_fluxFC[R] = -diff_coeff_FC* (q_CC[R] - q_CC[L])/dx;
    }
}

/* ---------------------------------------------------------------------
 Function~  q_flux_allFaces--
 Purpose~   computes the diffusion flux of q on all faces
 ---------------------------------------------------------------------  */
void q_flux_allFaces(DataWarehouse* new_dw,
                     const Patch* patch,
                     const bool use_vol_frac,   
                     const CCVariable<double>& q_CC,
                     const CCVariable<double>& diff_coeff,
                     const CCVariable<double>& vol_frac_CC,
                     SFCXVariable<double>& q_X_FC,
                     SFCYVariable<double>& q_Y_FC,
                     SFCZVariable<double>& q_Z_FC)
{
  Vector dx = patch->dCell();
  vector<IntVector> adj_offset(3);
  adj_offset[0] = IntVector(-1, 0, 0);    // X faces
  adj_offset[1] = IntVector(0, -1, 0);    // Y faces
  adj_offset[2] = IntVector(0,  0, -1);   // Z faces

  new_dw->allocateTemporary(q_X_FC, patch, Ghost::AroundCells, 1);
  new_dw->allocateTemporary(q_Y_FC, patch, Ghost::AroundCells, 1);
  new_dw->allocateTemporary(q_Z_FC, patch, Ghost::AroundCells, 1);

  q_X_FC.initialize(-9e30);
  q_Y_FC.initialize(-9e30);
  q_Z_FC.initialize(-9e30);

  //__________________________________
  // For multipatch problems adjust the iter limits
  // on the (left/bottom/back) patches to 
  // include the (right/top/front) faces
  // of the cells at the patch boundary. 
  // We compute q_X[right]-q_X[left] on each patch
  IntVector low,hi; 
  IntVector offset = IntVector(1,1,1) - patch->noNeighborsHigh();
       
  low = patch->getSFCXIterator().begin();    // X Face iterator
  hi  = patch->getSFCXIterator().end();
  hi[0] += offset[0];
  CellIterator X_FC_iterLimits(low,hi);
         
  low = patch->getSFCYIterator().begin();   // Y Face iterator
  hi  = patch->getSFCYIterator().end();
  hi[1] += offset[1];

  CellIterator Y_FC_iterLimits(low,hi); 
        
  low = patch->getSFCZIterator().begin();   // Z Face iterator
  hi  = patch->getSFCZIterator().end();
  hi[2] += offset[2];
  
  CellIterator Z_FC_iterLimits(low,hi);            
  //__________________________________
  //  For each face the diffusion flux
  q_flux_FC<SFCXVariable<double> >(X_FC_iterLimits,
                                   adj_offset[0],  diff_coeff, dx.x(),
                                   vol_frac_CC, q_CC, q_X_FC, use_vol_frac);

  q_flux_FC<SFCYVariable<double> >(Y_FC_iterLimits,
                                   adj_offset[1], diff_coeff, dx.y(),
                                   vol_frac_CC, q_CC, q_Y_FC, use_vol_frac);
  
  q_flux_FC<SFCZVariable<double> >(Z_FC_iterLimits,
                                   adj_offset[2],  diff_coeff, dx.z(),
                                   vol_frac_CC, q_CC, q_Z_FC, use_vol_frac); 
}
/*---------------------------------------------------------------------
 Function~  ICE::computeTauX_Components
 Purpose:   This function computes shear stress tau_xx, ta_xy, tau_xz 
 
  Note:   - The edge velocities are defined as the average velocity 
            of the 4 cells surrounding that edge, however we only use 2 cells
            to compute it.  When you take the difference of the edge velocities
            there are two common cells that automatically cancel themselves out.
          - The viscosity we're using isn't right if it varies spatially.   
 ---------------------------------------------------------------------  */
void computeTauX( const Patch* patch,
                  constCCVariable<double>& vol_frac_CC,  
                  constCCVariable<Vector>& vel_CC,      
                  const CCVariable<double>& viscosity,                
                  const Vector dx,                       
                  SFCXVariable<Vector>& tau_X_FC)        
{
  //__________________________________
  //  bullet proofing against AMR
  const Level* level = patch->getLevel();
  if (level->getIndex() > 0) {
    throw InternalError("AMRICE:computeTauX, computational footprint "
                        " has not been tested ", __FILE__, __LINE__ );
  }
  
  double term1, term2, grad_1, grad_2;
  double grad_uvel, grad_vvel, grad_wvel;
  //__________________________________
  // loop over the left cell faces
  // For multipatch problems adjust the iter limits
  // on the left patches to include the right face
  // of the cell at the patch boundary. 
  // We compute tau_ZZ[right]-tau_XX[left] on each patch
  CellIterator hi_lo = patch->getSFCXIterator();
  IntVector low,hi; 
  low = hi_lo.begin();
  hi  = hi_lo.end();
  hi[0] += patch->getBCType(patch->xplus) ==Patch::Neighbor?1:0; 
  CellIterator iterLimits(low,hi); 
  
  for(CellIterator iter = iterLimits;!iter.done();iter++){ 
    IntVector c = *iter;
    int i = c.x();
    int j = c.y();
    int k = c.z();

    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    IntVector left(i-1, j, k); 

    term1 =  viscosity[left] * vol_frac_CC[left];
    term2 =  viscosity[c] * vol_frac_CC[c];
    double vis_FC = (2.0 * term1 * term2)/(term1 + term2); 

    //__________________________________
    // - find indices of surrounding cells
    // - compute velocities at cell face edges see note above.
    double uvel_EC_top    = (vel_CC[IntVector(i-1,j+1,k  )].x()   + 
                             vel_CC[IntVector(i  ,j+1,k  )].x()   +
                             vel_CC[IntVector(i,  j  ,k  )].x()   +
                             vel_CC[IntVector(i-1,j  ,k  )].x() )/4.0; 
                                
    double uvel_EC_bottom = (vel_CC[IntVector(i-1,j-1,k  )].x()   + 
                             vel_CC[IntVector(i  ,j-1,k  )].x()   +      
                             vel_CC[IntVector(i,  j  ,k  )].x()   +
                             vel_CC[IntVector(i-1,j  ,k  )].x() )/4.0; 
                              
    double uvel_EC_front  = (vel_CC[IntVector(i-1,j  ,k+1)].x()   + 
                             vel_CC[IntVector(i  ,j  ,k+1)].x()   +
                             vel_CC[IntVector(i  ,j  ,k  )].x()   +
                             vel_CC[IntVector(i-1,j  ,k  )].x() )/4.0;
                             
    double uvel_EC_back   = (vel_CC[IntVector(i-1,j  ,k-1)].x()   + 
                             vel_CC[IntVector(i  ,j  ,k-1)].x()   +
                             vel_CC[IntVector(i  ,j  ,k  )].x()   +
                             vel_CC[IntVector(i-1,j  ,k  )].x())/4.0;
                             
    double vvel_EC_top    = (vel_CC[IntVector(i-1,j+1,k  )].y()   + 
                             vel_CC[IntVector(i  ,j+1,k  )].y()   +
                             vel_CC[IntVector(i  ,j  ,k  )].y()   +
                             vel_CC[IntVector(i-1,j  ,k  )].y())/4.0;
                             
    double vvel_EC_bottom = (vel_CC[IntVector(i-1,j-1,k  )].y()   + 
                             vel_CC[IntVector(i  ,j-1,k  )].y()   +
                             vel_CC[IntVector(i  ,j  ,k  )].y()   +
                             vel_CC[IntVector(i-1,j  ,k  )].y())/4.0;
                             
    double wvel_EC_front  = (vel_CC[IntVector(i-1,j  ,k+1)].z()   + 
                             vel_CC[IntVector(i  ,j  ,k+1)].z()   +
                             vel_CC[IntVector(i  ,j  ,k  )].z()   +
                             vel_CC[IntVector(i-1,j  ,k  )].z())/4.0;
                             
    double wvel_EC_back   = (vel_CC[IntVector(i-1,j  ,k-1)].z()   + 
                             vel_CC[IntVector(i  ,j  ,k-1)].z()   +
                             vel_CC[IntVector(i  ,j  ,k  )].z()   +
                             vel_CC[IntVector(i-1,j  ,k  )].z())/4.0;
    //__________________________________
    //  tau_XX
    grad_uvel = (vel_CC[c].x() - vel_CC[left].x())/delX;
    grad_vvel = (vvel_EC_top   - vvel_EC_bottom)  /delY;
    grad_wvel = (wvel_EC_front - wvel_EC_back )   /delZ;

    term1 = 2.0 * vis_FC * grad_uvel;
    term2 = (2.0/3.0) * vis_FC * (grad_uvel + grad_vvel + grad_wvel);
    tau_X_FC[c].x(term1 - term2); 

    //__________________________________
    //  tau_XY
    grad_1 = (uvel_EC_top   - uvel_EC_bottom)  /delY;
    grad_2 = (vel_CC[c].y() - vel_CC[left].y())/delX;
    tau_X_FC[c].y(vis_FC * (grad_1 + grad_2)); 

    //__________________________________
    //  tau_XZ
    grad_1 = (uvel_EC_front - uvel_EC_back)    /delZ;
    grad_2 = (vel_CC[c].z() - vel_CC[left].z())/delX;
    tau_X_FC[c].z(vis_FC * (grad_1 + grad_2)); 

    #if 0
    if (c == IntVector(0,51,0) || c == IntVector(50,51,0)){
      cout<<c<<" tau_XX: "<<tau_X_FC[c].x()<<
      " tau_XY: "<<tau_X_FC[c].y()<<
      " tau_XZ: "<<tau_X_FC[c].z()<<
      " patch: " <<patch->getID()<<endl;     
    } 
    #endif
  }
}


/*---------------------------------------------------------------------
 Function~  ICE::computeTauY_Components
 Purpose:   This function computes shear stress tau_YY, ta_yx, tau_yz 
  Note:   - The edge velocities are defined as the average velocity 
            of the 4 cells surrounding that edge, however we only use2 cells
            to compute it.  When you take the difference of the edge velocities
            there are two common cells that automatically cancel themselves out.
          - The viscosity we're using isn't right if it varies spatially. 
 ---------------------------------------------------------------------  */
void computeTauY( const Patch* patch,
                  constCCVariable<double>& vol_frac_CC,   
                  constCCVariable<Vector>& vel_CC,      
                  const CCVariable<double>& viscosity,                
                  const Vector dx,                       
                  SFCYVariable<Vector>& tau_Y_FC)        
{
  //__________________________________
  //  bullet proofing against AMR
  const Level* level = patch->getLevel();
  if (level->getIndex() > 0) {
    throw InternalError("AMRICE:computeTauY, computational footprint"
                        " has not been tested ", __FILE__, __LINE__ );
  }
  
  double term1, term2, grad_1, grad_2;
  double grad_uvel, grad_vvel, grad_wvel;
  //__________________________________
  // loop over the bottom cell faces
  // For multipatch problems adjust the iter limits
  // on the bottom patches to include the top face
  // of the cell at the patch boundary. 
  // We compute tau_YY[top]-tau_YY[bot] on each patch
  CellIterator hi_lo = patch->getSFCYIterator();
  IntVector low,hi; 
  low = hi_lo.begin();
  hi  = hi_lo.end();
  hi[1] += patch->getBCType(patch->yplus) ==Patch::Neighbor?1:0; 
  CellIterator iterLimits(low,hi); 
  
  for(CellIterator iter = iterLimits;!iter.done();iter++){ 
    IntVector c = *iter;
    int i = c.x();
    int j = c.y();
    int k = c.z();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    IntVector bottom(i,j-1,k);
    
    term1 =  viscosity[bottom] * vol_frac_CC[bottom];
    term2 =  viscosity[c] * vol_frac_CC[c];
    double vis_FC = (2.0 * term1 * term2)/(term1 + term2); 

    //__________________________________
    // - find indices of surrounding cells
    // - compute velocities at cell face edges see note above.
    double uvel_EC_right = (vel_CC[IntVector(i+1,j,  k  )].x()  + 
                            vel_CC[IntVector(i+1,j-1,k  )].x()  +
                            vel_CC[IntVector(i  ,j-1,k  )].x()  +
                            vel_CC[IntVector(i  ,j  ,k  )].x())/4.0;
                             
    double uvel_EC_left  = (vel_CC[IntVector(i-1,j,  k  )].x()  +
                            vel_CC[IntVector(i-1,j-1,k  )].x()  +
                            vel_CC[IntVector(i  ,j-1,k  )].x()  +
                            vel_CC[IntVector(i  ,j  ,k  )].x())/4.0;
                            
    double vvel_EC_right = (vel_CC[IntVector(i+1,j  ,k  )].y()  + 
                            vel_CC[IntVector(i+1,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j  ,k  )].y())/4.0;
                            
    double vvel_EC_left  = (vel_CC[IntVector(i-1,j  ,k  )].y()  + 
                            vel_CC[IntVector(i-1,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j  ,k  )].y())/4.0; 
                            
    double vvel_EC_front = (vel_CC[IntVector(i  ,j  ,k+1)].y()  +
                            vel_CC[IntVector(i  ,j-1,k+1)].y()  +
                            vel_CC[IntVector(i  ,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j  ,k  )].y())/4.0;
                            
    double vvel_EC_back  = (vel_CC[IntVector(i  ,j  ,k-1)].y()  +
                            vel_CC[IntVector(i  ,j-1,k-1)].y()  +
                            vel_CC[IntVector(i  ,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j  ,k  )].y())/4.0;
                            
    double wvel_EC_front = (vel_CC[IntVector(i  ,j  ,k+1)].z()  +
                            vel_CC[IntVector(i  ,j-1,k+1)].z()  +
                            vel_CC[IntVector(i  ,j-1,k  )].z()  +
                            vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
                            
    double wvel_EC_back  = (vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                            vel_CC[IntVector(i  ,j-1,k-1)].z()  +
                            vel_CC[IntVector(i  ,j-1,k  )].z()  +
                            vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
                            
    //__________________________________
    //  tau_YY
    grad_uvel = (uvel_EC_right - uvel_EC_left)      /delX;
    grad_vvel = (vel_CC[c].y() - vel_CC[bottom].y())/delY;
    grad_wvel = (wvel_EC_front - wvel_EC_back )     /delZ;

    term1 = 2.0 * vis_FC * grad_vvel;
    term2 = (2.0/3.0) * vis_FC * (grad_uvel + grad_vvel + grad_wvel);
    tau_Y_FC[c].y(term1 - term2); 
    
    //__________________________________
    //  tau_YX
    grad_1 = (vel_CC[c].x() - vel_CC[bottom].x())/delY;
    grad_2 = (vvel_EC_right - vvel_EC_left)      /delX;
    tau_Y_FC[c].x( vis_FC * (grad_1 + grad_2) ); 


    //__________________________________
    //  tau_YZ
    grad_1 = (vvel_EC_front - vvel_EC_back)      /delZ;
    grad_2 = (vel_CC[c].z() - vel_CC[bottom].z())/delY;
    tau_Y_FC[c].z( vis_FC * (grad_1 + grad_2)); 
    
    #if 0
     if (c == IntVector(0,51,0) || c == IntVector(50,51,0)){    
       cout<< c<< " tau_YX: "<<tau_Y_FC[c].x()<<
       " tau_YY: "<<tau_Y_FC[c].y()<<
       " tau_YZ: "<<tau_Y_FC[c].z()<<
       " patch: "<<patch->getID()<<endl;
     }
    #endif
  }
}

/*---------------------------------------------------------------------
 Function~  ICE::computeTauZ
 Purpose:   This function computes shear stress tau_zx, ta_zy, tau_zz 
  Note:   - The edge velocities are defined as the average velocity 
            of the 4 cells surrounding that edge, however we only use 2 cells
            to compute it.  When you take the difference of the edge velocities
            there are two common cells that automatically cancel themselves out.
          - The viscosity we're using isn't right if it varies spatially.
 ---------------------------------------------------------------------  */
void computeTauZ( const Patch* patch,
                  constCCVariable<double>& vol_frac_CC,   
                  constCCVariable<Vector>& vel_CC,      
                  const CCVariable<double>& viscosity,               
                  const Vector dx,                       
                  SFCZVariable<Vector>& tau_Z_FC)        
{
  //__________________________________
  //  bullet proofing against AMR
  const Level* level = patch->getLevel();
  if (level->getIndex() > 0) {
    throw InternalError("AMRICE:computeTauZ, computational footprint"
                        " has not been tested ", __FILE__, __LINE__ );
  }
  double term1, term2, grad_1, grad_2;
  double grad_uvel, grad_vvel, grad_wvel;
 
  //__________________________________
  // loop over the back cell faces
  // For multipatch problems adjust the iter limits
  // on the back patches to include the front face
  // of the cell at the patch boundary. 
  // We compute tau_ZZ[front]-tau_ZZ[back] on each patch
  CellIterator hi_lo = patch->getSFCZIterator();
  IntVector low,hi; 
  low = hi_lo.begin();
  hi  = hi_lo.end();
  hi[2] += patch->getBCType(patch->zplus) ==Patch::Neighbor?1:0; 
  CellIterator iterLimits(low,hi); 

  for(CellIterator iter = iterLimits;!iter.done();iter++){ 
    IntVector c = *iter; 
    int i = c.x();
    int j = c.y();
    int k = c.z();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    IntVector back(i, j, k-1); 
    
    term1 =  viscosity[back] * vol_frac_CC[back];
    term2 =  viscosity[c] * vol_frac_CC[c];
    double vis_FC = (2.0 * term1 * term2)/(term1 + term2); 

    //__________________________________
    // - find indices of surrounding cells
    // - compute velocities at cell face edges see note above.
    double uvel_EC_right  = (vel_CC[IntVector(i+1,j,  k  )].x()  + 
                             vel_CC[IntVector(i+1,j,  k-1)].x()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].x()  +
                             vel_CC[IntVector(i  ,j  ,k  )].x())/4.0;
                             
    double uvel_EC_left   = (vel_CC[IntVector(i-1,j,  k  )].x()  +
                             vel_CC[IntVector(i-1,j  ,k-1)].x()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].x()  +
                             vel_CC[IntVector(i  ,j  ,k  )].x())/4.0;
                             
    double vvel_EC_top    = (vel_CC[IntVector(i  ,j+1,k  )].y()  + 
                             vel_CC[IntVector(i  ,j+1,k-1)].y()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].y()  +
                             vel_CC[IntVector(i  ,j  ,k  )].y())/4.0;
                             
    double vvel_EC_bottom = (vel_CC[IntVector(i  ,j-1,k  )].y()  + 
                             vel_CC[IntVector(i  ,j-1,k-1)].y()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].y()  +
                             vel_CC[IntVector(i  ,j  ,k  )].y())/4.0;
                             
    double wvel_EC_right  = (vel_CC[IntVector(i+1,j,  k  )].z()  + 
                             vel_CC[IntVector(i+1,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
                             
    double wvel_EC_left   = (vel_CC[IntVector(i-1,j,  k  )].z()  +
                             vel_CC[IntVector(i-1,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
                             
    double wvel_EC_top    = (vel_CC[IntVector(i  ,j+1,k  )].z()  + 
                             vel_CC[IntVector(i  ,j+1,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
                             
    double wvel_EC_bottom = (vel_CC[IntVector(i  ,j-1,k  )].z()  + 
                             vel_CC[IntVector(i  ,j-1,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
    //__________________________________
    //  tau_ZX
    grad_1 = (vel_CC[c].x() - vel_CC[back].x()) /delZ;
    grad_2 = (wvel_EC_right - wvel_EC_left)     /delX;
    tau_Z_FC[c].x(vis_FC * (grad_1 + grad_2)); 

    //__________________________________
    //  tau_ZY
    grad_1 = (vel_CC[c].y() - vel_CC[back].y()) /delZ;
    grad_2 = (wvel_EC_top   - wvel_EC_bottom)   /delY;
    tau_Z_FC[c].y( vis_FC * (grad_1 + grad_2) ); 

    //__________________________________
    //  tau_ZZ
    grad_uvel = (uvel_EC_right - uvel_EC_left)    /delX;
    grad_vvel = (vvel_EC_top   - vvel_EC_bottom)  /delY;
    grad_wvel = (vel_CC[c].z() - vel_CC[back].z())/delZ;

    term1 = 2.0 * vis_FC * grad_wvel;
    term2 = (2.0/3.0) * vis_FC * (grad_uvel + grad_vvel + grad_wvel);
    tau_Z_FC[c].z( (term1 - term2)); 

//  cout<<"tau_ZX: "<<tau_Z_FC[cell].x()<<
//        " tau_ZY: "<<tau_Z_FC[cell].y()<<
//        " tau_ZZ: "<<tau_Z_FC[cell].z()<<endl;
  }
}

}  // using namespace Uintah
