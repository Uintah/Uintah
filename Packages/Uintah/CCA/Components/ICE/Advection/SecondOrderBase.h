#ifndef UINTAH_SECOND_ORDER_BASE_H
#define UINTAH_SECOND_ORDER_BASE_H

#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>

#define d_SMALL_NUM 1e-100
namespace Uintah {

class SecondOrderBase {

public:
  SecondOrderBase();
  virtual ~SecondOrderBase();
                           
  template<class T> void gradientLimiter(const CCVariable<T>& q_CC,
                                         const Patch* patch,
                                         CCVariable<T>& grad_lim,
                                         const CCVariable<T>& q_grad_x,
                                         const CCVariable<T>& q_grad_y,
                                         const CCVariable<T>& q_grad_z,
                                         DataWarehouse* new_dw);
                      
  template<class T> void q_CCMaxMin(const CCVariable<T>& q_CC,
                                    const Patch* patch,
                                    CCVariable<T>& q_CC_max, 
                                    CCVariable<T>& q_CC_min);
                                           
  template<class T> void q_vertexMaxMin(const CCVariable<T>& q_CC,
                                        const Patch* patch,
                                        const CCVariable<T>& q_grad_x,
                                        const CCVariable<T>& q_grad_y,
                                        const CCVariable<T>& q_grad_z,
                                        CCVariable<T>& q_vrtx_max, 
                                        CCVariable<T>& q_vrtx_min);
                                    
  template<class T> void gradQ(const CCVariable<T>& q_CC,
                               const Patch* patch,
                               CCVariable<T>& q_grad_x,
                               CCVariable<T>& q_grad_y,
                               CCVariable<T>& q_grad_z);
                                  
  CCVariable<fflux>  r_out_x, r_out_y, r_out_z;

};
/* ---------------------------------------------------------------------
 Function~ gradientLimiter
_____________________________________________________________________*/
template <class T>
void
SecondOrderBase::gradientLimiter(const CCVariable<T>& q_CC,
				 const Patch* patch,
				 CCVariable<T>& grad_lim, 
				 const CCVariable<T>& q_grad_x,
				 const CCVariable<T>& q_grad_y,
				 const CCVariable<T>& q_grad_z,
                             DataWarehouse* new_dw)
{
  T  frac,temp, zero(0.);
  T  grad_lim_max, grad_lim_min;
  T unit = T(1.0);
  T SN = T(d_SMALL_NUM);
  
  CCVariable<T> q_vrtx_max;
  CCVariable<T> q_vrtx_min;
  CCVariable<T> q_CC_max;
  CCVariable<T> q_CC_min;

  Ghost::GhostType  gac = Ghost::AroundCells;  
  new_dw->allocateTemporary(q_CC_max,   patch, gac, 1);
  new_dw->allocateTemporary(q_CC_min,   patch, gac, 1);
  new_dw->allocateTemporary(q_vrtx_max, patch, gac, 1);  
  new_dw->allocateTemporary(q_vrtx_min, patch, gac, 1);  
  //__________________________________
  // find q_cc max and min and
  // vertex max and min
  q_CCMaxMin(    q_CC, patch, q_CC_max, q_CC_min);
  q_vertexMaxMin(q_CC, patch, q_grad_x, q_grad_y, q_grad_z,
		 q_vrtx_max, q_vrtx_min);

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
    
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    IntVector c = *iter;

    T Q_CC = q_CC[c];
    frac = (q_CC_max[c] - Q_CC + SN)/(q_vrtx_max[c] - Q_CC + SN);
    grad_lim_max = Max(zero, frac);

    frac = (q_CC_min[c] - Q_CC + SN)/(q_vrtx_min[c] - Q_CC + SN);
    grad_lim_min = Max(zero, frac);
    
    temp = Min(unit, grad_lim_max);
    temp = Min(temp, grad_lim_min);
    grad_lim[c] = temp; 
  }
}
/* ---------------------------------------------------------------------
Function~  q_CCMaxMin
Purpose~   This calculates the max and min values of q_CC from the surrounding
           cells
_____________________________________________________________________*/

template <class T>
void
SecondOrderBase::q_CCMaxMin(const CCVariable<T>& q_CC,
			    const Patch* patch,
			    CCVariable<T>& q_CC_max, 
			    CCVariable<T>& q_CC_min)
{  
  T max_tmp, min_tmp;

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
    
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    IntVector c = *iter;      
    IntVector r = c + IntVector( 1, 0, 0);
    IntVector l = c + IntVector(-1, 0, 0);
    IntVector t = c + IntVector( 0, 1, 0);
    IntVector b = c + IntVector( 0,-1, 0);
    IntVector f = c + IntVector( 0, 0, 1);    
    IntVector bk= c + IntVector( 0, 0,-1); 
  
    max_tmp = Max(q_CC[r], q_CC[l]);
    min_tmp = Min(q_CC[r], q_CC[l]);
    
    max_tmp = Max(max_tmp, q_CC[t]);
    min_tmp = Min(min_tmp, q_CC[t]);
    
    max_tmp = Max(max_tmp, q_CC[b]);
    min_tmp = Min(min_tmp, q_CC[b]);
    
    max_tmp = Max(max_tmp, q_CC[f]);
    min_tmp = Min(min_tmp, q_CC[f]);
    
    max_tmp = Max(max_tmp, q_CC[bk]);
    min_tmp = Min(min_tmp, q_CC[bk]);
   
    q_CC_max[c] = max_tmp;
    q_CC_min[c] = min_tmp;
  }  
}                                                   

/* ---------------------------------------------------------------------
 Function~ q_vertexMaxMin
 Purpose   Find the max and min vertex value of q at the vertex.  
_____________________________________________________________________*/
template <class T>
void
SecondOrderBase::q_vertexMaxMin( const CCVariable<T>& q_CC,
				 const Patch* patch,
				 const CCVariable<T>& q_grad_x,
				 const CCVariable<T>& q_grad_y,
				 const CCVariable<T>& q_grad_z,
				 CCVariable<T>& q_vrtx_max, 
				 CCVariable<T>& q_vrtx_min)
{
  T q_vrtx1, q_vrtx2,q_vrtx3,q_vrtx4,q_vrtx5,q_vrtx6,q_vrtx7,q_vrtx8;
  T q_vrtx_tmp_max, q_vrtx_tmp_min;

  Vector dx = patch->dCell();
  double delX_2 = dx.x()/2.0;
  double delY_2 = dx.y()/2.0;
  double delZ_2 = dx.z()/2.0;

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
    
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    IntVector c = *iter;
    T Q_CC = q_CC[c];
    T xterm = q_grad_x[c] * delX_2;
    T yterm = q_grad_y[c] * delY_2;
    T zterm = q_grad_z[c] * delZ_2;
    //__________________________________
    // THINK ABOUT PULLING OUT QCC  --TODD
    q_vrtx1 = xterm  + yterm + zterm + Q_CC;
    q_vrtx2 = xterm  + yterm - zterm + Q_CC;
    q_vrtx3 = xterm  - yterm + zterm + Q_CC;     
    q_vrtx4 = xterm  - yterm - zterm + Q_CC;
    q_vrtx5 = -xterm + yterm + zterm + Q_CC;
    q_vrtx6 = -xterm + yterm - zterm + Q_CC;     
    q_vrtx7 = -xterm - yterm + zterm + Q_CC;
    q_vrtx8 = -xterm - yterm - zterm + Q_CC;

    q_vrtx_tmp_max = Max(q_vrtx1,q_vrtx2);
    q_vrtx_tmp_min = Min(q_vrtx1,q_vrtx2);
    q_vrtx_tmp_max = Max(q_vrtx3,q_vrtx_tmp_max);
    q_vrtx_tmp_min = Min(q_vrtx3,q_vrtx_tmp_min);
    q_vrtx_tmp_max = Max(q_vrtx4,q_vrtx_tmp_max);
    q_vrtx_tmp_min = Min(q_vrtx4,q_vrtx_tmp_min);
    q_vrtx_tmp_max = Max(q_vrtx5,q_vrtx_tmp_max);
    q_vrtx_tmp_min = Min(q_vrtx5,q_vrtx_tmp_min);
    q_vrtx_tmp_max = Max(q_vrtx6,q_vrtx_tmp_max);
    q_vrtx_tmp_min = Min(q_vrtx6,q_vrtx_tmp_min);
    q_vrtx_tmp_max = Max(q_vrtx7,q_vrtx_tmp_max);
    q_vrtx_tmp_min = Min(q_vrtx7,q_vrtx_tmp_min);
    q_vrtx_tmp_max = Max(q_vrtx8,q_vrtx_tmp_max);
    q_vrtx_tmp_min = Min(q_vrtx8,q_vrtx_tmp_min);

    q_vrtx_max[c] = q_vrtx_tmp_max;
    q_vrtx_min[c] = q_vrtx_tmp_min; 
  }    
}
/* ---------------------------------------------------------------------
 Function~ gradQ
 Purpose   Find the x, y, z gradients of q_CC. centered
           differencing 
_____________________________________________________________________*/
template <class T>
void
SecondOrderBase::gradQ( const CCVariable<T>& q_CC,
			const Patch* patch,
			CCVariable<T>& q_grad_x,
			CCVariable<T>& q_grad_y,
			CCVariable<T>& q_grad_z )
{
  Vector dx = patch->dCell();
  IntVector adjcell1, adjcell2;
  double inv_2delX = 1.0/(2.0 * dx.x());
  double inv_2delY = 1.0/(2.0 * dx.y());
  double inv_2delZ = 1.0/(2.0 * dx.z());

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
     
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {   
    IntVector c = *iter;      
    IntVector r = c + IntVector( 1, 0, 0);
    IntVector l = c + IntVector(-1, 0, 0);
    IntVector t = c + IntVector( 0, 1, 0);
    IntVector b = c + IntVector( 0,-1, 0);
    IntVector f = c + IntVector( 0, 0, 1);    
    IntVector bk= c + IntVector( 0, 0,-1);
    
    q_grad_x[c] = (q_CC[r] - q_CC[l]) * inv_2delX;
    q_grad_y[c] = (q_CC[t] - q_CC[b]) * inv_2delY;
    q_grad_z[c] = (q_CC[f] - q_CC[bk])* inv_2delZ;
  }
}

} // end namespace Uintah

#endif
