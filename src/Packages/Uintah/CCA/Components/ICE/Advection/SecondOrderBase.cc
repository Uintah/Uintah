#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderBase.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/Endian.h>
#include <iostream>

using namespace Uintah;


SecondOrderBase::SecondOrderBase() 
{
}

//______________________________________________________________________
//
template <class T> void SecondOrderBase::gradientLimiter(const CCVariable<T>& q_CC,
                                              const Patch* patch,
                                              CCVariable<T>& grad_lim, 
                                              const CCVariable<T>& q_grad_x,
                                              const CCVariable<T>& q_grad_y,
                                              const CCVariable<T>& q_grad_z,
						    T unit, T SN,
			                         DataWarehouse* new_dw)
{
  T  frac,temp, zero(0.);
  T  grad_lim_max, grad_lim_min;

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
                  q_vrtx_max, q_vrtx_min, new_dw);

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
//______________________________________________________________________
//    Function~  q_CCMaxMin
//    This calculates the max and min values of q_CC from the surrounding
//    cells
template <class T> void SecondOrderBase::q_CCMaxMin(const CCVariable<T>& q_CC,
                                                        const Patch* patch,
						              CCVariable<T>& q_CC_max, 
						              CCVariable<T>& q_CC_min)
{  
  T q_CC_max_tmp, q_CC_min_tmp;
  IntVector adjcell;

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
    
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    IntVector c = *iter;
    int i = c.x();
    int j = c.y();
    int k = c.z();

    adjcell = IntVector(i-1, j, k);
    q_CC_max_tmp = q_CC[adjcell];
    q_CC_min_tmp = q_CC[adjcell];
    adjcell = IntVector(i+1, j, k);
    q_CC_max_tmp = Max(q_CC_max_tmp, q_CC[adjcell]);
    q_CC_min_tmp = Min(q_CC_min_tmp, q_CC[adjcell]);
    adjcell = IntVector(i, j-1, k);
    q_CC_max_tmp = Max(q_CC_max_tmp, q_CC[adjcell]);
    q_CC_min_tmp = Min(q_CC_min_tmp, q_CC[adjcell]);
    adjcell = IntVector(i, j+1, k);
    q_CC_max_tmp = Max(q_CC_max_tmp, q_CC[adjcell]);
    q_CC_min_tmp = Min(q_CC_min_tmp, q_CC[adjcell]);
    adjcell = IntVector(i, j, k-1);
    q_CC_max_tmp = Max(q_CC_max_tmp, q_CC[adjcell]);
    q_CC_min_tmp = Min(q_CC_min_tmp, q_CC[adjcell]);
    adjcell = IntVector(i, j, k+1);
    q_CC_max_tmp = Max(q_CC_max_tmp, q_CC[adjcell]);
    q_CC_min_tmp = Min(q_CC_min_tmp, q_CC[adjcell]);

    q_CC_max[c] = q_CC_max_tmp;
    q_CC_min[c] = q_CC_min_tmp;
  }  
}						    

//______________________________________________________________________
template <class T> void SecondOrderBase::q_vertexMaxMin(const CCVariable<T>& q_CC,
                                              const Patch* patch,
                                              const CCVariable<T>& q_grad_x,
                                              const CCVariable<T>& q_grad_y,
                                              const CCVariable<T>& q_grad_z,
						    CCVariable<T>& q_vrtx_max, 
						    CCVariable<T>& q_vrtx_min,
			                         DataWarehouse* new_dw)
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
//______________________________________________________________________
//
template <class T> void SecondOrderBase::gradQ(const CCVariable<T>& q_CC,
                                              const Patch* patch,
						    CCVariable<T>& q_grad_x,
						    CCVariable<T>& q_grad_y,
						    CCVariable<T>& q_grad_z)
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
     int i = c.x();
     int j = c.y();
     int k = c.z();

     adjcell1 = IntVector(i+1, j, k);
     adjcell2 = IntVector(i-1, j, k);
     q_grad_x[c] = (q_CC[adjcell1] - q_CC[adjcell2]) * inv_2delX;

     adjcell1 = IntVector(i, j+1, k);
     adjcell2 = IntVector(i, j-1, k);
     q_grad_y[c] = (q_CC[adjcell1] - q_CC[adjcell2]) * inv_2delY;

     adjcell1 = IntVector(i, j, k+1);
     adjcell2 = IntVector(i, j, k-1);
     q_grad_z[c] = (q_CC[adjcell1] - q_CC[adjcell2]) * inv_2delZ;
   }
}
