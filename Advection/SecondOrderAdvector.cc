#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderAdvector.h>
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
#define d_SMALL_NUM 1.0e-100

using namespace Uintah;
using std::cerr;
using std::endl;


SecondOrderAdvector::SecondOrderAdvector() 
{
  OFS_CCLabel = 0;
}


SecondOrderAdvector::SecondOrderAdvector(DataWarehouse* new_dw, 
                                   const Patch* patch)
{
  OFS_CCLabel = VarLabel::create("OFS_CC",
                             CCVariable<fflux>::getTypeDescription());

  new_dw->allocateTemporary(d_OFS,  patch, Ghost::AroundCells,1);
  new_dw->allocateTemporary(r_out_x,   patch, Ghost::AroundCells,1);
  new_dw->allocateTemporary(r_out_y,   patch, Ghost::AroundCells,1);
  new_dw->allocateTemporary(r_out_z,   patch, Ghost::AroundCells,1);
  
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
            This includes the slabs and edge fluxes
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
            
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

void SecondOrderAdvector::inFluxOutFluxVolume(
                           const SFCXVariable<double>& uvel_FC,
                        const SFCYVariable<double>& vvel_FC,
                        const SFCZVariable<double>& wvel_FC,
                        const double& delT, 
                        const Patch* patch,
			   const int& indx)

{
  Vector dx = patch->dCell();

  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom,delX_right, delX_left, delZ_front, delZ_back;
  double delX_tmp, delY_tmp,   delZ_tmp;
  double delX = dx.x(), delY = dx.y(), delZ = dx.z();
  double r_x, r_y, r_z;

  // Compute outfluxes 
  const IntVector gc(1,1,1);
  double error_test = 0.0;
  int    num_cells = 0;
  
  for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
    IntVector curcell = *iter;
 
    delY_top    = std::max(0.0, (vvel_FC[curcell+IntVector(0,1,0)] * delT));
    delY_bottom = std::max(0.0,-(vvel_FC[curcell+IntVector(0,0,0)] * delT));
    delX_right  = std::max(0.0, (uvel_FC[curcell+IntVector(1,0,0)] * delT));
    delX_left   = std::max(0.0,-(uvel_FC[curcell+IntVector(0,0,0)] * delT));
    delZ_front  = std::max(0.0, (wvel_FC[curcell+IntVector(0,0,1)] * delT));
    delZ_back   = std::max(0.0,-(wvel_FC[curcell+IntVector(0,0,0)] * delT));
    
    delX_tmp    = delX - delX_right - delX_left;
    delY_tmp    = delY - delY_top   - delY_bottom;
    delZ_tmp    = delZ - delZ_front - delZ_back;
    
    //__________________________________
    //   SLAB outfluxes
    double delX_Z_tmp = delX_tmp * delZ_tmp;
    double delX_Y_tmp = delX_tmp * delY_tmp;
    double delY_Z_tmp = delY_tmp * delZ_tmp;
    d_OFS[curcell].d_fflux[TOP]   = delY_top   * delX_Z_tmp;
    d_OFS[curcell].d_fflux[BOTTOM]= delY_bottom* delX_Z_tmp;
    d_OFS[curcell].d_fflux[RIGHT] = delX_right * delY_Z_tmp;
    d_OFS[curcell].d_fflux[LEFT]  = delX_left  * delY_Z_tmp;
    d_OFS[curcell].d_fflux[FRONT] = delZ_front * delX_Y_tmp;
    d_OFS[curcell].d_fflux[BACK]  = delZ_back  * delX_Y_tmp; 
    
    //__________________________________
    //  Bullet proofing
    double total_fluxout = 0.0;
      total_fluxout  += d_OFS[curcell].d_fflux[TOP];
      total_fluxout  += d_OFS[curcell].d_fflux[BOTTOM];
      total_fluxout  += d_OFS[curcell].d_fflux[RIGHT];
      total_fluxout  += d_OFS[curcell].d_fflux[LEFT];
      total_fluxout  += d_OFS[curcell].d_fflux[FRONT];
      total_fluxout  += d_OFS[curcell].d_fflux[BACK];
       
      r_x = delX_left/2.0 -  delX_right/2.0;
      r_y = delY_bottom/2.0 - delY_top/2.0;
      r_z = delZ_back/2.0 - delZ_front/2.0;
      
      r_out_x[curcell].d_fflux[RIGHT] = delX/2.0 - delX_right/2.0;
      r_out_y[curcell].d_fflux[RIGHT] = r_y;
      r_out_z[curcell].d_fflux[RIGHT] = r_z;
  
      r_out_x[curcell].d_fflux[LEFT] = delX_left/2.0 - delX/2.0;
      r_out_y[curcell].d_fflux[LEFT] = r_y;
      r_out_z[curcell].d_fflux[LEFT] = r_z;
      
      r_out_x[curcell].d_fflux[TOP] = r_x;
      r_out_y[curcell].d_fflux[TOP] = delY/2.0 - delY_top/2.0;
      r_out_z[curcell].d_fflux[TOP] = r_z;
     
      r_out_x[curcell].d_fflux[BOTTOM] = r_x;
      r_out_y[curcell].d_fflux[BOTTOM] = delY_bottom/2.0 - delY/2.0;
      r_out_z[curcell].d_fflux[BOTTOM] = r_z;
      
      r_out_x[curcell].d_fflux[FRONT] = r_x;
      r_out_y[curcell].d_fflux[FRONT] = r_y;
      r_out_z[curcell].d_fflux[FRONT] = delZ/2.0 - delZ_front/2.0;
     
      r_out_x[curcell].d_fflux[BACK] = r_x;
      r_out_y[curcell].d_fflux[BACK] = r_y;
      r_out_z[curcell].d_fflux[BACK] = delZ_back/2.0 - delZ/2.0;
      
      num_cells++;
      error_test +=(vol - total_fluxout)/fabs(vol- total_fluxout);   

  }//cell iterator

  //__________________________________
  // if total_fluxout > vol then 
  // find the cell and throw an exception.  
  if (fabs(error_test - num_cells) > 1.0e-2) {
    for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
      IntVector curcell = *iter; 
      double total_fluxout = 0.0;
      for(int face = TOP; face <= BACK; face++ )  {
        total_fluxout  += d_OFS[curcell].d_fflux[face];
      }
      if (vol - total_fluxout < 0.0) {
        throw OutFluxVolume(*iter,total_fluxout, vol, indx);
      }
    }  // cell iter
  }  // if total_fluxout > vol  
}



/* ---------------------------------------------------------------------
 Function~  ICE::advectQSecond--ADVECTION:
 Purpose~   Calculate the advection of q_CC 
   
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
            
 Steps for each cell:      
- Compute q outflux and q influx for each cell.
- Finally sum the influx and outflux portions
       
 advect_preprocessing MUST be done prior to this function
 ---------------------------------------------------------------------  */

void SecondOrderAdvector::advectQ(const CCVariable<double>& q_CC,
                             const Patch* patch,
                             CCVariable<double>& q_advected,
			        DataWarehouse* new_dw)
{

  CCVariable<double> grad_lim;
  StaticArray<CCVariable<double> > q_OAFS(6);
  double unit = 1.0;
  double SN = d_SMALL_NUM;
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(grad_lim,   patch);
  for (int face = TOP; face <= BACK; face++) {
    new_dw->allocateTemporary(q_OAFS[face], patch,gac,1);
  }

  gradientLimiter(q_CC, patch, grad_lim, unit, SN, new_dw);
  qAverageFlux(q_CC, patch, grad_lim, q_OAFS, new_dw);
  advect(q_OAFS, patch, q_advected);

}


void SecondOrderAdvector::advectQ(const CCVariable<Vector>& q_CC,
                             const Patch* patch,
                             CCVariable<Vector>& q_advected,
			        DataWarehouse* new_dw)
{
  CCVariable<Vector> grad_lim;
  StaticArray<CCVariable<Vector> > q_OAFS(6);
  Vector unit(1.0,1.0,1.0);
  Vector SN(d_SMALL_NUM,d_SMALL_NUM,d_SMALL_NUM);
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(grad_lim,   patch);
  for (int face = TOP; face <= BACK; face++ ) {
    new_dw->allocateTemporary(q_OAFS[face], patch,gac,1);
  }
  
  gradientLimiter(q_CC, patch, grad_lim, unit, SN, new_dw);
  qAverageFlux(q_CC, patch, grad_lim, q_OAFS, new_dw);
  advect(q_OAFS, patch, q_advected);

}


template <class T> void SecondOrderAdvector::advect(StaticArray<CCVariable<T> >& q_OAFS, 
                                              const Patch* patch,
                                              CCVariable<T>& q_advected)
  
{
  T  sum_q_outflux, sum_q_influx, zero(0.);
  IntVector adjcell;

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    IntVector curcell = *iter;
    int i = curcell.x();
    int j = curcell.y();
    int k = curcell.z();
    
    sum_q_outflux      = zero;
    sum_q_influx       = zero;

    //__________________________________
    //  OUTFLUX: SLAB 
    
    sum_q_outflux  += q_OAFS[BOTTOM][curcell] * d_OFS[curcell].d_fflux[BOTTOM];

    sum_q_outflux  += q_OAFS[TOP][curcell] * d_OFS[curcell].d_fflux[TOP];
    
    sum_q_outflux  += q_OAFS[LEFT][curcell] * d_OFS[curcell].d_fflux[LEFT];

    sum_q_outflux  += q_OAFS[RIGHT][curcell] * d_OFS[curcell].d_fflux[RIGHT];
    
    sum_q_outflux  += q_OAFS[BACK][curcell] * d_OFS[curcell].d_fflux[BACK];

    sum_q_outflux  += q_OAFS[FRONT][curcell] * d_OFS[curcell].d_fflux[FRONT];
    
   
    //__________________________________
    //  INFLUX: SLABS
    adjcell = IntVector(i, j+1, k);       // TOP
    sum_q_influx  += q_OAFS[BOTTOM][adjcell] * d_OFS[adjcell].d_fflux[BOTTOM];

    adjcell = IntVector(i, j-1, k);       // BOTTOM
    sum_q_influx  += q_OAFS[TOP][adjcell] * d_OFS[adjcell].d_fflux[TOP];

    adjcell = IntVector(i+1, j, k);       // RIGHT
    sum_q_influx  += q_OAFS[LEFT][adjcell] * d_OFS[adjcell].d_fflux[LEFT];

    adjcell = IntVector(i-1, j, k);       // LEFT
    sum_q_influx  += q_OAFS[RIGHT][adjcell] * d_OFS[adjcell].d_fflux[RIGHT];

    adjcell = IntVector(i, j, k+1);       // FRONT
    sum_q_influx  += q_OAFS[BACK][adjcell] * d_OFS[adjcell].d_fflux[BACK];

    adjcell = IntVector(i, j, k-1);       // BACK
    sum_q_influx  += q_OAFS[FRONT][adjcell] * d_OFS[adjcell].d_fflux[FRONT];

    //__________________________________
    //  Calculate the advected q at t + delta t
    q_advected[curcell] = sum_q_influx - sum_q_outflux;

  }

}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

template <class T> void SecondOrderAdvector::gradientLimiter(const CCVariable<T>& q_CC,
                                              const Patch* patch,
                                              CCVariable<T>& grad_lim,
						    T unit, T SN,
			                         DataWarehouse* new_dw)
  
{
  T  frac,temp, zero(0.);
  T  grad_lim_max, grad_lim_min;

  CCVariable<T> q_vrtx_max;
  CCVariable<T> q_vrtx_min;
  CCVariable<T> q_CC_max;
  CCVariable<T> q_CC_min;
  
  new_dw->allocateTemporary(q_CC_max,   patch);
  new_dw->allocateTemporary(q_CC_min,   patch);
  new_dw->allocateTemporary(q_vrtx_max,   patch);
  new_dw->allocateTemporary(q_vrtx_min,   patch);

  q_CCMaxMin(q_CC, patch, q_CC_max, q_CC_min);
  q_vertexMaxMin(q_CC, patch, q_vrtx_max, q_vrtx_min, new_dw);
  
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
     IntVector curcell = *iter;

    frac = (q_CC_max[curcell]-q_CC[curcell]+SN)/(q_vrtx_max[curcell]-q_CC[curcell]+SN);
    grad_lim_max = Max(zero, frac);

    frac = (q_CC_min[curcell]-q_CC[curcell]+SN)/(q_vrtx_min[curcell]-q_CC[curcell]+SN);
    grad_lim_min = Max(zero, frac);
    
    temp = Min(unit, grad_lim_max);
    temp = Min(temp, grad_lim_min);
    grad_lim[curcell] = temp;
 
  }
}

template <class T> void SecondOrderAdvector::qAverageFlux(const CCVariable<T>& q_CC,
                                              const Patch* patch,
						    CCVariable<T>& grad_lim,
						    StaticArray<CCVariable<T> >& q_OAFS,
			                         DataWarehouse* new_dw)
  
{
 CCVariable<T> q_grad_x,q_grad_y,q_grad_z;

 new_dw->allocateTemporary(q_grad_x,   patch);
 new_dw->allocateTemporary(q_grad_y,   patch);
 new_dw->allocateTemporary(q_grad_z,   patch);
 
 gradQ(q_CC, patch, q_grad_x, q_grad_y, q_grad_z);

// Approximate q_OAFS[face][ghost cell] = q_CC[ghost cell]

/*      for(CellIterator iter = patch->getFaceCellIterator(Patch::xplus); !iter.done(); iter++) { 
       IntVector curcell = *iter;
       for (int face = TOP; face <= BACK; face++) {
	   q_OAFS[face][curcell] = q_CC[curcell];
	}
      } 
      for(CellIterator iter = patch->getFaceCellIterator(Patch::xminus); !iter.done(); iter++) { 
       IntVector curcell = *iter;
       for (int face = TOP; face <= BACK; face++) {
	   q_OAFS[face][curcell] = q_CC[curcell];
	}
      } 
      for(CellIterator iter = patch->getFaceCellIterator(Patch::yplus); !iter.done(); iter++) { 
       IntVector curcell = *iter;
       for (int face = TOP; face <= BACK; face++) {
	   q_OAFS[face][curcell] = q_CC[curcell];
	}
      } 
      for(CellIterator iter = patch->getFaceCellIterator(Patch::yminus); !iter.done(); iter++) { 
       IntVector curcell = *iter;
       for (int face = TOP; face <= BACK; face++) {
	   q_OAFS[face][curcell] = q_CC[curcell];
	}
      } 
      for(CellIterator iter = patch->getFaceCellIterator(Patch::zplus); !iter.done(); iter++) { 
       IntVector curcell = *iter;
       for (int face = TOP; face <= BACK; face++) {
	   q_OAFS[face][curcell] = q_CC[curcell];
	}
      } 
      for(CellIterator iter = patch->getFaceCellIterator(Patch::zminus); !iter.done(); iter++) { 
       IntVector curcell = *iter;
       for (int face = TOP; face <= BACK; face++) {
	   q_OAFS[face][curcell] = q_CC[curcell];
	}
      } */
      
  const IntVector gc(1,1,1);

  for(CellIterator iter = patch->getCellIterator(gc); !iter.done(); iter++){
    IntVector curcell = *iter;
         for (int face = TOP; face <= BACK; face ++){
	    q_OAFS[face][curcell] = q_CC[curcell];
	  }
    }

 for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    IntVector curcell = *iter;

    q_grad_x[curcell] = grad_lim[curcell] * q_grad_x[curcell];
    q_grad_y[curcell] = grad_lim[curcell] * q_grad_y[curcell];
    q_grad_z[curcell] = grad_lim[curcell] * q_grad_z[curcell];

    //__________________________________
    //  OUTAVERAGEFLUX: SLAB

          
    q_OAFS[BACK][curcell] =   q_grad_x[curcell] * r_out_x[curcell].d_fflux[BACK] + 
                              q_grad_y[curcell] * r_out_y[curcell].d_fflux[BACK] + 
				  q_grad_z[curcell] * r_out_z[curcell].d_fflux[BACK] + q_CC[curcell];
  				  
    q_OAFS[FRONT][curcell] =  q_grad_x[curcell] * r_out_x[curcell].d_fflux[FRONT] + 
                              q_grad_y[curcell] * r_out_y[curcell].d_fflux[FRONT] + 
				  q_grad_z[curcell] * r_out_z[curcell].d_fflux[FRONT] + q_CC[curcell];

    q_OAFS[BOTTOM][curcell] = q_grad_x[curcell] * r_out_x[curcell].d_fflux[BOTTOM] + 
                              q_grad_y[curcell] * r_out_y[curcell].d_fflux[BOTTOM] + 
				  q_grad_z[curcell] * r_out_z[curcell].d_fflux[BOTTOM] + q_CC[curcell];
  				  
    q_OAFS[TOP][curcell] =    q_grad_x[curcell] * r_out_x[curcell].d_fflux[TOP] + 
                              q_grad_y[curcell] * r_out_y[curcell].d_fflux[TOP] + 
				  q_grad_z[curcell] * r_out_z[curcell].d_fflux[TOP] + q_CC[curcell];
  				  
    q_OAFS[LEFT][curcell] =   q_grad_x[curcell] * r_out_x[curcell].d_fflux[LEFT] + 
                              q_grad_y[curcell] * r_out_y[curcell].d_fflux[LEFT] + 
				  q_grad_z[curcell] * r_out_z[curcell].d_fflux[LEFT] + q_CC[curcell];
  				  
    q_OAFS[RIGHT][curcell] =  q_grad_x[curcell] * r_out_x[curcell].d_fflux[RIGHT] + 
                              q_grad_y[curcell] * r_out_y[curcell].d_fflux[RIGHT] + 
				  q_grad_z[curcell] * r_out_z[curcell].d_fflux[RIGHT] + q_CC[curcell];  				 
  }
   
}

template <class T> void SecondOrderAdvector::q_CCMaxMin(const CCVariable<T>& q_CC,
                                              const Patch* patch,
						    CCVariable<T>& q_CC_max, 
						    CCVariable<T>& q_CC_min)
{  
     T q_CC_max_tmp, q_CC_min_tmp;
     IntVector adjcell;
     for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
       IntVector curcell = *iter;
	int i = curcell.x();
       int j = curcell.y();
       int k = curcell.z();
       
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
	
	q_CC_max[curcell] = q_CC_max_tmp;
	q_CC_min[curcell] = q_CC_min_tmp;

      }       
      
}						    


template <class T> void SecondOrderAdvector::q_vertexMaxMin(const CCVariable<T>& q_CC,
                                              const Patch* patch,
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
  
     CCVariable<T> q_grad_x,q_grad_y,q_grad_z;
 
     new_dw->allocateTemporary(q_grad_x,   patch);
     new_dw->allocateTemporary(q_grad_y,   patch);
     new_dw->allocateTemporary(q_grad_z,   patch);
     
     gradQ(q_CC, patch, q_grad_x, q_grad_y, q_grad_z);

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector curcell = *iter;
         
      q_vrtx1 = q_grad_x[curcell]*(delX_2)+q_grad_y[curcell]*(delY_2)+q_grad_z[curcell]*(delZ_2)+q_CC[curcell];
      q_vrtx2 = q_grad_x[curcell]*(delX_2)+q_grad_y[curcell]*(delY_2)+q_grad_z[curcell]*(-delZ_2)+q_CC[curcell];	
      q_vrtx3 = q_grad_x[curcell]*(delX_2)+q_grad_y[curcell]*(-delY_2)+q_grad_z[curcell]*(delZ_2)+q_CC[curcell];      
      q_vrtx4 = q_grad_x[curcell]*(delX_2)+q_grad_y[curcell]*(-delY_2)+q_grad_z[curcell]*(-delZ_2)+q_CC[curcell];	
      q_vrtx5 = q_grad_x[curcell]*(-delX_2)+q_grad_y[curcell]*(delY_2)+q_grad_z[curcell]*(delZ_2)+q_CC[curcell];	
      q_vrtx6 = q_grad_x[curcell]*(-delX_2)+q_grad_y[curcell]*(delY_2)+q_grad_z[curcell]*(-delZ_2)+q_CC[curcell];      
      q_vrtx7 = q_grad_x[curcell]*(-delX_2)+q_grad_y[curcell]*(-delY_2)+q_grad_z[curcell]*(delZ_2)+q_CC[curcell];
      q_vrtx8 = q_grad_x[curcell]*(-delX_2)+q_grad_y[curcell]*(-delY_2)+q_grad_z[curcell]*(-delZ_2)+q_CC[curcell];
      
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
                  
      q_vrtx_max[curcell] = q_vrtx_tmp_max;
      q_vrtx_min[curcell] = q_vrtx_tmp_min; 

    }    
}
template <class T> void SecondOrderAdvector::gradQ(const CCVariable<T>& q_CC,
                                              const Patch* patch,
						    CCVariable<T>& q_grad_x,
						    CCVariable<T>& q_grad_y,
						    CCVariable<T>& q_grad_z)
{  
    Vector dx = patch->dCell();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    IntVector adjcell1, adjcell2;
    
   for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    IntVector curcell = *iter;
       int i = curcell.x();
       int j = curcell.y();
       int k = curcell.z();
       
       adjcell1 = IntVector(i+1, j, k);
       adjcell2 = IntVector(i-1, j, k);
	q_grad_x[curcell] = (q_CC[adjcell1] - q_CC[adjcell2])/(2.0*delX);
	
	adjcell1 = IntVector(i, j+1, k);
       adjcell2 = IntVector(i, j-1, k);
	q_grad_y[curcell] = (q_CC[adjcell1] - q_CC[adjcell2])/(2.0*delY);

       adjcell1 = IntVector(i, j, k+1);
       adjcell2 = IntVector(i, j, k-1);
	q_grad_z[curcell] = (q_CC[adjcell1] - q_CC[adjcell2])/(2.0*delZ);

   }
   
}


namespace Uintah {

  static MPI_Datatype makeMPI_fflux()
  {
    ASSERTEQ(sizeof(SecondOrderAdvector::fflux), sizeof(double)*6);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 6, 6, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(SecondOrderAdvector::fflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                              "SecondOrderAdvector::fflux", true, 
                              &makeMPI_fflux);
    }
    return td;
  }
  
}

namespace SCIRun {

void swapbytes( Uintah::SecondOrderAdvector::fflux& f) {
  double *p = f.d_fflux;
  SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
  SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
}

} // namespace SCIRun
