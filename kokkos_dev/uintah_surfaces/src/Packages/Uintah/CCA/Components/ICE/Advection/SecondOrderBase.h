#ifndef UINTAH_SECOND_ORDER_BASE_H
#define UINTAH_SECOND_ORDER_BASE_H

#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InternalError.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>

#define d_SMALL_NUM 1e-100
namespace Uintah {

class SecondOrderBase {

public:
  SecondOrderBase();
  virtual ~SecondOrderBase();
  
  template<class T> 
  void limitedGradient(const CCVariable<T>& q_CC,
                       const Patch* patch,
                       const CCVariable<vertex<T> >& q_vertex, 
			  CCVariable<T>& q_grad_x,
			  CCVariable<T>& q_grad_y,
			  CCVariable<T>& q_grad_z);
                       
  template<class T> 
  void Q_vertex( const bool useCompatibleFluxes,
                 const CCVariable<T>& q_CC,
                 CCVariable<vertex<T> >& q_vertex,
                 const Patch* patch,
	          const CCVariable<T>& q_grad_x,
	          const CCVariable<T>& q_grad_y,
	          const CCVariable<T>& q_grad_z);
  
                                 
  template<class T> 
    inline void q_CCMaxMin(const CCVariable<T>& q_CC,
                           const IntVector& c,
			      T& q_CC_max, 
			      T& q_CC_min);
                           
  template<class T>                                    
  void gradQ( const CCVariable<T>& q_CC,
              const Patch* patch,
	       CCVariable<T>& q_grad_x,
	       CCVariable<T>& q_grad_y,
	       CCVariable<T>& q_grad_z);
 
  void mass_massVertex_ratio( const CCVariable<double>& mass_CC,
                            const Patch* patch,
			       const CCVariable<double>& q_grad_x,
			       const CCVariable<double>& q_grad_y,
			       const CCVariable<double>& q_grad_z);

  template <class T>
  bool 
  flux_to_primitive( const bool useCompatibleFluxes,
                     const bool is_Q_mass_specific,
                     const Patch* patch,
                     const CCVariable<T>& A_CC,
                     const CCVariable<double>& mass,
                     CCVariable<T>& q_CC);
                       
  CCVariable<fflux>  r_out_x, r_out_y, r_out_z;
  CCVariable<vertex<double> > d_mass_massVertex;
};

/*_____________________________________________________________________
 Function~ gradQ
 Purpose   Find the x, y, z gradients of q_CC, centered differencing 
_____________________________________________________________________*/
template <class T>
inline void
SecondOrderBase::gradQ( const CCVariable<T>& q_CC,
                        const Patch* patch,
			   CCVariable<T>& q_grad_x,
			   CCVariable<T>& q_grad_y,
			   CCVariable<T>& q_grad_z)
{
  Vector dx = patch->dCell();
  Vector inv_2del;
  inv_2del.x(1.0/(2.0 * dx.x()) );
  inv_2del.y(1.0/(2.0 * dx.y()) );
  inv_2del.z(1.0/(2.0 * dx.z()) );

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
    
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    const IntVector& c = *iter;
    
    IntVector r = c + IntVector( 1, 0, 0);
    IntVector l = c + IntVector(-1, 0, 0);
    IntVector t = c + IntVector( 0, 1, 0);
    IntVector b = c + IntVector( 0,-1, 0);
    IntVector f = c + IntVector( 0, 0, 1);    
    IntVector bk= c + IntVector( 0, 0,-1);

    q_grad_x[c] = (q_CC[r] - q_CC[l]) * inv_2del.x();
    q_grad_y[c] = (q_CC[t] - q_CC[b]) * inv_2del.y();
    q_grad_z[c] = (q_CC[f] - q_CC[bk])* inv_2del.z();
  }
}
/*_____________________________________________________________________
Function~  q_CCMaxMin
Purpose~   This calculates the max and min values of q_CC from the surrounding
           cells
_____________________________________________________________________*/
template <class T>
inline void
SecondOrderBase::q_CCMaxMin(const CCVariable<T>& q_CC,
                            const IntVector& c,
			       T& q_CC_max, 
			       T& q_CC_min)
{  
  T max_tmp, min_tmp;     
  IntVector r = c + IntVector( 1, 0, 0);
  IntVector l = c + IntVector(-1, 0, 0);
  IntVector t = c + IntVector( 0, 1, 0);
  IntVector b = c + IntVector( 0,-1, 0);
  IntVector f = c + IntVector( 0, 0, 1);    
  IntVector bk= c + IntVector( 0, 0,-1); 

  max_tmp = SCIRun::Max(q_CC[r], q_CC[l]);
  min_tmp = SCIRun::Min(q_CC[r], q_CC[l]);

  max_tmp = SCIRun::Max(max_tmp, q_CC[t]);
  min_tmp = SCIRun::Min(min_tmp, q_CC[t]);

  max_tmp = SCIRun::Max(max_tmp, q_CC[b]);
  min_tmp = SCIRun::Min(min_tmp, q_CC[b]);

  max_tmp = SCIRun::Max(max_tmp, q_CC[f]);
  min_tmp = SCIRun::Min(min_tmp, q_CC[f]);

  max_tmp = SCIRun::Max(max_tmp, q_CC[bk]);
  min_tmp = SCIRun::Min(min_tmp, q_CC[bk]);

  q_CC_max = max_tmp;
  q_CC_min = min_tmp;
}                                                   

/*_____________________________________________________________________
 Function~ q_vertex
 Purpose   compute q_vertex with gradient limited gradients 
           Equation 3.3.5 of reference.  Note that there 2 different
           equations depending on if q_CC is mass specific and we're using
           compatible flux formulation
_____________________________________________________________________*/
template<class T> 
void
SecondOrderBase::Q_vertex( const bool usingCompatibleFluxes,
                           const CCVariable<T>& q_CC,
                           CCVariable<vertex<T> >& q_vertex,
                           const Patch* patch,
			      const CCVariable<T>& q_grad_x,
			      const CCVariable<T>& q_grad_y,
			      const CCVariable<T>& q_grad_z)
{
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
  Vector dx = patch->dCell();
  double dx_2 = dx.x()/2.0;  // distance from cell center to vertex
  double dy_2 = dx.y()/2.0;
  double dz_2 = dx.z()/2.0;
  
  if(!usingCompatibleFluxes) {        // non-compatible advection
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      const IntVector& c = *iter;

      T xterm = q_grad_x[c] * dx_2;
      T yterm = q_grad_y[c] * dy_2;
      T zterm = q_grad_z[c] * dz_2;
      T q = q_CC[c];
      
      vertex<T>& q_vrtx = q_vertex[c];
      q_vrtx.d_vrtx[0] = q + (-xterm - yterm - zterm);
      q_vrtx.d_vrtx[1] = q + (-xterm - yterm + zterm);
      q_vrtx.d_vrtx[2] = q + (-xterm + yterm - zterm);     
      q_vrtx.d_vrtx[3] = q + (-xterm + yterm + zterm);
      q_vrtx.d_vrtx[4] = q + (xterm - yterm - zterm);
      q_vrtx.d_vrtx[5] = q + (xterm - yterm + zterm);    
      q_vrtx.d_vrtx[6] = q + (xterm + yterm - zterm);
      q_vrtx.d_vrtx[7] = q + (xterm + yterm + zterm);
    }
  }
  if(usingCompatibleFluxes) {         // compatible advection
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      const IntVector& c = *iter;
      T xterm = q_grad_x[c] * dx_2;
      T yterm = q_grad_y[c] * dy_2;
      T zterm = q_grad_z[c] * dz_2;
      
      T q = q_CC[c];
      vertex<T>& q_vrtx = q_vertex[c];
      const vertex<double>& mass_ratio = d_mass_massVertex[c];
      
      q_vrtx.d_vrtx[0] = q + (-xterm - yterm - zterm) * mass_ratio.d_vrtx[0];
      q_vrtx.d_vrtx[1] = q + (-xterm - yterm + zterm) * mass_ratio.d_vrtx[1];
      q_vrtx.d_vrtx[2] = q + (-xterm + yterm - zterm) * mass_ratio.d_vrtx[2];     
      q_vrtx.d_vrtx[3] = q + (-xterm + yterm + zterm) * mass_ratio.d_vrtx[3];
      q_vrtx.d_vrtx[4] = q + (xterm - yterm - zterm)  * mass_ratio.d_vrtx[4];
      q_vrtx.d_vrtx[5] = q + (xterm - yterm + zterm)  * mass_ratio.d_vrtx[5];    
      q_vrtx.d_vrtx[6] = q + (xterm + yterm - zterm)  * mass_ratio.d_vrtx[6];
      q_vrtx.d_vrtx[7] = q + (xterm + yterm + zterm)  * mass_ratio.d_vrtx[7];
    }
  }
}
/*_____________________________________________________________________
 Function~ convertFlux_to_primitive.
 Purpose: - IF using compatible fluxes convert flux into primitive variable  
          - For compatible fluxes, q_CC MUST be a mass-specific quantity
            Test for it.
_____________________________________________________________________*/
template <class T>
bool 
SecondOrderBase::flux_to_primitive( const bool useCompatibleFluxes,
                                    const bool is_Q_mass_specific,
                                    const Patch* patch,
                                    const CCVariable<T>& A_CC,
                                    const CCVariable<double>& mass,
                                    CCVariable<T>& q_CC)
{


  // bulletproofing
  if(useCompatibleFluxes && !is_Q_mass_specific){
    throw InternalError("ICE:SecondOrderAdvection:\n"
    " For compatible fluxes, Q_CC must be a mass-specific quantity \n");
  }
                // compatible fluxes.
  if(useCompatibleFluxes && is_Q_mass_specific) {
    CellIterator iter = patch->getExtraCellIterator();
    CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,2);
  
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      const IntVector& c = *iter;
      q_CC[c] = A_CC[c]/mass[c];
    }
    return true;
  } else {      // non-compatible fluxes
    q_CC.copyData(A_CC);
    return false;
  }
}
/*_____________________________________________________________________
 Function~ limitedGradient
_____________________________________________________________________*/
template <class T>
void SecondOrderBase::limitedGradient(const CCVariable<T>& q_CC,
                                      const Patch* patch,
                                      const CCVariable<vertex<T> >& q_vertex,
				          CCVariable<T>& q_grad_x,
				          CCVariable<T>& q_grad_y,
				          CCVariable<T>& q_grad_z)
{
  T  frac,temp, zero(0.);
  T  gradLim_max, gradLim_min;
  T unit = T(1.0);
  T SN = T(d_SMALL_NUM);
    
  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
  
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    const IntVector& c = *iter;

    T q_vrtx_max, q_vrtx_min, q_CC_max, q_CC_min;
    T Q_CC = q_CC[c];
    
    //__________________________________
    // q_vertex & q_CC (max/min)
    q_CCMaxMin( q_CC, c, q_CC_max, q_CC_min);
    
    q_vrtx_max = q_vertex[c].d_vrtx[0];
    q_vrtx_min = q_vertex[c].d_vrtx[0];
    
    for (int i=1;i<8;i++){
      q_vrtx_max = SCIRun::Max(q_vrtx_max,q_vertex[c].d_vrtx[i]);
      q_vrtx_min = SCIRun::Min(q_vrtx_min,q_vertex[c].d_vrtx[i]);
    }
    
    //__________________________________
    // gradient limiter
    frac = (q_CC_max - Q_CC + SN)/(q_vrtx_max - Q_CC + SN);
    gradLim_max = SCIRun::Max(zero, frac);

    frac = (q_CC_min - Q_CC + SN)/(q_vrtx_min - Q_CC + SN);
    gradLim_min = SCIRun::Max(zero, frac);

    temp = SCIRun::Min(unit, gradLim_max);
    temp = SCIRun::Min(temp, gradLim_min);
    T gradLim = temp;
           
    q_grad_x[c] = q_grad_x[c] * gradLim;
    q_grad_y[c] = q_grad_y[c] * gradLim;
    q_grad_z[c] = q_grad_z[c] * gradLim; 
  }
}

} // end namespace Uintah

#endif
