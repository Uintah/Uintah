#ifndef UINTAH_SECOND_ORDER_BASE_H
#define UINTAH_SECOND_ORDER_BASE_H

#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>

#define d_SMALL_NUM 1e-100
namespace Uintah {

class SecondOrderBase {

public:
  SecondOrderBase();
  virtual ~SecondOrderBase();
  
  template<class T> 
    inline void gradientLimiter(const CCVariable<T>& q_CC,
                                const IntVector& c,
                                Vector dx_2,
				    T& grad_lim, 
				    T& q_grad_x,
				    T& q_grad_y,
				    T& q_grad_z);
                      
  template<class T> 
    inline void q_CCMaxMin(const CCVariable<T>& q_CC,
                           const IntVector& c,
			      T& q_CC_max, 
			      T& q_CC_min);
                                           
  template<class T> 
    inline void q_vertexMaxMin(T& q_CC,
                               Vector dx_2,
				   T& q_grad_x,
				   T& q_grad_y,
				   T& q_grad_z,
				   T& q_vrtx_max, 
				   T& q_vrtx_min);
                                    
  template<class T> 
    inline void gradQ( const CCVariable<T>& q_CC,
                       const IntVector& c,
                       Vector inv_2del,
			  T& q_grad_x,
			  T& q_grad_y,
			  T& q_grad_z);
                                  
  CCVariable<fflux>  r_out_x, r_out_y, r_out_z; 
};

/* ---------------------------------------------------------------------
 Function~ gradientLimiter
_____________________________________________________________________*/
template <class T>
inline void
SecondOrderBase::gradientLimiter(const CCVariable<T>& q_CC,
                                const IntVector& c,
                                Vector dx_2,
				    T& grad_lim, 
				    T& q_grad_x,
				    T& q_grad_y,
				    T& q_grad_z)
{
  T  frac,temp, zero(0.);
  T  grad_lim_max, grad_lim_min;
  T unit = T(1.0);
  T SN = T(d_SMALL_NUM);
  
  T q_vrtx_max, q_vrtx_min, q_CC_max, q_CC_min;

  //__________________________________
  // find q_cc max and min and
  // vertex max and min
  T Q_CC = q_CC[c];
  q_CCMaxMin( q_CC, c, q_CC_max, q_CC_min);
  q_vertexMaxMin(Q_CC, dx_2, q_grad_x, q_grad_y, q_grad_z,
		   q_vrtx_max, q_vrtx_min);

  frac = (q_CC_max - Q_CC + SN)/(q_vrtx_max - Q_CC + SN);
  grad_lim_max = Max(zero, frac);

  frac = (q_CC_min - Q_CC + SN)/(q_vrtx_min - Q_CC + SN);
  grad_lim_min = Max(zero, frac);

  temp = Min(unit, grad_lim_max);
  temp = Min(temp, grad_lim_min);
  grad_lim = temp; 
}
/* ---------------------------------------------------------------------
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

  q_CC_max = max_tmp;
  q_CC_min = min_tmp;
}                                                   

/* ---------------------------------------------------------------------
 Function~ q_vertexMaxMin
 Purpose   Find the max and min vertex value of q at the vertex.  
_____________________________________________________________________*/
template <class T>
inline void
SecondOrderBase::q_vertexMaxMin( T& q_CC,
                                 Vector dx_2,
				     T& q_grad_x,
				     T& q_grad_y,
				     T& q_grad_z,
				     T& q_vrtx_max, 
				     T& q_vrtx_min)
{
  T q_vrtx1, q_vrtx2,q_vrtx3,q_vrtx4,q_vrtx5,q_vrtx6,q_vrtx7,q_vrtx8;
  T q_vrtx_tmp_max, q_vrtx_tmp_min;

  T xterm = q_grad_x * dx_2.x();
  T yterm = q_grad_y * dx_2.y();
  T zterm = q_grad_z * dx_2.z();
  //__________________________________
  // THINK ABOUT PULLING OUT QCC  --TODD
  q_vrtx1 = xterm  + yterm + zterm + q_CC;
  q_vrtx2 = xterm  + yterm - zterm + q_CC;
  q_vrtx3 = xterm  - yterm + zterm + q_CC;     
  q_vrtx4 = xterm  - yterm - zterm + q_CC;
  q_vrtx5 = -xterm + yterm + zterm + q_CC;
  q_vrtx6 = -xterm + yterm - zterm + q_CC;     
  q_vrtx7 = -xterm - yterm + zterm + q_CC;
  q_vrtx8 = -xterm - yterm - zterm + q_CC;

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

  q_vrtx_max = q_vrtx_tmp_max;
  q_vrtx_min = q_vrtx_tmp_min;  
}
/* ---------------------------------------------------------------------
 Function~ gradQ
 Purpose   Find the x, y, z gradients of q_CC. centered
           differencing 
_____________________________________________________________________*/
template <class T>
inline void
SecondOrderBase::gradQ( const CCVariable<T>& q_CC,
                        const IntVector& c,
                        Vector inv_2del,
			   T& q_grad_x,
			   T& q_grad_y,
			   T& q_grad_z)
{    
  IntVector r = c + IntVector( 1, 0, 0);
  IntVector l = c + IntVector(-1, 0, 0);
  IntVector t = c + IntVector( 0, 1, 0);
  IntVector b = c + IntVector( 0,-1, 0);
  IntVector f = c + IntVector( 0, 0, 1);    
  IntVector bk= c + IntVector( 0, 0,-1);

  q_grad_x = (q_CC[r] - q_CC[l]) * inv_2del.x();
  q_grad_y = (q_CC[t] - q_CC[b]) * inv_2del.y();
  q_grad_z = (q_CC[f] - q_CC[bk])* inv_2del.z();
} 
} // end namespace Uintah

#endif
