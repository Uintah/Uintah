#ifndef UINTAH_SECOND_ORDER_BASE_H
#define UINTAH_SECOND_ORDER_BASE_H
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>

namespace Uintah {
  class SecondOrderBase {

  public:
    SecondOrderBase();
                           
    template<class T> void gradientLimiter(const CCVariable<T>& q_CC,
                              const Patch* patch,
                              CCVariable<T>& grad_lim,
                              const CCVariable<T>& q_grad_x,
                              const CCVariable<T>& q_grad_y,
                              const CCVariable<T>& q_grad_z,
				  T unit, T SN,
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
				      CCVariable<T>& q_vrtx_min,
			             DataWarehouse* new_dw);
				  		  
    template<class T> void gradQ(const CCVariable<T>& q_CC,
                                  const Patch* patch,
				      CCVariable<T>& q_grad_x,
				      CCVariable<T>& q_grad_y,
				      CCVariable<T>& q_grad_z);
                                  
    enum FACE {TOP, BOTTOM, RIGHT, LEFT, FRONT, BACK};
    struct fflux { double d_fflux[6]; };    // face flux

    };
}
#endif
