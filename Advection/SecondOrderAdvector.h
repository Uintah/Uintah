#ifndef UINTAH_SECOND_ORDER_ADVECTOR_H
#define UINTAH_SECOND_ORDER_ADVECTOR_H

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderBase.h>

#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include <Core/Containers/StaticArray.h>

namespace Uintah {

  class SecondOrderAdvector : public Advector, public SecondOrderBase {

  public:
    SecondOrderAdvector();
    SecondOrderAdvector(DataWarehouse* new_dw, const Patch* patch);
    virtual ~SecondOrderAdvector();

    virtual SecondOrderAdvector* clone(DataWarehouse* new_dw, 
                                  const Patch* patch);


    virtual void inFluxOutFluxVolume(const SFCXVariable<double>& uvel_CC,
                                 const SFCYVariable<double>& vvel_CC,
                                 const SFCZVariable<double>& wvel_CC,
                                 const double& delT, 
                                 const Patch* patch,
				     const int& indx,
                                 const bool& bulletProof_test);
			 
    virtual void advectQ(const CCVariable<double>& q_CC,
                      const Patch* patch,
                      CCVariable<double>& q_advected,
			 DataWarehouse* new_dw);
    
    virtual void advectQ(const CCVariable<Vector>& q_CC,
                      const Patch* patch,
                      CCVariable<Vector>& q_advected,
			 DataWarehouse* new_dw);  
                                         
  private:
    CCVariable<fflux> d_OFS, r_out_x, r_out_y, r_out_z;
    const VarLabel* OFS_CCLabel;
    friend const TypeDescription* fun_getTypeDescription(fflux*); 

  private:                                 
    template<class T> 
    void qAverageFlux(const CCVariable<T>& q_CC,
		      const Patch* patch,
		      CCVariable<T>& grad_lim,
		      const CCVariable<T>& q_grad_x,
		      const CCVariable<T>& q_grad_y,
		      const CCVariable<T>& q_grad_z,
		      StaticArray<CCVariable<T> >& q_OAFS);
    
    template<class T> 
    void advect(StaticArray<CCVariable<T> >& q_OAFS,
		const Patch* patch,
		CCVariable<T>& q_advected);
	 
  };  
} // end namespace Uintah

#endif


