#ifndef UINTAH_FIRST_ORDER_ADVECTOR_H
#define UINTAH_FIRST_ORDER_ADVECTOR_H

#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

namespace Uintah {


  class FirstOrderAdvector : public Advector {

  public:
    FirstOrderAdvector();
    FirstOrderAdvector(DataWarehouse* new_dw, const Patch* patch);
    virtual ~FirstOrderAdvector();

    virtual FirstOrderAdvector* clone(DataWarehouse* new_dw, 
                                  const Patch* patch);


    virtual void inFluxOutFluxVolume(const SFCXVariable<double>& uvel_CC,
                                 const SFCYVariable<double>& vvel_CC,
                                 const SFCZVariable<double>& wvel_CC,
                                 const double& delT, 
                                 const Patch* patch,
                                 const int&  indx,
                                 const bool& bulletProof_test);

    virtual void  advectQ(const CCVariable<double>& q_CC,
                             const Patch* patch,
                             CCVariable<double>& q_advected,
                             SFCXVariable<double>& q_XFC,
                             SFCYVariable<double>& q_YFC,
                             SFCZVariable<double>& q_ZFC,
				 DataWarehouse* /*new_dw*/);

    virtual void advectQ(const CCVariable<double>& q_CC,
                      const Patch* patch,
                      CCVariable<double>& q_advected,
			 DataWarehouse* new_dw);
    
    virtual void advectQ(const CCVariable<Vector>& q_CC,
                      const Patch* patch,
                      CCVariable<Vector>& q_advected,
			 DataWarehouse* new_dw);
    
  private:
    CCVariable<fflux> d_OFS;
    
    friend class FirstOrderCEAdvector;

  private:
 
    template<class T, typename F> 
      void advectSlabs(const CCVariable<T>& q_CC,
                  const Patch* patch,
                  CCVariable<T>& q_advected,
                  SFCXVariable<double>& q_XFC,
                  SFCYVariable<double>& q_YFC,
                  SFCZVariable<double>& q_ZFC,
                  F function);                  
  };
}

#endif


