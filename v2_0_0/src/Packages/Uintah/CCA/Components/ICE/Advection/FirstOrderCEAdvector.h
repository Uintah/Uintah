#ifndef UINTAH_FIRST_ORDER_SLOW_ADVECTOR_H
#define UINTAH_FIRST_ORDER_SLOW_ADVECTOR_H

#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderAdvector.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>

namespace Uintah {

  class FirstOrderCEAdvector :  public Advector {

    typedef FirstOrderAdvector FOA;

  public:
    FirstOrderCEAdvector();

    FirstOrderCEAdvector(DataWarehouse* new_dw, const Patch* patch);
    virtual ~FirstOrderCEAdvector();

    virtual FirstOrderCEAdvector* clone(DataWarehouse* new_dw,
                                     const Patch* patch);


    virtual void inFluxOutFluxVolume(const SFCXVariable<double>& uvel_CC,
                                 const SFCYVariable<double>& vvel_CC,
                                 const SFCZVariable<double>& wvel_CC,
                                 const double& delT, 
                                 const Patch* patch,
                                 const int& indx,
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
    template <class T, typename F> 
      void advectCE(const CCVariable<T>& q_CC,
                    const Patch* patch,                                      
                    CCVariable<T>& q_advected,                     
                    SFCXVariable<double>& q_XFC,                   
                    SFCYVariable<double>& q_YFC,                   
                    SFCZVariable<double>& q_ZFC,                   
                    F save_q_FC);                                  

  private:
    CCVariable<fflux> d_OFS;
    CCVariable<eflux> d_OFE;
    CCVariable<cflux> d_OFC;
    const VarLabel* OFE_CCLabel;
    const VarLabel* OFC_CCLabel;
    const VarLabel* OFS_CCLabel;    
  };
}

#endif
