#ifndef UINTAH_SECOND_ORDER_SLOW_ADVECTOR_H
#define UINTAH_SECOND_ORDER_SLOW_ADVECTOR_H

#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderBase.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Containers/StaticArray.h>

namespace Uintah {

  class SecondOrderCEAdvector :  public Advector, public SecondOrderBase {

  public:
    SecondOrderCEAdvector();
    SecondOrderCEAdvector(DataWarehouse* new_dw, const Patch* patch);
    virtual ~SecondOrderCEAdvector();

    virtual SecondOrderCEAdvector* clone(DataWarehouse* new_dw,
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
    CCVariable<fflux> d_OFS, r_out_x, r_out_y, r_out_z;
    CCVariable<eflux> d_OFE, r_out_x_EF, r_out_y_EF, r_out_z_EF;
    CCVariable<cflux> d_OFC;
    
  private:
                         
    template<class T> 
      void qAverageFlux(const CCVariable<T>& q_CC,
                        const Patch* patch,
                        CCVariable<T>& grad_lim,
                        const CCVariable<T>& q_grad_x,
                        const CCVariable<T>& q_grad_y,
                        const CCVariable<T>& q_grad_z,
                        StaticArray<CCVariable<T> >& q_OAFS,
                        StaticArray<CCVariable<T> >& q_OAFE,
                        StaticArray<CCVariable<T> >& q_OAFC);

    template <class T>
      void  allocateAndCompute_Q_ave( const CCVariable<T>& q_CC,
                                      const Patch* patch,
                                      DataWarehouse* new_dw,
                                      StaticArray<CCVariable<T> >& q_OAFS,
                                      StaticArray<CCVariable<T> >& q_OAFE,
                                      StaticArray<CCVariable<T> >& q_OAFC ); 
                                 
    template <class T, typename F> 
      void advect(  StaticArray<CCVariable<T> >& q_OAFS,
                    StaticArray<CCVariable<T> >& q_OAFE,
		      StaticArray<CCVariable<T> >& q_OAFC,
                    const Patch* patch,
                    const CCVariable<T>& q_CC,
                    CCVariable<T>& q_advected,
                    SFCXVariable<double>& q_XFC,
                    SFCYVariable<double>& q_YFC,
                    SFCZVariable<double>& q_ZFC,
                    F save_q_FC);   // passed in function
  };
}

#endif
