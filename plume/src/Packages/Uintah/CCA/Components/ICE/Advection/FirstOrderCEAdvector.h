#ifndef UINTAH_FIRST_ORDER_SLOW_ADVECTOR_H
#define UINTAH_FIRST_ORDER_SLOW_ADVECTOR_H

#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderAdvector.h>
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
                                 const bool& bulletProof_test,
                                 DataWarehouse* new_dw);
                                 
    virtual void  advectQ(const CCVariable<double>& q_CC,
                          const Patch* patch,
                          CCVariable<double>& q_advected,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC,
                          DataWarehouse* /*new_dw*/);

    virtual void advectQ(const CCVariable<double>& q_CC,
                         const CCVariable<double>& mass,
                         CCVariable<double>& q_advected,
                         advectVarBasket* vb);
 
    virtual void advectQ(const CCVariable<Vector>& q_CC,
                         const CCVariable<double>& mass,
                         CCVariable<Vector>& q_advected,
                         advectVarBasket* vb); 
                         
    virtual void advectMass(const CCVariable<double>& mass,
                            CCVariable<double>& q_advected,
                            advectVarBasket* vb);

  private:
    template <class T, typename F> 
      void advectCE(const CCVariable<T>& q_CC,
                    const Patch* patch,                                      
                    CCVariable<T>& q_advected,                     
                    SFCXVariable<double>& q_XFC,                   
                    SFCYVariable<double>& q_YFC,                   
                    SFCZVariable<double>& q_ZFC,                   
                    F save_q_FC);

    template<class T, class V>
      void  q_FC_flux_operator(CellIterator iter, 
                               IntVector adj_offset,
                               const int face,
                               const CCVariable<V>& q_CC,
                               T& q_faceFlux);
                        
    void q_FC_PlusFaces(const CCVariable<double>& q_CC,
                        const Patch* patch,
                        SFCXVariable<double>& q_XFC,
                        SFCYVariable<double>& q_YFC,
                        SFCZVariable<double>& q_ZFC);
                                            
    template<class T>
      void q_FC_operator(CellIterator iter, 
                         IntVector adj_offset,
                         const int face,
                         const CCVariable<double >& q_CC,
                         T& q_FC);
                      
    template<class T>
      void  q_FC_fluxes(const CCVariable<T>& q_CC, 
                        const string& desc,
                        advectVarBasket* vb);                       

  private:
    CCVariable<fflux> d_OFS;
    CCVariable<eflux> d_OFE;
    CCVariable<cflux> d_OFC;   
  };
}

#endif
