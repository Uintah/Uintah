#ifndef UINTAH_FIRST_ORDER_ADVECTOR_H
#define UINTAH_FIRST_ORDER_ADVECTOR_H

#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

namespace Uintah {


  class FirstOrderAdvector : public Advector {

  public:
    FirstOrderAdvector();
    FirstOrderAdvector(DataWarehouse* new_dw, 
                       const Patch* patch,
                       const bool isNewGrid);
    virtual ~FirstOrderAdvector();

    virtual FirstOrderAdvector* clone(DataWarehouse* new_dw, 
                                      const Patch* patch,
                                      const bool isNewGrid);


    virtual void inFluxOutFluxVolume(const SFCXVariable<double>& uvel_CC,
                                     const SFCYVariable<double>& vvel_CC,
                                     const SFCZVariable<double>& wvel_CC,
                                     const double& delT, 
                                     const Patch* patch,
                                     const int&  indx,
                                     const bool& bulletProof_test,
                                     DataWarehouse* new_dw);

    virtual void  advectQ(const CCVariable<double>& q_CC,
                          CCVariable<double>& q_advected,
                          advectVarBasket* varBasket,
                          constSFCXVariable<double>& uvel_FC,
                          constSFCYVariable<double>& vvel_FC,
                          constSFCZVariable<double>& wvel_FC,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC);
 
    virtual void advectQ(const CCVariable<double>& q_CC,
                         const CCVariable<double>& mass,
                         CCVariable<double>& q_advected,
                         constSFCXVariable<double>& uvel_FC,
                         constSFCYVariable<double>& vvel_FC,
                         constSFCZVariable<double>& wvel_FC,
                         advectVarBasket* vb);
 
    virtual void advectQ(const CCVariable<Vector>& q_CC,
                         const CCVariable<double>& mass,
                         CCVariable<Vector>& q_advected,
                         constSFCXVariable<double>& uvel_FC,
                         constSFCYVariable<double>& vvel_FC,
                         constSFCZVariable<double>& wvel_FC,
                         advectVarBasket* vb); 
                         
    virtual void advectMass(const CCVariable<double>& mass,
                            CCVariable<double>& q_advected,
                            constSFCXVariable<double>& uvel_FC,
                            constSFCYVariable<double>& vvel_FC,
                            constSFCZVariable<double>& wvel_FC,
                            advectVarBasket* vb);
    
  private:
    CCVariable<fflux> d_OFS;
    

  private:
    void outFluxVolume( double ofs[6],
                        const SFCXVariable<double>& uvel_FC,
                        const SFCYVariable<double>& vvel_FC,
                        const SFCZVariable<double>& wvel_FC,
                        const IntVector c,
                        const double& delT, 
                        const Vector dx);
    
 
    template<class T, typename F> 
      void advectSlabs(const CCVariable<T>& q_CC,
                       CCVariable<T>& q_advected,
                       advectVarBasket* vb,
                       constSFCXVariable<double>& uvel_FC,
                       constSFCYVariable<double>& vvel_FC,
                       constSFCZVariable<double>& wvel_FC,
                       SFCXVariable<double>& q_XFC,
                       SFCYVariable<double>& q_YFC,
                       SFCZVariable<double>& q_ZFC,
                       F function); 
 
    template<class T, class V, typename F>
      void advect_operator(CellIterator iter,
                	      const IntVector dir,
                           advectVarBasket* varBasket,
                           T& vel_FC,
                	      const CCVariable<V>& q_CC,
                           CCVariable<V>& q_advected,
                           F save_q_FC);
                                                       
    template<class T>
      void q_FC_operator(CellIterator iter, 
                         IntVector adj_offset,
                         const int face,
                         const CCVariable<double>& q_CC,
                         T& q_FC);
                        
      void q_FC_PlusFaces(const CCVariable<double>& q_CC,
                          const Patch* patch,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC);
                                  
    template<class T, class V>
      void q_FC_flux_operator(CellIterator iter, 
                              IntVector adj_offset,
                              const int face,
                              const CCVariable<V>& q_CC,
                              T& q_FC);
                                               
    template<class T>
      void q_FC_fluxes(const CCVariable<T>& q_CC, 
                       const string& desc,
                       advectVarBasket* vb);
  };
}

#endif


