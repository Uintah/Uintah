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
                                     const bool& bulletProof_test,
                                     DataWarehouse* new_dw);
			 
    virtual void  advectQ(const CCVariable<double>& q_CC,
                          const Patch* patch,
                          CCVariable<double>& q_advected,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC,
			     DataWarehouse* /*new_dw*/);

    virtual void advectQ(const bool useCompatibleFluxes,
                         const bool is_Q_massSpecific,
                         const CCVariable<double>& q_CC,
                         const CCVariable<double>& mass,
                         const Patch* patch,
                         CCVariable<double>& q_advected,
                         DataWarehouse* new_dw);
    
    virtual void advectQ(const bool useCompatibleFluxes,
                         const bool is_Q_massSpecific,
                         const CCVariable<Vector>& q_CC,
                         const CCVariable<double>& mass,
                         const Patch* patch,
                         CCVariable<Vector>& q_advected,
                         DataWarehouse* new_dw); 
                         
    virtual void advectMass(const CCVariable<double>& mass,
                            const Patch* patch,
                            CCVariable<double>& q_advected,
			       DataWarehouse* new_dw);

  private:
    CCVariable<fflux> d_OFS;  // outflux slabs
    CCVariable<facedata<double> > d_mass_slabs;  // outflux slabs

  private:                                 
    template<class T> 
      void qAverageFlux(const bool useCompatibleFluxes,
                        const CCVariable<T>& q_CC,
                        const CCVariable<double>& mass_CC,
                        const Patch* patch,
                        CCVariable<facedata<T> >& q_OAFS,
                        const CCVariable<T>& grad_x,
                        const CCVariable<T>& grad_y,
                        const CCVariable<T>& grad_z);
                                         
    template<class T, typename F> 
      void advectSlabs(CCVariable<facedata<T> >& q_OAFS,
		         const Patch* patch,
                       const CCVariable<T>& q_CC,
		         CCVariable<T>& q_advected,
                       SFCXVariable<double>& q_XFC,
                       SFCYVariable<double>& q_YFC,
                       SFCZVariable<double>& q_ZFC,
                       F save_q_FC);  // passed in function
	 
  };  
} // end namespace Uintah

#endif


