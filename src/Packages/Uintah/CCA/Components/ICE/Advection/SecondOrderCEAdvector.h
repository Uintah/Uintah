#ifndef UINTAH_SECOND_ORDER_SLOW_ADVECTOR_H
#define UINTAH_SECOND_ORDER_SLOW_ADVECTOR_H

#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
//#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderAdvector.h>
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
				     const int& indx);

    virtual void advectQ(const CCVariable<double>& q_CC,
                      const Patch* patch,
                      CCVariable<double>& q_advected,
			 DataWarehouse* new_dw);
    
    virtual void advectQ(const CCVariable<Vector>& q_CC,
                      const Patch* patch,
                      CCVariable<Vector>& q_advected,
			 DataWarehouse* new_dw);
			 


    struct eflux { double d_eflux[12]; };         //edge flux
    struct cflux { double d_cflux[8]; };          //corner flux

    enum EDGE {TOP_R = 0, TOP_FR, TOP_L, TOP_BK, BOT_R, BOT_FR, BOT_L, BOT_BK,
              RIGHT_BK, RIGHT_FR, LEFT_BK, LEFT_FR };
    enum CORNER {TOP_R_BK = 0, TOP_R_FR, TOP_L_BK, TOP_L_FR, BOT_R_BK, 
               BOT_R_FR, BOT_L_BK, BOT_L_FR};

  private:
    CCVariable<fflux> d_OFS, r_out_x, r_out_y, r_out_z;
    CCVariable<eflux> d_OFE, r_out_x_EF, r_out_y_EF, r_out_z_EF;
    CCVariable<cflux> d_OFC;
    const VarLabel* OFS_CCLabel;
    const VarLabel* OFE_CCLabel;
    const VarLabel* OFC_CCLabel;
    
    friend const TypeDescription* fun_getTypeDescription(fflux*);
    friend const TypeDescription* fun_getTypeDescription(eflux*);
    friend const TypeDescription* fun_getTypeDescription(cflux*);
  
  private:
			 
    template<class T> void qAverageFlux(const CCVariable<T>& q_CC,
                                        const Patch* patch,
				            CCVariable<T>& grad_lim,
                                        const CCVariable<T>& q_grad_x,
                                        const CCVariable<T>& q_grad_y,
                                        const CCVariable<T>& q_grad_z,
				            StaticArray<CCVariable<T> >& q_OAFS,
				            StaticArray<CCVariable<T> >& q_OAFE,
				            StaticArray<CCVariable<T> >& q_OAFC,
			                   DataWarehouse* new_dw);
    
    template<class T> void advect(StaticArray<CCVariable<T> >& q_OAFS,
                                  StaticArray<CCVariable<T> >& q_OAFE,
				      StaticArray<CCVariable<T> >& q_OAFC,
                                  const Patch* patch,
                                  CCVariable<T>& q_advected);
  };
}

// Added for compatibility with core types
#include <Core/Datatypes/TypeName.h>
#include <string>
namespace SCIRun {
void swapbytes( Uintah::SecondOrderCEAdvector::fflux& ); 
void swapbytes( Uintah::SecondOrderCEAdvector::eflux& );
void swapbytes( Uintah::SecondOrderCEAdvector::cflux& );
} // namespace SCIRun

#endif
