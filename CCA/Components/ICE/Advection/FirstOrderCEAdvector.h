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
    template<class T> void advect(const CCVariable<T>& q_CC,
                              const Patch* patch,
                              CCVariable<T>& q_advected);


  private:
    FirstOrderAdvector d_advector;
    CCVariable<eflux> d_OFE;
    CCVariable<cflux> d_OFC;
    const VarLabel* OFE_CCLabel;
    const VarLabel* OFC_CCLabel;
    
    friend const TypeDescription* fun_getTypeDescription(eflux*);
    friend const TypeDescription* fun_getTypeDescription(cflux*);

    
  };


  
}

// Added for compatibility with core types
#include <Core/Datatypes/TypeName.h>
#include <string>
namespace SCIRun {
void swapbytes( Uintah::FirstOrderCEAdvector::eflux& );
void swapbytes( Uintah::FirstOrderCEAdvector::cflux& );
} // namespace SCIRun

#endif
