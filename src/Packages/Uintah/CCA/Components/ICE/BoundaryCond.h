#ifndef Packages_Uintah_CCA_Components_Ice_BoundaryCond_h
#define Packages_Uintah_CCA_Components_Ice_BoundaryCond_h

#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Containers/StaticArray.h>

/*`==========TESTING==========*/
#undef JET_BC    // needed if you want a jet for either LODI or ORG_BCS

#undef LODI_BCS  // note for LODI_BCs you also need ORG_BCS turned on

#undef ORG_BCS    // original setBC 

#define JOHNS_BC   // DEFAULT BOUNDARY CONDITIONS.
/*==========TESTING==========`*/
namespace Uintah {
  class DataWarehouse;
  void setHydrostaticPressureBC(CCVariable<double>& press,
                            Patch::FaceType face, Vector& gravity,
                            const CCVariable<double>& rho,
                            const Vector& dx,
                            IntVector offset = IntVector(0,0,0));
    
  void determineSpacingAndGravity(Patch::FaceType face, Vector& dx,
                              SimulationStateP& sharedState,
                              double& spacing, double& gravity);

  void setBC(CCVariable<double>& variable,const std::string& type, 
            const Patch* p,  SimulationStateP& sharedState,
            const int mat_id);
  
  void setBC(CCVariable<double>& press_CC, const CCVariable<double>& rho,
             const std::string& whichVar, const std::string& type, 
             const Patch* p, SimulationStateP& sharedState,
             const int mat_id, DataWarehouse*);
  
  void setBC(CCVariable<Vector>& variable,const std::string& type,
             const Patch* p, const int mat_id);

/*`==========TESTING==========*/
void setBCPress_LODI(CCVariable<double>& press_CC,
                     StaticArray<CCVariable<double> >& sp_vol_CC,
                     StaticArray<constCCVariable<double> >& Temp_CC,
                     StaticArray<CCVariable<double> >& f_theta,
                     const string& which_Var,
                     const string& kind, 
                     const Patch* patch,
                     SimulationStateP& sharedState, 
                     const int mat_id,
                     DataWarehouse* new_dw);

void setBCDensityLODI(CCVariable<double>& rho_CC,
                StaticArray<CCVariable<Vector> >& di,                      
                const CCVariable<double>& nux,
                const CCVariable<double>& nuy,
                const CCVariable<double>& nuz,
                constCCVariable<double>& rho_tmp,
                const CCVariable<double>& p,
                constCCVariable<Vector>& vel,            
                const double delT,
                const Patch* patch, 
                const int mat_id); 
              
void setBCVelLODI(CCVariable<Vector>& vel_CC,
            StaticArray<CCVariable<Vector> >& di,
            const CCVariable<double>& nux,
            const CCVariable<double>& nuy,
            const CCVariable<double>& nuz,
            constCCVariable<double>& rho_tmp,
            const CCVariable<double>& p,
            constCCVariable<Vector>& vel,
            const double delT,
            const Patch* patch, 
            const int mat_id); 
           
              
 void setBCTempLODI(CCVariable<double>& temp_CC,
              StaticArray<CCVariable<Vector> >& di,
              const CCVariable<double>& e,
              const CCVariable<double>& rho_CC,
              const CCVariable<double>& nux,
              const CCVariable<double>& nuy,
              const CCVariable<double>& nuz,
              constCCVariable<double>& rho_tmp,
              const CCVariable<double>& p,
              constCCVariable<Vector>& vel,
              const double delT,
              const double cv,
              const double gamma,
              const Patch* patch,
              const int mat_id);

void computeNu(CCVariable<double>& nux, 
               CCVariable<double>& nuy, 
               CCVariable<double>& nuz,
               const CCVariable<double>& p, 
               const Patch* patch);  
              
void computeDi(StaticArray<CCVariable<Vector> >& d,
               constCCVariable<double>& rho_old,  
               const CCVariable<double>& press_tmp, 
               constCCVariable<Vector>& vel_old, 
               constCCVariable<double>& speedSound, 
               const Patch* patch,
               const int mat_id);
                     
// end of characteristic boundary condition

/*==========TESTING==========`*/
  template<class T> void Neuman_SFC(T& var, const Patch* patch,
                                Patch::FaceType face,
                                const double value, const Vector& dx,
                                IntVector offset = IntVector(0,0,0));

  void determineSpacingAndSign(Patch::FaceType face, Vector& dx,
                            double& spacing, double& sign);  

  void setBC(SFCXVariable<double>& variable,const std::string& type,
             const std::string& comp, const Patch* p, const int mat_id);

  void setBC(SFCYVariable<double>& variable,const std::string& type,
             const std::string& comp, const Patch* p, const int mat_id);  

  void setBC(SFCZVariable<double>& variable,const std::string& type,
             const std::string& comp, const Patch* p, const int mat_id);   
  
  void setBC(SFCXVariable<Vector>& variable,const std::string& type,
             const Patch* p, const int mat_id);
  
  void ImplicitMatrixBC(CCVariable<Stencil7>& var, const Patch* patch);
  
} // End namespace Uintah
#endif
