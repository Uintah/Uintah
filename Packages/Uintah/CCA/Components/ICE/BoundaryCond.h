#ifndef Packages_Uintah_CCA_Components_Ice_BoundaryCond_h
#define Packages_Uintah_CCA_Components_Ice_BoundaryCond_h

#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Core/Containers/StaticArray.h>



/*`==========TESTING==========*/
//#define LODI_BCS
#undef  LODI_BCS 
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

// The following codes are written for chracteristic boundary condition
/*`==========TESTING==========*/
#ifdef LODI_BCS
void setBCPress_LODI(CCVariable<double>& press_CC,
                     StaticArray<constCCVariable<double> >& sp_vol_CC,
                     StaticArray<constCCVariable<double> >& Temp_CC,
                     StaticArray<constCCVariable<double> >& f_theta,
                     const string& kind, 
                     const Patch* patch,
                     SimulationStateP& sharedState, 
                     const int mat_id,
                     DataWarehouse* new_dw);

void setBCDensityLODI(CCVariable<double>& rho_CC,
                const CCVariable<double>& d1_x,                       
                const CCVariable<double>& d1_y,                       
                const CCVariable<double>& d1_z,                       
                const CCVariable<double>& nux,
                const CCVariable<double>& nuy,
                const CCVariable<double>& nuz,
                CCVariable<double>& rho_tmp,
                CCVariable<double>& p,
                CCVariable<Vector>& vel,
                constCCVariable<double>& c,
                const double delT,
                const double gamma,
                const double R_gas,
                const Patch* patch, 
                const int mat_id);
              
void setBCVelLODI(CCVariable<Vector>& vel_CC,
            const CCVariable<double>& d1_x, 
            const CCVariable<double>& d3_x, 
            const CCVariable<double>& d4_x, 
            const CCVariable<double>& d1_y,  
            const CCVariable<double>& d3_y, 
            const CCVariable<double>& d4_y, 
            const CCVariable<double>& d1_z,  
            const CCVariable<double>& d3_z, 
            const CCVariable<double>& d4_z, 
            const CCVariable<double>& nux,
            const CCVariable<double>& nuy,
            const CCVariable<double>& nuz,
            CCVariable<double>& rho_tmp,
            const CCVariable<double>& p,
            CCVariable<Vector>& vel,
            constCCVariable<double>& c,
            const double delT,
            const double gamma,
            const double R_gas,
            const Patch* patch, 
            const int mat_id);
              
 void setBCTempLODI(CCVariable<double>& temp_CC,
              const CCVariable<double>& d1_x, 
              const CCVariable<double>& d2_x, 
              const CCVariable<double>& d3_x, 
              const CCVariable<double>& d4_x, 
              const CCVariable<double>& d5_x,
              const CCVariable<double>& d1_y, 
              const CCVariable<double>& d2_y, 
              const CCVariable<double>& d3_y, 
              const CCVariable<double>& d4_y, 
              const CCVariable<double>& d5_y,
              const CCVariable<double>& d1_z, 
              const CCVariable<double>& d2_z, 
              const CCVariable<double>& d3_z, 
              const CCVariable<double>& d4_z, 
              const CCVariable<double>& d5_z,
              const CCVariable<double>& e,
              const CCVariable<double>& rho_CC,
              const CCVariable<double>& nux,
              const CCVariable<double>& nuy,
              const CCVariable<double>& nuz,
              CCVariable<double>& rho_tmp,
              CCVariable<double>& p,
              CCVariable<Vector>& vel,
              constCCVariable<double>& c,
              const double delT,
              const double cv,
              const double gamma,
              const Patch* patch, 
              const int mat_id);

void computeDiFirstOrder(const double& faceNormal, double& d1, double& d2, 
                       double& d3, double& d4, double& d5, const double& rho1,
                         const double& rho2,const double& p1, const double& p2, 
                        const double& c, const Vector& vel1, const Vector& vel2,
                         const double& vel_cross_bound, const double& dx);

void computeDiSecondOrder(const double& faceNormal, double& d1, double& d2, double& d3, double& d4,
                          double& d5, const double& rho1, const double& rho2, const double& rho3,
                         const double& p1, const double& p2, const double& p3, const double& c,
                          const Vector& vel1, const Vector& vel2, const Vector& vel3,
                          const double& vel_cross_bound, const double& dx);

void computeNu(CCVariable<double>& nux, 
               CCVariable<double>& nuy, 
               CCVariable<double>& nuz,
               CCVariable<double>& p, 
               const Patch* patch);

void computeEnergy(CCVariable<double>& e,
                   CCVariable<Vector>& vel,
                   CCVariable<double>& rho,
                   const Patch* patch);   
              
void computeLODIFirstOrder(CCVariable<double>& d1_x, 
                           CCVariable<double>& d2_x, 
                           CCVariable<double>& d3_x, 
                           CCVariable<double>& d4_x, 
                           CCVariable<double>& d5_x,
                           CCVariable<double>& d1_y, 
                           CCVariable<double>& d2_y, 
                           CCVariable<double>& d3_y, 
                           CCVariable<double>& d4_y, 
                           CCVariable<double>& d5_y,
                           CCVariable<double>& d1_z, 
                           CCVariable<double>& d2_z, 
                           CCVariable<double>& d3_z, 
                           CCVariable<double>& d4_z, 
                           CCVariable<double>& d5_z,
                           CCVariable<double>& rho_old,  
                           CCVariable<double>& press_tmp, 
                           CCVariable<Vector>& vel_old, 
                      constCCVariable<double>& speedSound, 
                      const Patch* patch,
                      const int mat_id);
                      
void computeLODISecondOrdder(CCVariable<double>& d1_x, 
                            CCVariable<double>& d2_x, 
                            CCVariable<double>& d3_x, 
                            CCVariable<double>& d4_x, 
                            CCVariable<double>& d5_x,
                            CCVariable<double>& d1_y, 
                            CCVariable<double>& d2_y, 
                            CCVariable<double>& d3_y, 
                            CCVariable<double>& d4_y, 
                            CCVariable<double>& d5_y,
                            CCVariable<double>& d1_z, 
                            CCVariable<double>& d2_z, 
                            CCVariable<double>& d3_z, 
                            CCVariable<double>& d4_z, 
                            CCVariable<double>& d5_z,
                       constCCVariable<double>& rho_old,  
                       const CCVariable<double>& press_tmp, 
                       constCCVariable<Vector>& vel_old, 
                       constCCVariable<double>& speedSound, 
                       const Patch* patch,
                       const int mat_id);
// end of characteristic boundary condition
#endif
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
