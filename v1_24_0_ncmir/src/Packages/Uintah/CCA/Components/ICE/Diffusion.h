#ifndef UINTAH_ICEDIFFUSION_H
#define UINTAH_ICEDIFFUSION_H
//#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>

namespace Uintah { 
  class DataWarehouse;
  
  void q_flux_allFaces(DataWarehouse* new_dw,
                       const Patch* patch,
                       const bool use_vol_frac,                        
                       const CCVariable<double>& rho_CC,            
                       const CCVariable<double>& sp_vol_CC,         
                       const CCVariable<double>& Temp_CC,          
                       const CCVariable<double>& diff_coeff,                
                       SFCXVariable<double>& q_X_FC,               
                       SFCYVariable<double>& q_Y_FC,               
                       SFCZVariable<double>& q_Z_FC);

  void scalarDiffusionOperator(DataWarehouse* new_dw,
                         const Patch* patch,
                         const bool use_vol_frac,
                         const CCVariable<double>& rho_CC,     
                         const CCVariable<double>& sp_vol_CC,  
                         const CCVariable<double>& q_CC,
                         CCVariable<double>& q_diffusion_src,
                         const CCVariable<double>& diff_coeff,
                         const double delT);            

  template <class T> 
  void q_flux_FC(CellIterator iter, 
                 IntVector adj_offset,
                 const CCVariable<double>& diff_coeff,
                 const double dx,
                 const CCVariable<double>& rho_CC,      
                 const CCVariable<double>& sp_vol_CC,   
                 const CCVariable<double>& Temp_CC,
                 T& q_FC,
                 const bool use_vol_frac);
                 
   void computeTauX( const Patch* patch,
                     constCCVariable<double>& rho_CC,     
                     constCCVariable<double>& sp_vol_CC,  
                     constCCVariable<Vector>& vel_CC,     
                     const CCVariable<double>& viscosity,               
                     const Vector dx,                      
                     SFCXVariable<Vector>& tau_X_FC);      

   void computeTauY( const Patch* patch,
                     constCCVariable<double>& rho_CC,     
                     constCCVariable<double>& sp_vol_CC,  
                     constCCVariable<Vector>& vel_CC,     
                     const CCVariable<double>& viscosity,              
                     const Vector dx,                      
                     SFCYVariable<Vector>& tau_Y_FC);      

   void computeTauZ( const Patch* patch,
                     constCCVariable<double>& rho_CC,     
                     constCCVariable<double>& sp_vol_CC,  
                     constCCVariable<Vector>& vel_CC,     
                     const CCVariable<double>& viscosity,              
                     const Vector dx,                      
                     SFCZVariable<Vector>& tau_Z_FC);                 
} // End namespace Uintah

#endif
