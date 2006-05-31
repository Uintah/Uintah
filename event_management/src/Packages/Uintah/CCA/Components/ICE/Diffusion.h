#ifndef UINTAH_ICEDIFFUSION_H
#define UINTAH_ICEDIFFUSION_H
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>

namespace Uintah { 
  class DataWarehouse;
  
  void q_flux_allFaces(DataWarehouse* new_dw,
                       const Patch* patch,
                       const bool use_vol_frac,                                
                       const CCVariable<double>& Temp_CC,          
                       const CCVariable<double>& diff_coeff,
                       const SFCXVariable<double>& vol_fracX_FC,  
                       const SFCYVariable<double>& vol_fracY_FC,  
                       const SFCZVariable<double>& vol_fracZ_FC,                  
                       SFCXVariable<double>& q_X_FC,               
                       SFCYVariable<double>& q_Y_FC,               
                       SFCZVariable<double>& q_Z_FC);

  void scalarDiffusionOperator(DataWarehouse* new_dw,
                         const Patch* patch,
                         const bool use_vol_frac,
                         const CCVariable<double>& q_CC,
                         const SFCXVariable<double>& vol_fracX_FC,
                         const SFCYVariable<double>& vol_fracY_FC,
                         const SFCZVariable<double>& vol_fracZ_FC,
                         CCVariable<double>& q_diffusion_src,
                         const CCVariable<double>& diff_coeff,
                         const double delT);            

  template <class T> 
  void q_flux_FC(CellIterator iter, 
                 IntVector adj_offset,
                 const CCVariable<double>& diff_coeff,
                 const double dx,
                 const T& vol_frac_FC,   
                 const CCVariable<double>& Temp_CC,
                 T& q_FC,
                 const bool use_vol_frac);
                 
   void computeTauX( const Patch* patch,
                     constSFCXVariable<double>& vol_fracX_FC,  
                     constCCVariable<Vector>& vel_CC,     
                     const CCVariable<double>& viscosity,               
                     const Vector dx,                      
                     SFCXVariable<Vector>& tau_X_FC);      

   void computeTauY( const Patch* patch,
                     constSFCYVariable<double>& vol_fracX_FC,  
                     constCCVariable<Vector>& vel_CC,     
                     const CCVariable<double>& viscosity,              
                     const Vector dx,                      
                     SFCYVariable<Vector>& tau_Y_FC);      

   void computeTauZ( const Patch* patch,
                     constSFCZVariable<double>& vol_fracZ_FC,  
                     constCCVariable<Vector>& vel_CC,     
                     const CCVariable<double>& viscosity,              
                     const Vector dx,                      
                     SFCZVariable<Vector>& tau_Z_FC);                 
} // End namespace Uintah

#endif
