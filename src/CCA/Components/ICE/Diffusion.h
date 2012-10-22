/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_ICEDIFFUSION_H
#define UINTAH_ICEDIFFUSION_H
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

namespace Uintah { 
  class DataWarehouse;
  
  void q_flux_allFaces(DataWarehouse* new_dw,
                       const Patch* patch,
                       const bool use_vol_frac,                                
                       const CCVariable<double>& Temp_CC,          
                       const CCVariable<double>& diff_coeff,
                       const CCVariable<double>& vol_frac_CC,                
                       SFCXVariable<double>& q_X_FC,               
                       SFCYVariable<double>& q_Y_FC,               
                       SFCZVariable<double>& q_Z_FC);

   void scalarDiffusionOperator(DataWarehouse* new_dw,
                         const Patch* patch,
                         const bool use_vol_frac,
                         const CCVariable<double>& q_CC,
                         const CCVariable<double>& vol_frac_CC,
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
                     constCCVariable<double>& vol_frac_CC,  
                     constCCVariable<Vector>& vel_CC,     
                     const CCVariable<double>& viscosity,               
                     const Vector dx,                      
                     SFCXVariable<Vector>& tau_X_FC);      

   void computeTauY( const Patch* patch,
                     constCCVariable<double>& vol_frac_CC,  
                     constCCVariable<Vector>& vel_CC,     
                     const CCVariable<double>& viscosity,              
                     const Vector dx,                      
                     SFCYVariable<Vector>& tau_Y_FC);      

   void computeTauZ( const Patch* patch,
                     constCCVariable<double>& vol_frac_CC,  
                     constCCVariable<Vector>& vel_CC,     
                     const CCVariable<double>& viscosity,              
                     const Vector dx,                      
                     SFCZVariable<Vector>& tau_Z_FC);                 
} // End namespace Uintah

#endif
