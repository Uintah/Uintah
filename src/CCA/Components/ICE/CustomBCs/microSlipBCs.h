/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#ifndef Packages_Uintah_CCA_Components_Ice_Slip_h
#define Packages_Uintah_CCA_Components_Ice_Slip_h

#include <CCA/Ports/DataWarehouse.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Containers/StaticArray.h>
#include <typeinfo>
using namespace Uintah;
namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. variables that are carried around
  struct Slip_variable_basket{
    double alpha_momentum;        // momentum accommidation coeff
    double alpha_temperature;     // temperature accomidation coeff.
  };    
  //____________________________________________________________
  // This struct contains the additional variables required to compute
  // mean free path and the gradients
  struct Slip_vars{
    constCCVariable<double> gamma;   
    constCCVariable<double> rho_CC;
    constCCVariable<Vector> vel_CC;
    constCCVariable<double> press_CC;        
    constCCVariable<double> Temp_CC;
    constCCVariable<double> viscosity;
    CCVariable<double> lamda;    // mean free path
    double alpha_momentum;
    double alpha_temperature; 
  };
  
  bool read_MicroSlip_BC_inputs(const ProblemSpecP&,
                                Slip_variable_basket* svb);
                                 
  void addRequires_MicroSlip(Task* t, 
                             const string& where,
                             ICELabel* lb,
                             const MaterialSubset* ice_matls,
                             Slip_variable_basket* sv);
                      
  void preprocess_MicroSlip_BCs(DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                ICELabel* lb,
                                const Patch* patch,
                                const string& where,
                                const int indx,
                                SimulationStateP& sharedState,
                                bool& setSlipBcs,
                                Slip_vars* sv,
                                Slip_variable_basket* svb);
                                  
  bool is_MicroSlip_face(const Patch* patch,
                         Patch::FaceType face,
                         SimulationStateP& sharedState);
                  
  int set_MicroSlipVelocity_BC(const Patch* patch,
                              const Patch::FaceType face,
                              CCVariable<Vector>& vel_CC,
                              const string& var_desc,
                              Iterator& bound_ptr,
                              const string& bc_kind,
                              const Vector wall_velocity,
                              Slip_vars* sv);

  int  set_MicroSlipTemperature_BC(const Patch* patch,
                              const Patch::FaceType face,
                              CCVariable<double>& temp_CC,
                              Iterator& bound_ptr,
                              const string& bc_kind,
                              const double wall_temperature,
                              Slip_vars* sv);                          
} // End namespace Uintah
#endif
