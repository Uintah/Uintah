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

#ifndef Packages_Uintah_CCA_Components_Ice_CustomBCs_C_BC_driver_h
#define Packages_Uintah_CCA_Components_Ice_CustomBCs_C_BC_driver_h
#include <CCA/Components/ICE/CustomBCs/sine.h>
#include <CCA/Components/ICE/CustomBCs/MMS_BCs.h>
#include <CCA/Components/ICE/CustomBCs/microSlipBCs.h>
#include <CCA/Components/ICE/CustomBCs/LODI2.h>
#include <CCA/Components/ICE/CustomBCs/inletVelocity.h>
#include <Core/Grid/SimulationState.h>

namespace Uintah {
  class DataWarehouse;
 
  //_____________________________________________________________
  // This struct contains misc. variables that are needed by the 
  // the different custom boundary conditions and are considered global
  //  Don't put anything in here that is local to a task.cu
  struct customBC_globalVars {

    customBC_globalVars() {
      Lodi_var_basket     = NULL;
      Slip_var_basket     = NULL;
      mms_var_basket      = NULL;
      sine_var_basket     = NULL;
      inletVel_var_basket = NULL;

      usingLodi         = false;
      usingMicroSlipBCs = false;
      using_MMS_BCs     = false;
      using_Sine_BCs    = false;
      using_inletVel_BCs= false;

    };
    
    ~customBC_globalVars() {};
    // LODI boundary condtions
    bool usingLodi;
    Lodi_variable_basket* Lodi_var_basket;
    
    // Micro slip boundary conditions
    bool usingMicroSlipBCs;
    Slip_variable_basket* Slip_var_basket;
    
    // method of manufactured Solution BCs
    bool using_MMS_BCs;
    mms_variable_basket* mms_var_basket;
    
    // Sine boundary conditions
    bool using_Sine_BCs;
    sine_variable_basket* sine_var_basket;
    
    // powerLawProfile or logLawProfile inlet velocity profile
    bool using_inletVel_BCs;
    inletVel_variable_basket* inletVel_var_basket;
    
    SimulationStateP sharedState;
    Vector d_gravity;

  };
  
  //_____________________________________________________________
  // This struct contains variables that are local to a task or function
  struct customBC_localVars {

    customBC_localVars() {
      
      lv          = NULL;
      sv          = NULL;
      mms_v       = NULL;
      sine_v      = NULL;
      
      setLodiBcs        = false;
      setMicroSlipBcs   = false;
      set_MMS_BCs       = false;
      set_Sine_BCs      = false;
      set_inletVel_BCs  = false;
      recursiveTask     = false;
    };
    
    ~customBC_localVars() {};
    
    // are tasks recursive
    bool recursiveTask;
    
    // LODI boundary condtions
    bool setLodiBcs;
    Lodi_vars* lv;
    
    // Micro slip boundary conditions
    bool setMicroSlipBcs;
    Slip_vars* sv;
    
    // method of manufactured Solution BCs
    bool set_MMS_BCs;
    mms_vars* mms_v;
    
    // Sine boundary conditions
    bool set_Sine_BCs;
    sine_vars* sine_v;
    
    // powerLawProfile or logLawProfile inlet velocity profile
    bool set_inletVel_BCs;
    inletVel_vars* inletVel_v;
  };  

  //______________________________________________________________________
  //
  
   void computesRequires_CustomBCs(Task* t, 
                                   const string& where,                      
                                   ICELabel* lb,                             
                                   const MaterialSubset* ice_matls,          
                                   customBC_globalVars* global,
                                   const bool recursiveTask = false);        
 
   void preprocess_CustomBCs(const string& where,
                             DataWarehouse* old_dw,                    
                             DataWarehouse* new_dw,                    
                             ICELabel* lb,                             
                             const Patch* patch,                       
                             const int indx,                           
                             customBC_globalVars* global,
                             customBC_localVars* localVars);        
                            
   void delete_CustomBCs( customBC_globalVars* global,
                          customBC_localVars* local );
  
} // End namespace Uintah  
#endif
