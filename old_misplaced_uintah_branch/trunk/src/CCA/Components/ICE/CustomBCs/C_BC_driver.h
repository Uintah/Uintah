#ifndef Packages_Uintah_CCA_Components_Ice_CustomBCs_C_BC_driver_h
#define Packages_Uintah_CCA_Components_Ice_CustomBCs_C_BC_driver_h
#include <CCA/Components/ICE/CustomBCs/sine.h>
#include <CCA/Components/ICE/CustomBCs/MMS_BCs.h>
#include <CCA/Components/ICE/CustomBCs/microSlipBCs.h>
#include <CCA/Components/ICE/CustomBCs/LODI2.h>
#include <Core/Grid/SimulationState.h>

#include <CCA/Components/ICE/uintahshare.h>
namespace Uintah {
  class DataWarehouse;

  //_____________________________________________________________
  // This struct contains misc. variables that are need by the 
  // the different custom boundary conditions
  struct customBC_var_basket {

    customBC_var_basket() {
      Lodi_var_basket = 0;
      lv = 0;
      sv = 0;
    };
    ~customBC_var_basket() {};
    // LODI boundary condtions
    bool usingLodi;
    Lodi_variable_basket* Lodi_var_basket;
    Lodi_vars* lv;
    bool setLodiBcs;
    
    // Micro slip boundary conditions
    bool usingMicroSlipBCs;
    bool setMicroSlipBcs;
    Slip_vars* sv;
    Slip_variable_basket* Slip_var_basket;
    
    // method of manufactured Solution BCs
    bool using_MMS_BCs;
    bool set_MMS_BCs;
    mms_vars* mms_v;
    mms_variable_basket* mms_var_basket;

    bool using_Sine_BCs;
    bool set_Sine_BCs;
    sine_vars* sine_v;
    sine_variable_basket* sine_var_basket;
    
    SimulationStateP sharedState;

  };
  
  
  UINTAHSHARE void computesRequires_CustomBCs(Task* t, 
                                           const string& where,
                                           ICELabel* lb,
                                           const MaterialSubset* ice_matls,
                                           customBC_var_basket* C_BC_basket);
 
  UINTAHSHARE void preprocess_CustomBCs(const string& where,
                                     DataWarehouse* old_dw, 
                                     DataWarehouse* new_dw,
                                     ICELabel* lb,
                                     const Patch* patch,
                                     const int indx,
                                     customBC_var_basket* C_BC_basket);
                            
  UINTAHSHARE void delete_CustomBCs(customBC_var_basket* C_BC_basket);
  
} // End namespace Uintah  
#endif
