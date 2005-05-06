#ifndef Packages_Uintah_CCA_Components_Ice_CustomBCs_C_BC_driver_h
#define Packages_Uintah_CCA_Components_Ice_CustomBCs_C_BC_driver_h

#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/microSlipBCs.h>
#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/NG_NozzleBCs.h>
#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/LODI2.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>

namespace Uintah {
  class DataWarehouse;

  //_____________________________________________________________
  // This struct contains misc. variables that are need by the 
  // the different custom boundary conditions
  struct customBC_var_basket {

    customBC_var_basket() {
      Lodi_var_basket = 0;
      lv = 0;
      ng = 0;
      sv = 0;
    };
    ~customBC_var_basket() {};
    // LODI boundary condtions
    bool usingLodi;
    Lodi_variable_basket* Lodi_var_basket;
    Lodi_vars* lv;
    bool setLodiBcs;
 
    // Northrup Grumman Boundary Conditions
    bool usingNG_nozzle;
    bool setNGBcs;
    NG_BC_vars* ng;
    
    // Micro slip boundary conditions
    bool usingMicroSlipBCs;
    bool setMicroSlipBcs;
    Slip_vars* sv;
    Slip_variable_basket* Slip_var_basket;
    
    Output* dataArchiver; 
    SimulationStateP sharedState;

  };
  
  
  void computesRequires_CustomBCs(Task* t, 
                                  const string& where,
                                  ICELabel* lb,
                                  const MaterialSubset* ice_matls,
                                  customBC_var_basket* C_BC_basket);
 
  void preprocess_CustomBCs(const string& where,
                            DataWarehouse* old_dw, 
                            DataWarehouse* new_dw,
                            ICELabel* lb,
                            const Patch* patch,
                            const int indx,
                            customBC_var_basket* C_BC_basket);
                            
  void delete_CustomBCs(customBC_var_basket* C_BC_basket);
  
} // End namespace Uintah  
#endif
