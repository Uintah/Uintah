
#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/C_BC_driver.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>

using namespace Uintah;
namespace Uintah {
//__________________________________
// Function~  add the computes and requires for each of the custom BC
//______________________________________________________________________
void computesRequires_CustomBCs(Task* t, 
                                const string& where,
                                ICELabel* lb,
                                const MaterialSubset* ice_matls,
                                customBC_var_basket* C_BC_basket)
{
  if(C_BC_basket->usingNG_nozzle){        // NG nozzle 
    addRequires_NGNozzle(t, where,  lb, ice_matls);
  }   
  if(C_BC_basket->usingLodi){             // LODI         
    addRequires_Lodi( t, where,  lb, ice_matls, C_BC_basket->Lodi_var_basket);
  }
  if(C_BC_basket->usingMicroSlipBCs){     // MicroSlip          
    addRequires_MicroSlip( t, where,  lb, ice_matls, C_BC_basket->Slip_var_basket);
  }
  if(C_BC_basket->using_MMS_BCs){     // method of manufactured solutions         
    addRequires_MMS( t, where,  lb, ice_matls);
  }         
}
//______________________________________________________________________
// Function:  preprocess_CustomBCs
// Purpose:   Get variables and precompute any data before setting the Bcs.
//______________________________________________________________________
void preprocess_CustomBCs(const string& where,
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw,
                          ICELabel* lb,
                          const Patch* patch,
                          const int indx,
                          customBC_var_basket* C_BC_basket)
{
  delt_vartype delT;
  const Level* level = patch->getLevel();
  old_dw->get(delT, C_BC_basket->sharedState->get_delt_label(),level);
  //__________________________________
  //    NG_nozzle
  if(C_BC_basket->usingNG_nozzle){        // NG nozzle 
    C_BC_basket->ng = scinew NG_BC_vars;
    C_BC_basket->ng->dataArchiver = C_BC_basket->dataArchiver;
    
    getVars_for_NGNozzle(old_dw, new_dw, lb, patch,where,
                         C_BC_basket->setNGBcs,
                         C_BC_basket->ng );
  }   
  //__________________________________
  //   LODI
  if(C_BC_basket->usingLodi){  
    C_BC_basket->lv = scinew Lodi_vars();
    
    preprocess_Lodi_BCs( old_dw, new_dw, lb, patch, where,
                       indx,  
                       C_BC_basket->sharedState,
                       C_BC_basket->setLodiBcs,
                       C_BC_basket->lv, 
                       C_BC_basket->Lodi_var_basket);        
  }
  //__________________________________
  //  micro slip boundary conditions
  if(C_BC_basket->usingMicroSlipBCs){  
    C_BC_basket->sv = scinew Slip_vars();
    
    preprocess_MicroSlip_BCs( old_dw, new_dw, lb, patch, where,
                              indx,  
                              C_BC_basket->sharedState,
                              C_BC_basket->setMicroSlipBcs,
                              C_BC_basket->sv, 
                              C_BC_basket->Slip_var_basket);        
  }
  //__________________________________
  //  method of manufactured solutions boundary conditions
  if(C_BC_basket->using_MMS_BCs){  
    C_BC_basket->mms_v = scinew mms_vars();
    C_BC_basket->mms_v->delT = (double)delT;
    C_BC_basket->mms_var_basket->delT= (double)delT;
    preprocess_MMS_BCs( new_dw,old_dw, lb,indx,patch, where,
                        C_BC_basket->set_MMS_BCs, 
                        C_BC_basket->mms_v);        
  }         
}

//______________________________________________________________________
// Function:   delete_CustomBCs
//______________________________________________________________________
void delete_CustomBCs(customBC_var_basket* C_BC_basket)
{
  if(C_BC_basket->usingNG_nozzle){
    if(C_BC_basket->ng) {
      delete C_BC_basket->ng;
    }
  }
  if(C_BC_basket->usingLodi){  
    if(C_BC_basket->lv) {
      delete C_BC_basket->lv;
    }
  }
  if(C_BC_basket->usingMicroSlipBCs){  
    if(C_BC_basket->sv) {
      delete C_BC_basket->sv;
    }
  }
  if(C_BC_basket->using_MMS_BCs){
    if(C_BC_basket->mms_v) {
      delete C_BC_basket->mms_v;
    } 
  }
}


}  // using namespace Uintah
