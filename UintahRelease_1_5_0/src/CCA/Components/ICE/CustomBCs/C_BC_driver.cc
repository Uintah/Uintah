/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <CCA/Components/ICE/CustomBCs/C_BC_driver.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>

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
  if(C_BC_basket->usingLodi){             // LODI         
    addRequires_Lodi( t, where,  lb, ice_matls, C_BC_basket->Lodi_var_basket);
  }
  if(C_BC_basket->usingMicroSlipBCs){     // MicroSlip          
    addRequires_MicroSlip( t, where,  lb, ice_matls, C_BC_basket->Slip_var_basket);
  }
  if(C_BC_basket->using_MMS_BCs){         // method of manufactured solutions         
    addRequires_MMS( t, where,  lb, ice_matls);
  }
  if(C_BC_basket->using_Sine_BCs){         // method of manufactured solutions         
    addRequires_Sine( t, where,  lb, ice_matls);
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
   
   C_BC_basket->setMicroSlipBcs = false;
   C_BC_basket->set_MMS_BCs     = false;
   C_BC_basket->set_Sine_BCs    = false;
   C_BC_basket->setLodiBcs      = false;
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
  //__________________________________
  //  Sine boundary conditions
  if(C_BC_basket->using_Sine_BCs){  
    C_BC_basket->sine_v = scinew sine_vars();
    C_BC_basket->sine_v->delT = (double)delT;
    C_BC_basket->sine_var_basket->delT= (double)delT;
    preprocess_Sine_BCs( new_dw,old_dw, lb,indx,patch, where,
                        C_BC_basket->set_Sine_BCs, 
                        C_BC_basket->sine_v);        
  }         
}

//______________________________________________________________________
// Function:   delete_CustomBCs
//______________________________________________________________________
void delete_CustomBCs(customBC_var_basket* C_BC_basket)
{
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
  if(C_BC_basket->using_Sine_BCs){
    if(C_BC_basket->sine_v) {
      delete C_BC_basket->sine_v;
    } 
  }
}


}  // using namespace Uintah
