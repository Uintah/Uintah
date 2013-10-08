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
                                customBC_globalVars* globalVars,
                                const bool recursiveTask)
{   
  if(globalVars->usingLodi){             // LODI         
    addRequires_Lodi( t, where,  lb, ice_matls, globalVars->Lodi_var_basket);
  }
  if(globalVars->usingMicroSlipBCs){     // MicroSlip          
    addRequires_MicroSlip( t, where,  lb, ice_matls, globalVars->Slip_var_basket);
  }
  if(globalVars->using_MMS_BCs){         // method of manufactured solutions         
    addRequires_MMS( t, where,  lb, ice_matls);
  }
  if(globalVars->using_Sine_BCs){         // method of manufactured solutions         
    addRequires_Sine( t, where,  lb, ice_matls);
  }
  if(globalVars->using_inletVel_BCs){               
    addRequires_inletVel( t, where,  lb, ice_matls, recursiveTask);
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
                          customBC_globalVars* globalVars,
                          customBC_localVars* localVars)
{
  delt_vartype delT;
  const Level* level = patch->getLevel();
  old_dw->get(delT, globalVars->sharedState->get_delt_label(),level);
   
  //__________________________________
  //   LODI
  if( globalVars->usingLodi ){  
    localVars->lv = scinew Lodi_vars();
    
    preprocess_Lodi_BCs( old_dw, new_dw, lb, patch, where,
                       indx,  
                       globalVars->sharedState,
                       localVars->setLodiBcs,
                       localVars->lv, 
                       globalVars->Lodi_var_basket);        
  }
  //__________________________________
  //  micro slip boundary conditions
  if( globalVars->usingMicroSlipBCs ){  
    localVars->sv = scinew Slip_vars();
    
    preprocess_MicroSlip_BCs( old_dw, new_dw, lb, patch, where,
                              indx,  
                              globalVars->sharedState,
                              localVars->setMicroSlipBcs,
                              localVars->sv, 
                              globalVars->Slip_var_basket);        
  }
  //__________________________________
  //  method of manufactured solutions boundary conditions
  if( globalVars->using_MMS_BCs ){  
    localVars->mms_v = scinew mms_vars();
    localVars->mms_v->delT = (double)delT;
    preprocess_MMS_BCs( new_dw,old_dw, lb,indx,patch, where,
                        localVars->set_MMS_BCs, 
                        localVars->mms_v);        
  }
  //__________________________________
  //  Sine boundary conditions
  if( globalVars->using_Sine_BCs ){  
    localVars->sine_v = scinew sine_vars();
    localVars->sine_v->delT = (double)delT;
    globalVars->sine_var_basket->delT= (double)delT;
    preprocess_Sine_BCs( new_dw,old_dw, lb,indx,patch, where,
                        localVars->set_Sine_BCs, 
                        localVars->sine_v);        
  }  
  
  //__________________________________
  //  inletVelocity conditions
  if( globalVars->using_inletVel_BCs ){
    localVars->inletVel_v = scinew inletVel_vars();
    preprocess_inletVelocity_BCs(  old_dw, lb, indx, patch, where, 
                                   localVars->set_inletVel_BCs,
                                   localVars->recursiveTask,
                                   localVars->inletVel_v );        
  }       
}

//______________________________________________________________________
// Function:   delete_CustomBCs
//______________________________________________________________________
void delete_CustomBCs(customBC_globalVars* global,
                      customBC_localVars* local)
{
  if( global->usingLodi ){  
    if( local->lv ) {
      delete local->lv;
    }
  }
  if( global->usingMicroSlipBCs ){  
    if( local->sv ) {
      delete local->sv;
    }
  }
  if( global->using_MMS_BCs ){
    if(local->mms_v) {
      delete local->mms_v;
    } 
  }
  if( global->using_Sine_BCs ){
    if( local->sine_v ) {
      delete local->sine_v;
    } 
  }
  
  if( global->using_inletVel_BCs ){
    if(local->inletVel_v ){
      delete local->inletVel_v;
    }
  }
  
  delete local;
}


}  // using namespace Uintah
