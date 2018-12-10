/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>

using namespace std;

namespace Uintah {
//__________________________________
// Function~  add the computes and requires for each of the custom BC
//______________________________________________________________________
void computesRequires_CustomBCs(Task* t, 
                                const string& where,
                                ICELabel* lb,
                                const MaterialSubset* ice_matls,
                                customBC_globalVars* gv,
                                const bool recursiveTask)
{   
  if( gv->usingLodi ){             // LODI         
    addRequires_Lodi( t, where,  lb, ice_matls, gv->lodi);
  }
  if( gv->usingMicroSlipBCs ){     // MicroSlip          
    addRequires_MicroSlip( t, where,  lb, ice_matls, gv->slip);
  }
  if( gv->using_MMS_BCs ){         // method of manufactured solutions         
    addRequires_MMS( t, where,  lb, ice_matls);
  }
  if( gv->using_Sine_BCs ){         // method of manufactured solutions         
    addRequires_Sine( t, where,  lb, ice_matls);
  }
  if( gv->using_inletVel_BCs ){               
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
                          customBC_globalVars* gv,
                          customBC_localVars* lv)
{
  simTime_vartype simTime;
  old_dw->get(simTime, lb->simulationTimeLabel);

  delt_vartype delT;
  const Level* level = patch->getLevel();
  old_dw->get(delT, lb->delTLabel, level);
   
  //__________________________________
  //   LODI
  if( gv->usingLodi ){  
    lv->lodi = scinew Lodi_localVars();
    
    preprocess_Lodi_BCs( old_dw, new_dw, lb, patch, where,
                       indx,  
                       gv->materialManager,
                       lv->setLodiBcs,
                       lv->lodi, 
                       gv->lodi);        
  }
  //__________________________________
  //  micro slip boundary conditions
  if( gv->usingMicroSlipBCs ){  
    lv->slip = scinew slip_localVars();
    
    preprocess_MicroSlip_BCs( old_dw, new_dw, lb, patch, where,
                              indx,  
                              gv->materialManager,
                              lv->setMicroSlipBcs,
                              lv->slip, 
                              gv->slip);        
  }
  //__________________________________
  //  method of manufactured solutions boundary conditions
  if( gv->using_MMS_BCs ){  
    lv->mms = scinew mms_localVars();
    lv->mms->simTime = (double)simTime;
    lv->mms->delT = (double)delT;
    preprocess_MMS_BCs( new_dw,old_dw, lb,indx,patch, where,
                        lv->set_MMS_BCs, 
                        lv->mms);        
  }
  //__________________________________
  //  Sine boundary conditions
  if( gv->using_Sine_BCs ){  
    lv->sine = scinew sine_localVars();
    lv->sine->simTime = (double)simTime;
    gv->sine->delT = (double)delT;
    preprocess_Sine_BCs( new_dw,old_dw, lb,indx,patch, where,
                        lv->set_Sine_BCs, 
                        lv->sine);        
  }  
  
  //__________________________________
  //  inletVelocity conditions
  if( gv->using_inletVel_BCs ){
    lv->inletVel = scinew inletVel_localVars();
    preprocess_inletVelocity_BCs(  old_dw, lb, indx, patch, where, 
                                   lv->set_inletVel_BCs,
                                   lv->recursiveTask,
                                   gv->inletVel,
                                   lv->inletVel );        
  }       
}

//______________________________________________________________________
// Function:   delete_CustomBCs
//______________________________________________________________________
void delete_CustomBCs(customBC_globalVars* gv,
                      customBC_localVars* lv)
{
  if( gv->usingLodi ){  
    if( lv->lodi ) {
      delete lv->lodi;
    }
  }
  if( gv->usingMicroSlipBCs ){  
    if( lv->slip ) {
      delete lv->slip;
    }
  }
  if( gv->using_MMS_BCs ){
    if(lv->mms) {
      delete lv->mms;
    } 
  }
  if( gv->using_Sine_BCs ){
    if( lv->sine ) {
      delete lv->sine;
    } 
  }
  
  if( gv->using_inletVel_BCs ){
    if(lv->inletVel ){
      delete lv->inletVel;
    }
  }
  
  delete lv;
}


}  // using namespace Uintah
