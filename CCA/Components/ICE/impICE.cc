#include "ICE.h"
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h> 
#include <Packages/Uintah/Core/Grid/Stencil7.h>


using namespace SCIRun;
using namespace Uintah;

static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false);


/* ---------------------------------------------------------------------
 Function~  ICE::scheduleImplicitPressureSolve--
_____________________________________________________________________*/
void ICE::scheduleImplicitPressureSolve(  SchedulerP& sched,
                                          const LevelP& level,
                                          SolverInterface* solver, 
                                          const SolverParameters* solver_parameters,
                                          const PatchSet* patches,
                                          const MaterialSubset* press_matl,
                                          const MaterialSubset* ice_matls,
                                          const MaterialSubset* mpm_matls,
                                          const MaterialSet* all_matls)
{
  Task* t;

  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None; 
  //__________________________________
  //  Form the matrix
  cout_doing << "ICE::scheduleSetupMatrix" << endl;
  t = scinew Task("setupMatrix", this, &ICE::setupMatrix);
  t->requires( Task::OldDW, lb->delTLabel);
  t->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,1);
  t->requires( Task::NewDW, lb->uvel_FCMELabel,     gac,2);
  t->requires( Task::NewDW, lb->vvel_FCMELabel,     gac,2);
  t->requires( Task::NewDW, lb->wvel_FCMELabel,     gac,2);
  t->requires( Task::NewDW, lb->rho_CCLabel,        gn);
  
  t->computes(lb->matrixLabel);
  t->computes(lb->rhsLabel);
  t->computes(lb->initialGuessLabel);
  sched->addTask(t, patches, all_matls);

  //__________________________________
  //  solve for delP
  cout_doing << "ICE::scheduleSolve" << endl;
  solver->scheduleSolve(level, sched, all_matls,
                        lb->matrixLabel, lb->imp_delPLabel, false,
                        lb->rhsLabel,    lb->initialGuessLabel,
                        Task::NewDW,     solver_parameters);
                        
  //__________________________________
  // update the pressure
  cout_doing << "ICE::scheduleUpdatePressure" << endl;
  t = scinew Task("updatePressure", this, &ICE::updatePressure);
  t->requires(Task::NewDW, lb->imp_delPLabel,        press_matl,  gac, 1); 
  t->requires(Task::NewDW, lb->press_equil_CCLabel,  press_matl,  gn);  
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,                    gn);
  t->requires(Task::NewDW, lb->rho_CCLabel,                      gn);
    
  t->computes(lb->delP_DilatateLabel, press_matl);
  t->computes(lb->delP_MassXLabel,    press_matl);  
  t->computes(lb->press_CCLabel,      press_matl); 
  t->computes(lb->sum_rho_CCLabel,    press_matl);  // only one mat subset     
      
  sched->addTask(t, patches, all_matls);
                       
}

/* --------------------------------------------------------------------- 
 Function~  ICE::setupMatrix-- 
_____________________________________________________________________*/
void ICE::setupMatrix(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* ,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<<"Doing setupMatrix on patch "
              << patch->getID() <<"\t\t\t ICE" << endl;
    
    int numMatls  = d_sharedState->getNumMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx     = patch->dCell();
    double vol    = dx.x()*dx.y()*dx.z();
    double invvol = 1./vol;

    Advector* advector = d_advector->clone(new_dw,patch);
    CCVariable<double> q_CC, q_advected;                     
    CCVariable<double> rhs, initialGuess;
    CCVariable<Stencil7> matrix; 
    
    const IntVector gc(1,1,1);
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->allocateAndPut(matrix,       lb->matrixLabel,       0, patch);              
    new_dw->allocateAndPut(rhs,          lb->rhsLabel,          0, patch);   
    new_dw->allocateAndPut(initialGuess, lb->initialGuessLabel, 0, patch); 
    new_dw->allocateTemporary(q_advected, patch);
    new_dw->allocateTemporary(q_CC,       patch, gac,1);    
    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;
      constCCVariable<double> vol_frac;
      constCCVariable<double> rho_CC;      

      new_dw->get(uvel_FC, lb->uvel_FCMELabel,         indx,patch,gac, 2);   
      new_dw->get(vvel_FC, lb->vvel_FCMELabel,         indx,patch,gac, 2);   
      new_dw->get(wvel_FC, lb->wvel_FCMELabel,         indx,patch,gac, 2);   
      new_dw->get(vol_frac,lb->vol_frac_CCLabel,       indx,patch,gac, 1);   
      new_dw->get(rho_CC,  lb->rho_CCLabel,            indx,patch,gn,  0);  
         
      //__________________________________
      // Advection preprocessing
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,indx); 

      for(CellIterator iter = patch->getCellIterator(gc); !iter.done();iter++){
       IntVector c = *iter;
        q_CC[c] = vol_frac[c] * invvol;
      }
      //__________________________________
      //   First order advection of q_CC 
      advector->advectQ(q_CC, patch, q_advected, new_dw);

    }  //matl loop
    delete advector;  
    
    cout << "setup matrix: done with advection" <<endl;     
    //__________________________________
    //  setup up the matrix an rhs
    // DUMMY matrix for right now
    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      matrix[c].p = 1.0;
      matrix[c].n = 1.0;
      matrix[c].s = 1.0;
      matrix[c].e = 1.0;
      matrix[c].w = 1.0;
      matrix[c].t = 1.0;
      matrix[c].b = 1.0;
      rhs[c]      = 1.0;
     initialGuess[c] = 0.0;
    }
    cout << "setup Matrix: done setting up matrix"<<endl;          
  }
}

/* --------------------------------------------------------------------- 
 Function~  ICE::updatePressure-- 
_____________________________________________________________________*/
void ICE::updatePressure(const ProcessorGroup*,
                         const PatchSubset* patches,                    
                         const MaterialSubset* ,                        
                         DataWarehouse* old_dw,                         
                         DataWarehouse* new_dw)                         
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<<"Doing updatePressure on patch "
              << patch->getID() <<"\t\t\t ICE" << endl;
   
    int numMatls  = d_sharedState->getNumMatls(); 
      
    CCVariable<double> press_CC;
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> sum_rho_CC;        
    constCCVariable<double> imp_delP;
    constCCVariable<double> pressure;
    constCCVariable<double> rho_CC;
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
    
    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(imp_delP,      lb->imp_delPLabel,        0,patch,gn,0);
    new_dw->get(pressure,      lb->press_equil_CCLabel,  0,patch,gn,0);
    
    new_dw->allocateAndPut(delP_Dilatate,lb->delP_DilatateLabel,0, patch);
    new_dw->allocateAndPut(delP_MassX,   lb->delP_MassXLabel,   0, patch);
    new_dw->allocateAndPut(press_CC,     lb->press_CCLabel,     0, patch);
    new_dw->allocateAndPut(sum_rho_CC,   lb->sum_rho_CCLabel,   0, patch);

    sum_rho_CC.initialize(0.0);
     
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->get(sp_vol_CC[m],lb->sp_vol_CCLabel, indx,patch,gn,0);
      new_dw->get(rho_CC,      lb->rho_CCLabel,    indx,patch,gn,0);
      //__________________________________
      //  compute sum_rho_CC used by press_FC
      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        sum_rho_CC[c] += rho_CC[c];
      } 
    }    
/*`==========TESTING==========*/
    delP_Dilatate.initialize(0.0);
    delP_MassX.initialize(0.0); 
/*==========TESTING==========`*/
    
    //__________________________________
    //  add delP to press
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      press_CC[c] = pressure[c];
    }

    setBC(press_CC, sp_vol_CC[SURROUND_MAT],
          "sp_vol", "Pressure", patch ,d_sharedState, 0, new_dw);
  }
}
