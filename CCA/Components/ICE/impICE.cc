#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h> 
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/InternalError.h>

using namespace SCIRun;
using namespace Uintah;

static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false);





/* ---------------------------------------------------------------------
                               T O   D O
_____________________________________________________________________*/

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleSetupMatrix--
_____________________________________________________________________*/
void ICE::scheduleSetupMatrix(  SchedulerP& sched,
                               const LevelP&,
                               const PatchSet* patches,
                               const MaterialSubset* one_matl,
                               const MaterialSet* all_matls)
{
  Task* t;
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
   
  //__________________________________
  //  Form the matrix
  cout_doing << "ICE::scheduleSetupMatrix" << endl;
  t = scinew Task("ICE::setupMatrix", this, &ICE::setupMatrix);
  t->requires( Task::ParentOldDW, lb->delTLabel);    
  t->requires( Task::ParentNewDW, lb->sp_vol_CCLabel,     gac,1);    
  t->requires( Task::ParentNewDW, lb->vol_frac_CCLabel,   gac,1); 
  t->requires( Task::NewDW,       lb->uvel_FCMELabel,     gac,2); 
  t->requires( Task::NewDW,       lb->vvel_FCMELabel,     gac,2); 
  t->requires( Task::NewDW,       lb->wvel_FCMELabel,     gac,2); 
  t->requires( Task::NewDW,       lb->sp_volX_FCLabel,    gac,1);
  t->requires( Task::NewDW,       lb->sp_volY_FCLabel,    gac,1);
  t->requires( Task::NewDW,       lb->sp_volZ_FCLabel,    gac,1);
  t->requires( Task::NewDW,       lb->vol_fracX_FCLabel,  gac,1);  
  t->requires( Task::NewDW,       lb->vol_fracY_FCLabel,  gac,1);  
  t->requires( Task::NewDW,       lb->vol_fracZ_FCLabel,  gac,1);  
  t->requires( Task::NewDW,       lb->betaLabel,          one_matl,  gn,0);  

  t->computes(lb->matrixLabel,  one_matl);
  sched->addTask(t, patches, all_matls);                     
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleSetupRHS--
_____________________________________________________________________*/
void ICE::scheduleSetupRHS(  SchedulerP& sched,
                             const LevelP&,
                             const PatchSet* patches,
                             const MaterialSubset* one_matl,
                             const MaterialSet* all_matls)
{
  Task* t;
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
 
  cout_doing << "ICE::scheduleSetupRHS" << endl;
  t = scinew Task("ICE::setupRHS", this, &ICE::setupRHS);
 
  const MaterialSubset* press_matl = one_matl; 
  t->requires( Task::ParentOldDW, lb->delTLabel);
  t->requires( Task::ParentNewDW, lb->press_equil_CCLabel,press_matl,gn,0);
  
/*`==========TESTING==========*/
  if(d_models.size() == 0){  //MODEL REMOVE
    t->requires( Task::ParentNewDW, lb->burnedMass_CCLabel,          gn,0);
  } else {
    t->requires( Task::ParentNewDW, lb->modelMass_srcLabel,          gn,0);
  }  
/*==========TESTING==========`*/
  t->requires( Task::ParentNewDW, lb->sp_vol_CCLabel,                gn,0);
  t->requires( Task::ParentNewDW, lb->speedSound_CCLabel,            gn,0);
  t->requires( Task::ParentNewDW, lb->vol_frac_CCLabel,              gac,2); 
  t->requires( Task::NewDW,       lb->uvel_FCMELabel,                gac,2);
  t->requires( Task::NewDW,       lb->vvel_FCMELabel,                gac,2);
  t->requires( Task::NewDW,       lb->wvel_FCMELabel,                gac,2);
  t->requires( Task::OldDW,       lb->press_CCLabel,      press_matl,gn,0);
  t->requires( Task::OldDW,       lb->imp_delPLabel,      press_matl,gn,0);

  t->computes(lb->vol_fracX_FCLabel);
  t->computes(lb->vol_fracY_FCLabel);
  t->computes(lb->vol_fracZ_FCLabel);
  t->computes(lb->term2Label,        one_matl);
  t->computes(lb->rhsLabel,          one_matl);
  t->computes(lb->initialGuessLabel, one_matl);
  t->computes(lb->betaLabel,         one_matl);
  t->computes(lb->max_RHSLabel);
  
  sched->addTask(t, patches, all_matls);                     
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleUpdatePressure--
_____________________________________________________________________*/
void ICE::scheduleUpdatePressure(  SchedulerP& sched,
                                   const LevelP&,
                                   const PatchSet* patches,
                                   const MaterialSubset* press_matl,
                                   const MaterialSet* all_matls)
{
  Task* t;
  Ghost::GhostType  gn  = Ghost::None;
   
  //__________________________________
  // update the pressure
  cout_doing << "ICE::scheduleUpdatePressure" << endl;
  t = scinew Task("ICE::updatePressure", this, &ICE::updatePressure);
  t->requires(Task::OldDW,         lb->press_CCLabel,        press_matl,  gn);
  t->requires(Task::NewDW,         lb->imp_delPLabel,        press_matl,  gn);
  t->computes(lb->press_CCLabel,      press_matl); 
  sched->addTask(t, patches, all_matls);                 
} 

/* ---------------------------------------------------------------------
 Function~  ICE::--scheduleImplicitVel_FC
_____________________________________________________________________*/
void ICE::scheduleImplicitVel_FC(SchedulerP& sched,
                                 const PatchSet* patches,                  
                                 const MaterialSubset* ice_matls,          
                                 const MaterialSubset* mpm_matls,          
                                 const MaterialSubset* press_matl,         
                                 const MaterialSet* all_matls, 
                                 const bool recursion)
{ 
  Task* t = 0;
  if (d_RateForm) {     //RATE FORM
    throw ProblemSetupException("implicit Rate Form ICE isn't working");
  }
  else if (d_EqForm) {       // EQ 
    cout_doing << "ICE::Implicit scheduleComputeVel_FC" << endl;
    t = scinew Task("ICE::scheduleImplicitVel_FC",
              this, &ICE::computeVel_FC, recursion);
  }
           
  Ghost::GhostType  gac = Ghost::AroundCells;                      
  t->requires(Task::ParentOldDW,lb->delTLabel);

  t->requires(Task::ParentNewDW,lb->sp_vol_CCLabel,    /*all_matls*/ gac,1);
  t->requires(Task::ParentNewDW,lb->rho_CCLabel,       /*all_matls*/ gac,1);
  t->requires(Task::ParentOldDW,lb->vel_CCLabel,         ice_matls,  gac,1);
  t->requires(Task::ParentNewDW,lb->vel_CCLabel,         mpm_matls,  gac,1); 
  t->requires(Task::OldDW,      lb->press_CCLabel,       press_matl, gac,1);

  t->computes(lb->uvel_FCLabel);
  t->computes(lb->vvel_FCLabel);
  t->computes(lb->wvel_FCLabel);
  sched->addTask(t, patches, all_matls);
  
  //__________________________________
  //  added exchange to 
  cout_doing << "ICE::Implicit scheduleAddExchangeContributionToFCVel" << endl;
  Task* task = scinew Task("ICE::addExchangeContributionToFCVel",
                     this, &ICE::addExchangeContributionToFCVel, recursion);

  task->requires(Task::ParentOldDW, lb->delTLabel);  
  task->requires(Task::ParentNewDW, lb->sp_vol_CCLabel,    gac,1);
  task->requires(Task::ParentNewDW, lb->vol_frac_CCLabel,  gac,1);
  task->requires(Task::NewDW,       lb->uvel_FCLabel,      gac,2);
  task->requires(Task::NewDW,       lb->vvel_FCLabel,      gac,2);
  task->requires(Task::NewDW,       lb->wvel_FCLabel,      gac,2);
  task->computes(lb->sp_volX_FCLabel);
  task->computes(lb->sp_volY_FCLabel);
  task->computes(lb->sp_volZ_FCLabel);
   
  task->computes(lb->uvel_FCMELabel);
  task->computes(lb->vvel_FCMELabel);
  task->computes(lb->wvel_FCMELabel);
  
  sched->addTask(task, patches, all_matls);
} 
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputeDel_P--
_____________________________________________________________________*/
void ICE::scheduleComputeDel_P(  SchedulerP& sched,
                                 const LevelP&,
                                 const PatchSet* patches,           
                                 const MaterialSubset* one_matl,    
                                 const MaterialSubset* press_matl,  
                                 const MaterialSet* all_matls)      
{
  Task* t;
  Ghost::GhostType  gn  = Ghost::None;
  //__________________________________
  // update the pressure
  cout_doing << "ICE::scheduleComputeDel_P" << endl;
  t = scinew Task("ICE::scheduleComputeDel_P", this, &ICE::computeDel_P);

  t->requires(Task::NewDW, lb->press_equil_CCLabel,  press_matl,  gn);
  t->requires(Task::NewDW, lb->rho_CCLabel,                       gn);
  t->requires(Task::NewDW, lb->press_CCLabel,        press_matl,  gn);      
  t->requires(Task::NewDW, lb->betaLabel,            one_matl,    gn);      
  t->requires(Task::NewDW, lb->term2Label,           one_matl,    gn);      

  t->computes(lb->delP_DilatateLabel, press_matl);
  t->computes(lb->delP_MassXLabel,    press_matl);
  t->computes(lb->sum_rho_CCLabel,    one_matl);  
  
  sched->addTask(t, patches, all_matls);                 
} 
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleImplicitPressureSolve--
_____________________________________________________________________*/
void ICE::scheduleImplicitPressureSolve(  SchedulerP& sched,
                                          const LevelP& level,
                                          const PatchSet*,
                                          const MaterialSubset* one_matl,
                                          const MaterialSubset* press_matl,
                                          const MaterialSubset* ice_matls,
                                          const MaterialSubset* mpm_matls,
                                          const MaterialSet* all_matls)
{
  cout_doing << "ICE::scheduleImplicitPressureSolve" << endl;
  
  Task* t = scinew Task("ICE::implicitPressureSolve", 
                   this, &ICE::implicitPressureSolve,
                   level, sched.get_rep(), ice_matls, mpm_matls);
 
  t->hasSubScheduler();
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
  //__________________________________
  // The computes and requires when you're looking up 
  // from implicitPressure solve
  // OldDW = ParentOldDW
  // NewDW = ParentNewDW
  //__________________________________
  // common Variables
  t->requires( Task::OldDW, lb->delTLabel);    
  t->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,2); 
  t->requires( Task::NewDW, lb->sp_vol_CCLabel,     gac,1);
  t->requires( Task::NewDW, lb->press_equil_CCLabel, press_matl,  gac,1);

  //__________________________________
  // SetupRHS
/*`==========TESTING==========*/
  if(d_models.size() == 0){  //MODEL REMOVE
    t->requires( Task::NewDW, lb->burnedMass_CCLabel, gn,0);
  } else {
    t->requires( Task::NewDW, lb->modelMass_srcLabel, gn,0);
  } 
/*==========TESTING==========`*/
  t->requires( Task::NewDW, lb->speedSound_CCLabel, gn,0);
  t->requires( Task::OldDW, lb->imp_delPLabel,   press_matl, gn,0);
   
  //__________________________________
  // Update Pressure
  t->requires( Task::NewDW, lb->rho_CCLabel,                 gac,1);            
  t->requires( Task::NewDW, lb->press_CCLabel,   press_matl, gac,1);  
  
  //__________________________________
  // ImplicitVel_FC
  t->requires(Task::OldDW,lb->vel_CCLabel,       ice_matls,  gac,1);    
  t->requires(Task::NewDW,lb->vel_CCLabel,       mpm_matls,  gac,1);   

  t->modifies(lb->press_CCLabel,      press_matl); 
  t->computes(lb->imp_delPLabel,      press_matl); 
  t->computes(lb->betaLabel,          one_matl);  
  t->computes(lb->term2Label,         one_matl);
  
  t->computes(lb->vol_fracX_FCLabel);
  t->computes(lb->vol_fracY_FCLabel);
  t->computes(lb->vol_fracZ_FCLabel);  
  
  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perproc_patches =  
                        loadBal->createPerProcessorPatchSet(level, d_myworld);
  sched->addTask(t, perproc_patches, all_matls);
}

/* --------------------------------------------------------------------- 
 Function~  ICE::setupMatrix-- 
_____________________________________________________________________*/
void ICE::setupMatrix(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* ,
                      DataWarehouse*,
                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<<"Doing setupMatrix on patch "
              << patch->getID() <<"\t\t\t\t ICE" << endl;
              
    // define parent_new/old_dw
    DataWarehouse* parent_new_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentNewDW);
    DataWarehouse* parent_old_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentOldDW); 
            
    delt_vartype delT;
    parent_old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx     = patch->dCell();
    int numMatls  = d_sharedState->getNumMatls();
    CCVariable<Stencil7> A; 
    constCCVariable<double> beta;
   
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->allocateAndPut(A,    lb->matrixLabel,  0, patch, gn, 0);
    new_dw->get(beta,            lb->betaLabel,    0, patch, gn, 0); 
    IntVector right, left, top, bottom, front, back;
    IntVector R_CC, L_CC, T_CC, B_CC, F_CC, BK_CC;

    //__________________________________
    //  Initialize beta and A
    for(CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      Stencil7&  A_tmp=A[c];
      A_tmp.p = 0.0; 
      A_tmp.n = 0.0;   A_tmp.s = 0.0;
      A_tmp.e = 0.0;   A_tmp.w = 0.0; 
      A_tmp.t = 0.0;   A_tmp.b = 0.0;
    } 
  
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      constSFCXVariable<double> uvel_FC, sp_volX_FC, vol_fracX_FC;
      constSFCYVariable<double> vvel_FC, sp_volY_FC, vol_fracY_FC;
      constSFCZVariable<double> wvel_FC, sp_volZ_FC, vol_fracZ_FC;
      constCCVariable<double> vol_frac, sp_vol_CC;     

      new_dw->get(uvel_FC,          lb->uvel_FCMELabel,     indx,patch,gac, 2);
      new_dw->get(vvel_FC,          lb->vvel_FCMELabel,     indx,patch,gac, 2);
      new_dw->get(wvel_FC,          lb->wvel_FCMELabel,     indx,patch,gac, 2);
      
      new_dw->get(sp_volX_FC,       lb->sp_volX_FCLabel,    indx,patch,gac, 1);
      new_dw->get(sp_volY_FC,       lb->sp_volY_FCLabel,    indx,patch,gac, 1);
      new_dw->get(sp_volZ_FC,       lb->sp_volZ_FCLabel,    indx,patch,gac, 1);

      new_dw->get(vol_fracX_FC,     lb->vol_fracX_FCLabel,  indx,patch,gac, 1);    
      new_dw->get(vol_fracY_FC,     lb->vol_fracY_FCLabel,  indx,patch,gac, 1);    
      new_dw->get(vol_fracZ_FC,     lb->vol_fracZ_FCLabel,  indx,patch,gac, 1);    
            
      parent_new_dw->get(vol_frac,  lb->vol_frac_CCLabel,   indx,patch,gac, 1);
      parent_new_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,     indx,patch,gac, 1);
      
      //__________________________________
      // Sum (<upwinded volfrac> * sp_vol on faces)
      // +x -x +y -y +z -z
      //  e, w, n, s, t, b
      
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) { 
        IntVector c = *iter;
        Stencil7&  A_tmp=A[c];
        right  = c + IntVector(1,0,0);      left   = c;  
        top    = c + IntVector(0,1,0);      bottom = c;  
        front  = c + IntVector(0,0,1);      back   = c;  

        //  use the upwinded vol_frac    
        A_tmp.e += vol_fracX_FC[right]  * sp_volX_FC[right];               
        A_tmp.w += vol_fracX_FC[left]   * sp_volX_FC[left];                   
        A_tmp.n += vol_fracY_FC[top]    * sp_volY_FC[top];               
        A_tmp.s += vol_fracY_FC[bottom] * sp_volY_FC[bottom];
        A_tmp.t += vol_fracZ_FC[front]  * sp_volZ_FC[front];
        A_tmp.b += vol_fracZ_FC[back]   * sp_volZ_FC[back];
        
      }

      //---- P R I N T   D A T A ------ 
      if (switchDebug_setupMatrix ) {
        ostringstream desc;
        desc << "setupMatrix_Mat_" << indx << "_patch_"<< patch->getID(); 
        printData_FC( indx, patch,1, desc.str(), "vol_fracX_FC", vol_fracX_FC);
        printData_FC( indx, patch,1, desc.str(), "vol_fracY_FC", vol_fracY_FC);
        printData_FC( indx, patch,1, desc.str(), "vol_fracZ_FC", vol_fracZ_FC);
      }    
    }  //matl loop
        
    //__________________________________
    //  Multiple stencil by delT^2 * area/dx
    double delT_2 = delT * delT;
    double tmp_e_w = delT_2/( dx.x() * dx.x() );
    double tmp_n_s = delT_2/( dx.y() * dx.y() );
    double tmp_t_b = delT_2/( dx.z() * dx.z() );
        
   for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){ 
      IntVector c = *iter;
      Stencil7&  A_tmp=A[c];
      A_tmp.e *= -tmp_e_w;
      A_tmp.w *= -tmp_e_w;
      
      A_tmp.n *= -tmp_n_s;
      A_tmp.s *= -tmp_n_s;

      A_tmp.t *= -tmp_t_b;
      A_tmp.b *= -tmp_t_b;
    }  
    //__________________________________
    //  Boundary conditons on A.e, A.w, A.n, A.s, A.t, A.b
    ImplicitMatrixBC( A, patch);   
     
    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){ 
      IntVector c = *iter;
      Stencil7&  A_tmp=A[c];
      A_tmp.p = beta[c] -
                (A_tmp.n + A_tmp.s + A_tmp.e + A_tmp.w + A_tmp.t + A_tmp.b);
    }        
    //---- P R I N T   D A T A ------   
    if (switchDebug_setupMatrix) {    
      ostringstream desc;
      desc << "BOT_setupMatrix_patch_" << patch->getID();
      printStencil( 0, patch, 1, desc.str(), "A", A);
    }         
  }
}

/* -------------------------------------------------------------------
 Function~  ICE::setupRHS-- 
_____________________________________________________________________*/
void ICE::setupRHS(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* ,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<<"Doing setupRHS on patch "
              << patch->getID() <<"\t\t\t\t ICE" << endl;
    // define parent_new/old_dw          
    DataWarehouse* parent_new_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentNewDW);
    DataWarehouse* parent_old_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentOldDW);  
           
    int numMatls  = d_sharedState->getNumMatls();
    delt_vartype delT;
    parent_old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx     = patch->dCell();
    double vol    = dx.x()*dx.y()*dx.z();
    double invvol = 1./vol;

    Advector* advector = d_advector->clone(new_dw,patch);
    CCVariable<double> q_advected, beta; 
    CCVariable<double> rhs, initialGuess;
    CCVariable<double> sumAdvection, massExchTerm;
    constCCVariable<double> press_CC, oldPressure, imp_delP, speedSound;
    
    const IntVector gc(1,1,1);
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;   

    parent_new_dw->get(oldPressure,     lb->press_equil_CCLabel,0,patch,gn,0);
    old_dw->get(press_CC,               lb->press_CCLabel,      0,patch,gn,0);
    old_dw->get(imp_delP,               lb->imp_delPLabel,      0,patch,gn,0); 
 
    new_dw->allocateAndPut(rhs,         lb->rhsLabel,           0,patch);    
    new_dw->allocateAndPut(initialGuess,lb->initialGuessLabel,  0,patch); 
    new_dw->allocateAndPut(beta,        lb->betaLabel,          0,patch);
    new_dw->allocateAndPut(massExchTerm,lb->term2Label,         0,patch);
    new_dw->allocateTemporary(q_advected,       patch);
    new_dw->allocateTemporary(sumAdvection,     patch);  
 
    rhs.initialize(0.0);
/*`==========TESTING==========*/
    //initialGuess.copyData(imp_delP);
    initialGuess.initialize(0.0);         // CAN WE DO BETTER THAN THIS?
/*===========TESTING==========`*/
    sumAdvection.initialize(0.0);
    massExchTerm.initialize(0.0);
    beta.initialize(0.0);
    double rhs_max = 0.0;
    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;
      SFCXVariable<double> vol_fracX_FC;
      SFCYVariable<double> vol_fracY_FC;
      SFCZVariable<double> vol_fracZ_FC;      
      constCCVariable<double> vol_frac, burnedMass, sp_vol_CC, speedSound;

      new_dw->allocateAndPut(vol_fracX_FC, lb->vol_fracX_FCLabel, indx,patch);
      new_dw->allocateAndPut(vol_fracY_FC, lb->vol_fracY_FCLabel, indx,patch);
      new_dw->allocateAndPut(vol_fracZ_FC, lb->vol_fracZ_FCLabel, indx,patch);
      
      // lowIndex is the same for all vel_FC
      IntVector lowIndex(patch->getSFCXLowIndex());   
      vol_fracX_FC.initialize(0.0, lowIndex,patch->getSFCXHighIndex());
      vol_fracY_FC.initialize(0.0, lowIndex,patch->getSFCYHighIndex());
      vol_fracZ_FC.initialize(0.0, lowIndex,patch->getSFCZHighIndex()); 
      
      new_dw->get(uvel_FC,           lb->uvel_FCMELabel,     indx,patch,gac, 2);
      new_dw->get(vvel_FC,           lb->vvel_FCMELabel,     indx,patch,gac, 2);
      new_dw->get(wvel_FC,           lb->wvel_FCMELabel,     indx,patch,gac, 2);
      parent_new_dw->get(vol_frac,   lb->vol_frac_CCLabel,   indx,patch,gac, 2);
/*`==========TESTING==========*/
      if(d_models.size() == 0){   //MODELS REMOVE
        parent_new_dw->get(burnedMass, lb->burnedMass_CCLabel, indx,patch,gn,0);
      }else{
        parent_new_dw->get(burnedMass, lb->modelMass_srcLabel, indx,patch,gn,0);
      } 
/*==========TESTING==========`*/
      parent_new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn,0);
      parent_new_dw->get(speedSound, lb->speedSound_CCLabel, indx,patch,gn,0);

      //---- P R I N T   D A T A ------  
      if (switchDebug_setupRHS) {
        ostringstream desc;
        desc << "Top_setupRHS_Mat_"<<indx<<"_patch_"<<patch->getID();
        printData_FC( indx, patch,1, desc.str(), "uvel_FC",    uvel_FC);
        printData_FC( indx, patch,1, desc.str(), "vvel_FC",    vvel_FC);
        printData_FC( indx, patch,1, desc.str(), "wvel_FC",    wvel_FC);
      }
        
      //__________________________________
      // Advection preprocessing
      bool bulletProof_test=false;
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,indx, 
                                    bulletProof_test); 

      advector->advectQ(vol_frac, patch, q_advected,  
                        vol_fracX_FC, vol_fracY_FC,  vol_fracZ_FC, new_dw);  
    
      //__________________________________
      //  sum Advecton (<vol_frac> vel_FC)
      //  sum mass Exchange term
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        sumAdvection[c] += q_advected[c];
        
        massExchTerm[c] += burnedMass[c] * (sp_vol_CC[c] * invvol);
      }
      
      //__________________________________
      // sum beta = sum ( vol_frac * sp_vol/speedSound^2)
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        beta[c] += vol_frac[c] * sp_vol_CC[c]/(speedSound[c] * speedSound[c]);
      }    
    }  //matl loop
    delete advector;  
      
    //__________________________________
    //  Form RHS
    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      
      double term1 = beta[c] *  (press_CC[c] - oldPressure[c]); 
      double term2 = delT * massExchTerm[c];
      
      rhs[c] = -term1 + term2 + sumAdvection[c];
      
      rhs_max = std::max(rhs_max, rhs[c] * rhs[c]);
    }
    new_dw->put(max_vartype(rhs_max), lb->max_RHSLabel);

    //---- P R I N T   D A T A ------  
    if (switchDebug_setupRHS) {
      ostringstream desc;
      desc << "BOT_setupRHS_patch_" << patch->getID();
      printData( 0, patch, 0,desc.str(), "rhs",              rhs);
      printData( 0, patch, 0, desc.str(), "beta",            beta);
      printData( 0, patch, 0,desc.str(), "sumAdvection",     sumAdvection);
  //  printData( 0, patch, 0,desc.str(), "MassExchangeTerm", massExchTerm);
  //  printData( 0, patch, 0,desc.str(), "InitialGuess",     initialGuess);
      printData( 0, patch, 0,desc.str(), "Press_CC",         press_CC);
      printData( 0, patch, 0,desc.str(), "oldPress",         oldPressure);
    }  
  }  // patches loop
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
              << patch->getID() <<"\t\t\t\t ICE" << endl;

   // define parent_new_dw
    DataWarehouse* parent_new_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentNewDW); 
            
    int numMatls  = d_sharedState->getNumMatls(); 
      
    CCVariable<double> press_CC;     
    constCCVariable<double> imp_delP;
    constCCVariable<double> pressure;
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
    
    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(pressure,             lb->press_CCLabel,    0,patch,gn,0);
    new_dw->get(imp_delP,             lb->imp_delPLabel,    0,patch,gn,0);
    new_dw->allocateAndPut(press_CC,  lb->press_CCLabel,    0,patch);  
    press_CC.initialize(0.0);
    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      parent_new_dw->get(sp_vol_CC[m],lb->sp_vol_CCLabel, indx,patch,gn,0);
    }             
    //__________________________________
    //  add delP to press
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      press_CC[c] = pressure[c] + imp_delP[c];
    }  
    setBC(press_CC, sp_vol_CC[SURROUND_MAT],
          "sp_vol", "Pressure", patch ,d_sharedState, 0, new_dw);
    
    //---- P R I N T   D A T A ------  
    if (switchDebug_updatePressure) {
      ostringstream desc;
      desc << "BOT_updatePressure_patch_" << patch->getID();
      printData( 0, patch, 1,desc.str(), "imp_delP",      imp_delP); 
      printData( 0, patch, 1,desc.str(), "Press_CC",      press_CC);
    }  
  } // patch loop
}
 
/* --------------------------------------------------------------------- 
 Function~  ICE::computeDel_P-- 
_____________________________________________________________________*/
void ICE::computeDel_P(const ProcessorGroup*,
                         const PatchSubset* patches,                    
                         const MaterialSubset* ,                        
                         DataWarehouse*,
                         DataWarehouse* new_dw)                         
{
   
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<<"Doing computeDel_P on patch "
              << patch->getID() <<"\t\t\t\t ICE" << endl;
            
    int numMatls  = d_sharedState->getNumMatls(); 
      
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> sum_rho_CC;
    constCCVariable<double> rho_CC;
    constCCVariable<double> beta;
    constCCVariable<double> massExchTerm; 
    constCCVariable<double> press_equil, press_CC;
    
    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(press_equil,          lb->press_equil_CCLabel,  0,patch,gn,0);
    new_dw->get(press_CC,             lb->press_CCLabel,        0,patch,gn,0);
    new_dw->get(beta,                 lb->betaLabel,            0,patch,gn,0);
    new_dw->get(massExchTerm,         lb->term2Label,           0,patch,gn,0);
             
    new_dw->allocateAndPut(delP_Dilatate,lb->delP_DilatateLabel,0, patch);
    new_dw->allocateAndPut(delP_MassX,   lb->delP_MassXLabel,   0, patch);
    new_dw->allocateAndPut(sum_rho_CC,   lb->sum_rho_CCLabel,   0, patch);
 
    sum_rho_CC.initialize(0.0);
    delP_Dilatate.initialize(0.0);
    delP_MassX.initialize(0.0); 
         
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->get(rho_CC,      lb->rho_CCLabel,    indx,patch,gn,0);
      //__________________________________
      //  compute sum_rho_CC used by press_FC
      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        sum_rho_CC[c] += rho_CC[c];
      } 
    }
    //__________________________________
    // backout delP_Dilatate and delP_MassX
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      delP_MassX[c]    = massExchTerm[c]/beta[c];
      delP_Dilatate[c] = (press_CC[c] - press_equil[c]) - delP_MassX[c];
    }    

    //---- P R I N T   D A T A ------  
    if (switchDebug_computeDelP) {
      ostringstream desc;
      desc << "BOT_computeDelP_patch_" << patch->getID();
      printData( 0, patch, 1,desc.str(), "delP_Dilatate", delP_Dilatate);
      printData( 0, patch, 1,desc.str(), "press_equil",   press_equil);
      printData( 0, patch, 1,desc.str(), "press_CC",      press_CC);
    //printData( 0, patch, 1,desc.str(), "delP_MassX",    delP_MassX);
    }
  } // patch loop
}
 
/* --------------------------------------------------------------------- 
 Function~  ICE::implicitPressureSolve-- 
_____________________________________________________________________*/
void ICE::implicitPressureSolve(const ProcessorGroup* pg,
		                  const PatchSubset* patch_sub, 
		                  const MaterialSubset*,       
		                  DataWarehouse* ParentOldDW,    
                                DataWarehouse* ParentNewDW,    
		                  LevelP level,                 
                                Scheduler* sched,
                                const MaterialSubset* ice_matls,
                                const MaterialSubset* mpm_matls)
{
  cout_doing<<"Doing implicitPressureSolve "<<"\t\t\t\t ICE" << endl;
  static int n_passes;                  
  n_passes ++;
  //__________________________________
  // define Matls   
  Ghost::GhostType  gn = Ghost::None;
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  MaterialSubset* one_matl    = scinew MaterialSubset();
  MaterialSet* press_matlSet  = scinew MaterialSet();
  one_matl->add(0);
  press_matlSet->add(0);
  one_matl->addReference();
  press_matlSet->addReference(); 
  MaterialSubset* press_matl = one_matl;
 
  //__________________________________
  // - create subScheduler
  // - turn scrubbing off
  // - assign Data Warehouse
  
  DataWarehouse::ScrubMode ParentOldDW_scrubmode =
                           ParentOldDW->setScrubbing(DataWarehouse::ScrubNone);
  DataWarehouse::ScrubMode ParentNewDW_scrubmode =
                           ParentNewDW->setScrubbing(DataWarehouse::ScrubNone);
                           
  SchedulerP subsched = sched->createSubScheduler();
  subsched->initialize(3, 1, ParentOldDW, ParentNewDW);
  subsched->clearMappings();
  subsched->mapDataWarehouse(Task::ParentOldDW, 0);
  subsched->mapDataWarehouse(Task::ParentNewDW, 1);
  subsched->mapDataWarehouse(Task::OldDW, 2);
  subsched->mapDataWarehouse(Task::NewDW, 3);
  
  DataWarehouse* subOldDW = subsched->get_dw(2);
  DataWarehouse* subNewDW = subsched->get_dw(3); 
     
  GridP grid = level->getGrid();
  subsched->advanceDataWarehouse(grid);
  
  const PatchSet* patch_set = level->eachPatch();
  //__________________________________
  // Create the tasks
  bool recursion = true;
  scheduleImplicitVel_FC( subsched,     patch_set,      ice_matls,
                                                        mpm_matls, 
                                                        press_matl, 
                                                        all_matls,
                                                        recursion);
                                                        
  scheduleSetupRHS(       subsched, level,  patch_set,  one_matl, 
                                                        all_matls);
                                                        
  scheduleSetupMatrix(    subsched, level,  patch_set,  one_matl, 
                                                        all_matls);
  
  solver->scheduleSolve(level, subsched, press_matlSet,
                        lb->matrixLabel, lb->imp_delPLabel, false,
                        lb->rhsLabel,    lb->initialGuessLabel,
                        Task::NewDW,     solver_parameters);
                        
  scheduleUpdatePressure( subsched,  level, patch_set,  press_matl,  
                                                        all_matls);

  subsched->compile(d_myworld);      
                                                      
  //__________________________________
  //  Move data from parentOldDW to subSchedNewDW.
   delt_vartype dt;
   subNewDW = subsched->get_dw(3);
   ParentOldDW->get(dt, d_sharedState->get_delt_label());
   subNewDW->put(dt, d_sharedState->get_delt_label()); 
   subNewDW->transferFrom(ParentNewDW,lb->press_CCLabel,patch_sub, press_matl); 
   subNewDW->transferFrom(ParentOldDW,lb->imp_delPLabel,patch_sub, press_matl); 
  //__________________________________
  //  Iteration Loop
  int counter = 0;
  max_vartype max_RHS = 1/d_SMALL_NUM;
  double max_RHS_old = max_RHS;
  bool restart = false;
  
  while( counter < d_max_iter_implicit && max_RHS > d_outer_iter_tolerance) {
    subsched->advanceDataWarehouse(grid);   // move subscheduler forward
    
    subOldDW = subsched->get_dw(2);
    subNewDW = subsched->get_dw(3);
    subOldDW->setScrubbing(DataWarehouse::ScrubComplete);
    subNewDW->setScrubbing(DataWarehouse::ScrubNone);
    subsched->execute(d_myworld);
    subNewDW->get(max_RHS,   lb->max_RHSLabel);
    counter ++;
    
    // restart timestep if working too hard
    if (counter > d_iters_before_timestep_restart ){
      restart = true;
      cout << "\nWARNING: max iterations befor timestep restart reached\n"<<endl;
    }
    if (subsched->get_dw(3)->timestepRestarted() ) {
      cout << "\nWARNING: Solver had requested a restart\n" <<endl;
      restart = true;
    }
    if (((max_RHS - max_RHS_old) > 1000.0 * max_RHS_old) ){
      cout << "\nWARNING: outer interation is diverging now "
           << "restarting the timestep\n"<< endl;
      restart = true;
    }
    if(restart){
      ParentNewDW->abortTimestep();
      ParentNewDW->restartTimestep();
      return;
    }
    
    if(pg->myrank() == 0) {
      cout << "Outer iteration " << counter<< " max_rhs "<< max_RHS<< endl;
      max_RHS_old = max_RHS;
    }
  }
  //__________________________________
  //  BULLET PROOFING
  if ( (counter == d_max_iter_implicit)   && 
       (max_RHS > d_outer_iter_tolerance) &&
       counter > 1) {
    ostringstream s;
    s <<"ERROR ICE::implicitPressureSolve, the maximum number of outer"
      <<" iterations was reached. \n " 
      << "Try either increasing the max_outer_iterations "
      <<" or decrease outer_iteration_tolerance\n ";
    throw ConvergenceFailure(s.str(),counter,max_RHS,d_outer_iter_tolerance);
  }

  //__________________________________
  // Move products of iteration (only) from sub_new_dw -> parent_new_dw 
  subNewDW  = subsched->get_dw(3);
  const MaterialSubset* all_mats = all_matls->getUnion();
  ParentNewDW->transferFrom(subNewDW, lb->imp_delPLabel,  patch_sub,one_matl); 
  ParentNewDW->transferFrom(subNewDW, lb->betaLabel,      patch_sub,one_matl);          
  ParentNewDW->transferFrom(subNewDW, lb->term2Label,     patch_sub,one_matl);
  
  ParentNewDW->transferFrom(subNewDW, lb->vol_fracX_FCLabel,patch_sub,all_mats);     
  ParentNewDW->transferFrom(subNewDW, lb->vol_fracY_FCLabel,patch_sub,all_mats);     
  ParentNewDW->transferFrom(subNewDW, lb->vol_fracZ_FCLabel,patch_sub,all_mats);
  
 // for modified variables you need to do it manually
  constCCVariable<double> press_CC;
  CCVariable<double>      press_new;

  for (int p = 0; p < patch_sub->size();p++) {
    const Patch* patch = patch_sub->get(p);
    subNewDW->get(press_CC,               lb->press_CCLabel,0,patch,gn,0);
    ParentNewDW->getModifiable(press_new, lb->press_CCLabel,0,patch);
    press_new.copyData(press_CC);
  }
  //__________________________________
  //  Turn scrubbing back on
  ParentOldDW->setScrubbing(ParentOldDW_scrubmode);
  ParentNewDW->setScrubbing(ParentNewDW_scrubmode);
}
