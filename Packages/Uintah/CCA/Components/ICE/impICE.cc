#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h> 
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/InternalError.h>
#include <cmath>
using namespace SCIRun;
using namespace Uintah;

static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false);

#define OREN 1

/*___________________________________________________________________
 Function~  ICE::scheduleSetupMatrix--
_____________________________________________________________________*/
void ICE::scheduleSetupMatrix(  SchedulerP& sched,
                               const LevelP&,
                               const PatchSet* patches,
                               const MaterialSubset* one_matl,
                               const MaterialSet* all_matls,
                               const bool firstIteration)
{
  Task* t;
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  int levelIndex = getLevel(patches)->getIndex();
  
  Task::WhichDW whichDW;
  if(firstIteration) {
    whichDW  = Task::ParentNewDW;
  } else {
    whichDW  = Task::OldDW;
  }   
  //__________________________________
  //  Form the matrix
  cout_doing << d_myworld->myrank()<< " ICE::scheduleSetupMatrix" 
            << "\t\t\t\t\tL-" << levelIndex<< endl;
  t = scinew Task("ICE::setupMatrix", this, 
                  &ICE::setupMatrix, firstIteration);
//  t->requires( Task::ParentOldDW, lb->delTLabel);  for AMR
  t->requires( whichDW,   lb->sp_volX_FCLabel,    gac,1);        
  t->requires( whichDW,   lb->sp_volY_FCLabel,    gac,1);        
  t->requires( whichDW,   lb->sp_volZ_FCLabel,    gac,1);        
  t->requires( whichDW,   lb->vol_fracX_FCLabel,  gac,1);        
  t->requires( whichDW,   lb->vol_fracY_FCLabel,  gac,1);        
  t->requires( whichDW,   lb->vol_fracZ_FCLabel,  gac,1);        
  t->requires( Task::ParentNewDW,   
                          lb->sumKappaLabel, one_matl,oims,gn,0);      

  t->computes(lb->matrixLabel,  one_matl, oims);
  sched->addTask(t, patches, all_matls);                     
}

/*___________________________________________________________________
 Function~  ICE::scheduleSetupRHS--
_____________________________________________________________________*/
void ICE::scheduleSetupRHS(  SchedulerP& sched,
                             const PatchSet* patches,
                             const MaterialSubset* one_matl,
                             const MaterialSet* all_matls,
                             const bool insideOuterIterLoop,
                             const string& computes_or_modifies)
{
  Task* t;
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  int levelIndex = getLevel(patches)->getIndex();
 
  cout_doing << d_myworld->myrank()<< " ICE::scheduleSetupRHS" 
             << "\t\t\t\t\t\tL-" << levelIndex<< endl;
  t = scinew Task("ICE::setupRHS", this, 
                  &ICE::setupRHS, insideOuterIterLoop,computes_or_modifies);
 
  Task::WhichDW pNewDW;
  Task::WhichDW pOldDW;
  if(insideOuterIterLoop) {
    pNewDW  = Task::ParentNewDW;
    pOldDW  = Task::ParentOldDW; 
  } else {
    pNewDW  = Task::NewDW;
    pOldDW  = Task::OldDW;
  }
 
  const MaterialSubset* press_matl = one_matl; 
//  t->requires( pOldDW, lb->delTLabel);     AMR
  
  if(d_models.size() > 0){  
    t->requires(pNewDW, lb->modelMass_srcLabel,           gn,0);
  } 
  t->requires( pNewDW,      lb->press_equil_CCLabel,press_matl,oims,gn,0); 
  t->requires( pNewDW,      lb->sumKappaLabel,      press_matl,oims,gn,0);
  t->requires( pNewDW,      lb->sp_vol_CCLabel,                gn,0);
  t->requires( pNewDW,      lb->speedSound_CCLabel,            gn,0);
  t->requires( pNewDW,      lb->vol_frac_CCLabel,              gac,2);
  t->requires( Task::NewDW, lb->uvel_FCMELabel,                gac,2);          
  t->requires( Task::NewDW, lb->vvel_FCMELabel,                gac,2);          
  t->requires( Task::NewDW, lb->wvel_FCMELabel,                gac,2);          
  t->requires( Task::NewDW, lb->press_CCLabel,      press_matl,oims,gn,0);
        
  t->computes(lb->vol_fracX_FCLabel);
  t->computes(lb->vol_fracY_FCLabel);
  t->computes(lb->vol_fracZ_FCLabel);
  
  t->computes(lb->term2Label,        one_matl,oims);
  t->computes(lb->max_RHSLabel);
  
  if(computes_or_modifies =="computes"){
    t->computes(lb->rhsLabel,          one_matl,oims);
  }
  if(computes_or_modifies =="modifies"){
    t->modifies(lb->rhsLabel,          one_matl,oims);
  }
  
  sched->addTask(t, patches, all_matls);                     
}

/*___________________________________________________________________
 Function~  ICE::scheduleUpdatePressure--
_____________________________________________________________________*/
void ICE::scheduleUpdatePressure(  SchedulerP& sched,
                                   const LevelP& level,
                                   const PatchSet* patches,
                                   const MaterialSubset* ice_matls,          
                                   const MaterialSubset* /*mpm_matls*/,
                                   const MaterialSubset* press_matl,
                                   const MaterialSet* all_matls)
{
  Task* t;
  Ghost::GhostType  gn  = Ghost::None;
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet. 
  //__________________________________
  // update the pressure
  cout_doing << d_myworld->myrank()<<" ICE::scheduleUpdatePressure" 
             << "\t\t\t\t\tL-" <<level->getIndex()<<endl;
  t = scinew Task("ICE::updatePressure", this, &ICE::updatePressure);
  t->requires(Task::OldDW,  lb->press_CCLabel,   press_matl,oims,gn);       
  t->requires(Task::NewDW,  lb->imp_delPLabel,   press_matl,oims,gn);       
 
  computesRequires_CustomBCs(t, "imp_update_press_CC", lb, ice_matls,
                              d_customBC_var_basket);
  
  t->computes(lb->press_CCLabel,      press_matl,oims); 
  sched->addTask(t, patches, all_matls);                 
} 

/*___________________________________________________________________
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
  int levelIndex = getLevel(patches)->getIndex();

  cout_doing << d_myworld->myrank()<< " ICE::Implicit scheduleComputeVel_FC" 
             << "\t\t\t\tL-"<< levelIndex<<endl;
  t = scinew Task("ICE::scheduleImplicitVel_FC",
            this, &ICE::computeVel_FC, recursion);
           
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.               
//  t->requires(Task::ParentOldDW,lb->delTLabel);  AMR

  t->requires(Task::ParentNewDW,lb->sp_vol_CCLabel,    /*all_matls*/ gac,1);
  t->requires(Task::ParentNewDW,lb->rho_CCLabel,       /*all_matls*/ gac,1);
  t->requires(Task::ParentOldDW,lb->vel_CCLabel,         ice_matls,  gac,1);
  t->requires(Task::ParentNewDW,lb->vel_CCLabel,         mpm_matls,  gac,1); 
  t->requires(Task::NewDW,      lb->press_CCLabel,       press_matl, oims, gac,1);

  t->computes(lb->uvel_FCLabel);
  t->computes(lb->vvel_FCLabel);
  t->computes(lb->wvel_FCLabel);
  sched->addTask(t, patches, all_matls);
  
  //__________________________________
  //  added exchange to 
  cout_doing << d_myworld->myrank()<< " ICE::Implicit scheduleAddExchangeContributionToFCVel" 
             << "\t\tL-"<< levelIndex<<endl;
  Task* task = scinew Task("ICE::addExchangeContributionToFCVel",
                     this, &ICE::addExchangeContributionToFCVel, recursion);

//  task->requires(Task::ParentOldDW, lb->delTLabel);   AMR
  task->requires(Task::ParentNewDW, lb->sp_vol_CCLabel,    gac,1);
  task->requires(Task::ParentNewDW, lb->vol_frac_CCLabel,  gac,1);
  task->requires(Task::NewDW,       lb->uvel_FCLabel,      gac,2);
  task->requires(Task::NewDW,       lb->vvel_FCLabel,      gac,2);
  task->requires(Task::NewDW,       lb->wvel_FCLabel,      gac,2);
  
  computesRequires_CustomBCs(task, "imp_velFC_Exchange", lb, ice_matls,
                              d_customBC_var_basket);
  
  task->computes(lb->sp_volX_FCLabel);
  task->computes(lb->sp_volY_FCLabel);
  task->computes(lb->sp_volZ_FCLabel);
   
  task->computes(lb->uvel_FCMELabel);
  task->computes(lb->vvel_FCMELabel);
  task->computes(lb->wvel_FCMELabel);
  
  sched->addTask(task, patches, all_matls);
} 
/*___________________________________________________________________
 Function~  ICE::scheduleComputeDel_P--
 Note:      This task is scheduled outside the iteration loop
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
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  int levelIndex = getLevel(patches)->getIndex();
  //__________________________________
  // update the pressure
  cout_doing << d_myworld->myrank()<< " ICE::scheduleComputeDel_P" 
             << "\t\t\t\t\tL-"<< levelIndex<<endl;
  t = scinew Task("ICE::scheduleComputeDel_P", this, &ICE::computeDel_P);

  t->requires(Task::NewDW, lb->press_equil_CCLabel,  press_matl, oims, gn);
  t->requires(Task::NewDW, lb->press_CCLabel,        press_matl, oims, gn);      
  t->requires(Task::NewDW, lb->sumKappaLabel,        one_matl,   oims, gn);      
  t->requires(Task::NewDW, lb->term2Label,           one_matl,   oims, gn);
  t->requires(Task::NewDW, lb->rho_CCLabel,                       gn);    

  t->computes(lb->delP_DilatateLabel, press_matl,oims);
  t->computes(lb->delP_MassXLabel,    press_matl,oims);
  t->computes(lb->sum_rho_CCLabel,    one_matl,  oims);
  t->computes(lb->initialGuessLabel,  one_matl,  oims);  
  
  sched->addTask(t, patches, all_matls);                 
} 
/*___________________________________________________________________
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
  cout_doing << d_myworld->myrank()
              <<" ICE::scheduleImplicitPressureSolve" << endl;
  
  Task* t = scinew Task("ICE::implicitPressureSolve", 
                   this, &ICE::implicitPressureSolve,
                   level, sched.get_rep(), ice_matls, mpm_matls);
 
  t->hasSubScheduler();
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  //__________________________________
  // The computes and requires when you're looking up 
  // from implicitPressure solve
  // OldDW = ParentOldDW
  // NewDW = ParentNewDW
  //__________________________________
  // common Variables
//  t->requires( Task::OldDW, lb->delTLabel);    AMR
  t->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,2); 
  t->requires( Task::NewDW, lb->sp_vol_CCLabel,     gac,1);
  t->requires( Task::NewDW, lb->press_equil_CCLabel, press_matl, oims,gac,1);
  t->requires( Task::NewDW, lb->rhsLabel,            one_matl,   oims,gn,0);
  //t->requires( Task::OldDW, lb->initialGuessLabel, one_matl,   oims,gn,0);
  //__________________________________
  // SetupRHS
  if(d_models.size() > 0){  
    t->requires(Task::NewDW,lb->modelMass_srcLabel, gn,0);
  } 
  t->requires( Task::NewDW, lb->speedSound_CCLabel, gn,0);
  t->requires( Task::NewDW, lb->max_RHSLabel);
  
  //__________________________________
  // setup Matrix
  t->requires( Task::NewDW, lb->sp_volX_FCLabel,    gac,1);            
  t->requires( Task::NewDW, lb->sp_volY_FCLabel,    gac,1);            
  t->requires( Task::NewDW, lb->sp_volZ_FCLabel,    gac,1);            
  t->requires( Task::NewDW, lb->vol_fracX_FCLabel,  gac,1);            
  t->requires( Task::NewDW, lb->vol_fracY_FCLabel,  gac,1);            
  t->requires( Task::NewDW, lb->vol_fracZ_FCLabel,  gac,1);            
  t->requires( Task::NewDW, lb->sumKappaLabel,  press_matl,oims,gn,0);         
   
  //__________________________________
  // Update Pressure
  t->requires( Task::NewDW, lb->rho_CCLabel,                       gac,1);            
  t->requires( Task::NewDW, lb->press_CCLabel,   press_matl, oims, gac,1);  
  
  computesRequires_CustomBCs(t, "implicitPressureSolve", lb, ice_matls,
                             d_customBC_var_basket);

  //__________________________________
  // ImplicitVel_FC
  t->requires(Task::OldDW,lb->vel_CCLabel,       ice_matls,  gac,1);    
  t->requires(Task::NewDW,lb->vel_CCLabel,       mpm_matls,  gac,1);

  //__________________________________
  //  what's produced from this task
  t->modifies(lb->press_CCLabel, press_matl,oims);  
  t->modifies(lb->term2Label,    one_matl,  oims);   

  t->modifies(lb->uvel_FCMELabel);
  t->modifies(lb->vvel_FCMELabel);
  t->modifies(lb->wvel_FCMELabel);
  
  t->modifies(lb->vol_fracX_FCLabel);
  t->modifies(lb->vol_fracY_FCLabel);
  t->modifies(lb->vol_fracZ_FCLabel);  
  
  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perproc_patches =  
                        loadBal->createPerProcessorPatchSet(level);
  sched->addTask(t, perproc_patches, all_matls);
}

/*___________________________________________________________________
 Function~  ICE::setupMatrix-- 
_____________________________________________________________________*/
void ICE::setupMatrix(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* ,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const bool firstIteration)
{
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<< d_myworld->myrank()<<" Doing setupMatrix on patch "
              << patch->getID() <<"\t\t\t\t ICE \tL-" 
              << level->getIndex()<<endl;
              
    // define parent_new/old_dw
    DataWarehouse* whichDW;
    if(firstIteration) {
      whichDW  = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
    } else {
      whichDW  = old_dw; // inside the outer iter use old_dw
    }
    DataWarehouse* parent_old_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentOldDW); 
    DataWarehouse* parent_new_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentNewDW);
            
    delt_vartype delT;
    parent_old_dw->get(delT, d_sharedState->get_delt_label(),level);
    Vector dx     = patch->dCell();
    int numMatls  = d_sharedState->getNumMatls();
    CCVariable<Stencil7> A; 
    constCCVariable<double> sumKappa;
   
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->allocateAndPut(A,    lb->matrixLabel,  0, patch, gn, 0);
    parent_new_dw->get(sumKappa, lb->sumKappaLabel,0, patch, gn, 0); 
    
    IntVector right, left, top, bottom, front, back;
    IntVector R_CC, L_CC, T_CC, B_CC, F_CC, BK_CC;

    //__________________________________
    //  Initialize A
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
      constSFCXVariable<double> sp_volX_FC, vol_fracX_FC;
      constSFCYVariable<double> sp_volY_FC, vol_fracY_FC;
      constSFCZVariable<double> sp_volZ_FC, vol_fracZ_FC;
      
      whichDW->get(sp_volX_FC,   lb->sp_volX_FCLabel,    indx,patch,gac, 1);     
      whichDW->get(sp_volY_FC,   lb->sp_volY_FCLabel,    indx,patch,gac, 1);     
      whichDW->get(sp_volZ_FC,   lb->sp_volZ_FCLabel,    indx,patch,gac, 1);     

      whichDW->get(vol_fracX_FC, lb->vol_fracX_FCLabel,  indx,patch,gac, 1);        
      whichDW->get(vol_fracY_FC, lb->vol_fracY_FCLabel,  indx,patch,gac, 1);        
      whichDW->get(vol_fracZ_FC, lb->vol_fracZ_FCLabel,  indx,patch,gac, 1);        
            
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
#if OREN
    // Oren: scale matrix to finite volume formulation, otherwise
    // it is hard to properly define the AMR analogue of the
    // implicit pressure solver system.
    double vol     = dx.x()*dx.y()*dx.z();
    double tmp_e_w = vol * delT_2/( dx.x() * dx.x() );
    double tmp_n_s = vol * delT_2/( dx.y() * dx.y() );
    double tmp_t_b = vol * delT_2/( dx.z() * dx.z() );
#else
    double tmp_e_w = delT_2/( dx.x() * dx.x() );
    double tmp_n_s = delT_2/( dx.y() * dx.y() );
    double tmp_t_b = delT_2/( dx.z() * dx.z() );
#endif
        
   for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){ 
      IntVector c = *iter;
      Stencil7&  A_tmp=A[c];
      A_tmp.e *= -tmp_e_w;
      A_tmp.w *= -tmp_e_w;
      
      A_tmp.n *= -tmp_n_s;
      A_tmp.s *= -tmp_n_s;

      A_tmp.t *= -tmp_t_b;
      A_tmp.b *= -tmp_t_b;

#if OREN
      // Remember to scale the identity term (sumKappa) by vol
      // as well, to match the off diagonal entries' scaling.
      A_tmp.p = vol * sumKappa[c] -
          (A_tmp.n + A_tmp.s + A_tmp.e + A_tmp.w + A_tmp.t + A_tmp.b);
#else      
      A_tmp.p = sumKappa[c] -
          (A_tmp.n + A_tmp.s + A_tmp.e + A_tmp.w + A_tmp.t + A_tmp.b);
#endif
    }  
    //__________________________________
    //  Boundary conditons on A.e, A.w, A.n, A.s, A.t, A.b
    ImplicitMatrixBC( A, patch);   

    //---- P R I N T   D A T A ------   
    if (switchDebug_setupMatrix) {    
      ostringstream desc;
      desc << "BOT_setupMatrix_patch_" << patch->getID();
      printStencil( 0, patch, 1, desc.str(), "A", A);
    }         
  }
}

/*___________________________________________________________________
 Function~  ICE::setupRHS-- 
_____________________________________________________________________*/
void ICE::setupRHS(const ProcessorGroup*,
                   const PatchSubset* patches,
                   const MaterialSubset* ,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   const bool insideOuterIterLoop,
                   const string computes_or_modifies)
{
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<< d_myworld->myrank()<<" Doing setupRHS on patch "
              << patch->getID() <<"\t\t\t\t ICE \tL-"
              << level->getIndex()<<endl;
    // define parent_new/old_dw 
    
    DataWarehouse* pNewDW;
    DataWarehouse* pOldDW;
    if(insideOuterIterLoop) {
      pNewDW  = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      pOldDW  = new_dw->getOtherDataWarehouse(Task::ParentOldDW); 
    } else {
      pNewDW  = new_dw;
      pOldDW  = old_dw;
    }
           
    int numMatls  = d_sharedState->getNumMatls();
    delt_vartype delT;
    pOldDW->get(delT, d_sharedState->get_delt_label(), level);
    
    Vector dx     = patch->dCell();
    double vol    = dx.x()*dx.y()*dx.z();
    double invvol = 1./vol;

    Advector* advector = d_advector->clone(new_dw,patch);
    CCVariable<double> q_advected, rhs;
    CCVariable<double> sumAdvection, massExchTerm;
    constCCVariable<double> press_CC, oldPressure, speedSound, sumKappa;
    
    const IntVector gc(1,1,1);
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;   

    pNewDW->get(oldPressure,            lb->press_equil_CCLabel,0,patch,gn,0);
    pNewDW->get(sumKappa,               lb->sumKappaLabel,      0,patch,gn,0);
    new_dw->get(press_CC,               lb->press_CCLabel,      0,patch,gn,0);
    
    
    new_dw->allocateAndPut(massExchTerm,lb->term2Label,         0,patch);
    new_dw->allocateTemporary(q_advected,       patch);
    new_dw->allocateTemporary(sumAdvection,     patch);  
 
    if(computes_or_modifies == "computes"){
      new_dw->allocateAndPut(rhs, lb->rhsLabel, 0,patch);
    }
    if(computes_or_modifies == "modifies"){
      new_dw->getModifiable(rhs, lb->rhsLabel, 0,patch);
    }
    
    rhs.initialize(0.0);
    sumAdvection.initialize(0.0);
    massExchTerm.initialize(0.0);
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

      new_dw->allocateAndPut(vol_fracX_FC, lb->vol_fracX_FCLabel,  indx,patch);
      new_dw->allocateAndPut(vol_fracY_FC, lb->vol_fracY_FCLabel,  indx,patch);
      new_dw->allocateAndPut(vol_fracZ_FC, lb->vol_fracZ_FCLabel,  indx,patch);
      
      // lowIndex is the same for all vel_FC
      IntVector lowIndex(patch->getSFCXLowIndex());
      double nan= getNan();
      vol_fracX_FC.initialize(nan, lowIndex,patch->getSFCXHighIndex());
      vol_fracY_FC.initialize(nan, lowIndex,patch->getSFCYHighIndex());
      vol_fracZ_FC.initialize(nan, lowIndex,patch->getSFCZHighIndex()); 
      
      new_dw->get(uvel_FC,    lb->uvel_FCMELabel,     indx,patch,gac, 2);       
      new_dw->get(vvel_FC,    lb->vvel_FCMELabel,     indx,patch,gac, 2);       
      new_dw->get(wvel_FC,    lb->wvel_FCMELabel,     indx,patch,gac, 2);       
      pNewDW->get(vol_frac,   lb->vol_frac_CCLabel,   indx,patch,gac, 2);
      pNewDW->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn,0);
      pNewDW->get(speedSound, lb->speedSound_CCLabel, indx,patch,gn,0);

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
                                    bulletProof_test, pNewDW); 

      advector->advectQ(vol_frac, patch, q_advected,  
                        vol_fracX_FC, vol_fracY_FC,  vol_fracZ_FC, new_dw);  
    
      //__________________________________
      //  sum Advecton (<vol_frac> vel_FC)
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        sumAdvection[c] += q_advected[c];

      }
      //__________________________________
      //  sum mass Exchange term
      if(d_models.size() > 0){
        pNewDW->get(burnedMass,lb->modelMass_srcLabel,indx,patch,gn,0);
        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
          IntVector c = *iter;
          massExchTerm[c] += burnedMass[c] * (sp_vol_CC[c] * invvol);
        }
      }     
    }  //matl loop
    delete advector;  
      
    //__________________________________
    //  Form RHS
    // note:  massExchangeTerm has delT incorporated inside of it
    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      
      double term1 = sumKappa[c] *  (press_CC[c] - oldPressure[c]); 
      rhs[c] = -term1 + massExchTerm[c] + sumAdvection[c];
#if OREN
      // Oren: Max RHS should be max(abs(rhs)) over all cells,
      // not max(rhs^2). See implicit ICE document, p. 1.
      rhs_max = std::max(rhs_max, std::abs(rhs[c]));
      // Oren: scale RHS to finite volume formulation, otherwise
      // it is hard to properly define the AMR analogue of the
      // implicit pressure solver system. Notice that the max
      // RHS is not scaled so that it is scale-invariant and
      // can be compared against a scale-invariant tolerance
      // in the outer iteration.
      rhs[c] *= vol;
#else
      rhs_max = std::max(rhs_max, rhs[c] * rhs[c]); 
#endif
    }
    new_dw->put(max_vartype(rhs_max), lb->max_RHSLabel);

    //---- P R I N T   D A T A ------  
    if (switchDebug_setupRHS) {
      ostringstream desc;
      desc << "BOT_setupRHS_patch_" << patch->getID();
      printData( 0, patch, 0,desc.str(), "rhs",              rhs);
      printData( 0, patch, 0,desc.str(), "sumAdvection",     sumAdvection);
  //  printData( 0, patch, 0,desc.str(), "MassExchangeTerm", massExchTerm);
      printData( 0, patch, 0,desc.str(), "Press_CC",         press_CC);
      printData( 0, patch, 0,desc.str(), "oldPress",         oldPressure);
    }  
  }  // patches loop
}

/*___________________________________________________________________
 Function~  ICE::updatePressure-- 
_____________________________________________________________________*/
void ICE::updatePressure(const ProcessorGroup*,
                         const PatchSubset* patches,                    
                         const MaterialSubset* ,                        
                         DataWarehouse* old_dw,                         
                         DataWarehouse* new_dw)                         
{
  const Level* level = getLevel(patches);
 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<< d_myworld->myrank()<<" Doing updatePressure on patch "
              << patch->getID() <<"\t\t\t ICE \tL-" 
              << level->getIndex()<<endl;

   // define parent_dw
    DataWarehouse* parent_new_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentNewDW); 
    DataWarehouse* parent_old_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentOldDW);
                    
    int numMatls  = d_sharedState->getNumMatls(); 
      
    CCVariable<double> press_CC;     
    constCCVariable<double> imp_delP;
    constCCVariable<double> pressure;
    StaticArray<CCVariable<double> > placeHolder(0);
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
    
    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(pressure,             lb->press_CCLabel,    0,patch,gn,0);
    new_dw->get(imp_delP,             lb->imp_delPLabel,    0,patch,gn,0);
    new_dw->allocateAndPut(press_CC,  lb->press_CCLabel,    0,patch);  
    press_CC.initialize(d_EVIL_NUM);
    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      parent_new_dw->get(sp_vol_CC[m],lb->sp_vol_CCLabel, indx,patch,gn,0);
    }             
    //__________________________________
    //  add delP to press
    //  AMR:  hit the extra cells, BC aren't set an you need a valid pressure
    // imp_delP is ill-defined in teh extraCells
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      press_CC[c] = pressure[c];
    }    
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      press_CC[c] += imp_delP[c];
    }  
    //____ C L A M P ________________
    // This was done to help robustify the equilibration
    // pressure calculation in MPMICE.  Also, in rate form, negative
    // mean pressures are allowed.
    if(d_EqForm){
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
        IntVector c = *iter;
        press_CC[c] = max(1.0e-12, press_CC[c]); 
      } 
    }

    //__________________________________
    //  set boundary conditions   
    preprocess_CustomBCs("update_press_CC",parent_old_dw,parent_new_dw, 
                            lb,  patch, 999,d_customBC_var_basket);

    setBC(press_CC, placeHolder, sp_vol_CC, d_surroundingMatl_indx,
          "sp_vol", "Pressure", patch ,d_sharedState, 0, new_dw, 
           d_customBC_var_basket);
           
    delete_CustomBCs(d_customBC_var_basket);      
        
    //---- P R I N T   D A T A ------  
    if (switchDebug_updatePressure) {
      ostringstream desc;
      desc << "BOT_updatePressure_patch_" << patch->getID();
      printData( 0, patch, 1,desc.str(), "imp_delP",      imp_delP); 
      printData( 0, patch, 1,desc.str(), "Press_CC",      press_CC);
    }
    //____ B U L L E T   P R O O F I N G----
    IntVector neg_cell;
    if(!areAllValuesPositive(press_CC, neg_cell)) {
      ostringstream warn;
      warn <<"ERROR ICE::updatePressure cell "
           << neg_cell << " negative pressure\n ";        
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
  } // patch loop
}
 
/*___________________________________________________________________ 
 Function~  ICE::computeDel_P-- 
_____________________________________________________________________*/
void ICE::computeDel_P(const ProcessorGroup*,
                         const PatchSubset* patches,                    
                         const MaterialSubset* ,                        
                         DataWarehouse*,
                         DataWarehouse* new_dw)                         
{
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<< d_myworld->myrank()<<" Doing computeDel_P on patch "
              << patch->getID() <<"\t\t\t\t ICE \tL-" 
              << level->getIndex()<<endl;
            
    int numMatls  = d_sharedState->getNumMatls(); 
      
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> sum_rho_CC, initialGuess;
    constCCVariable<double> rho_CC;
    constCCVariable<double> sumKappa;
    constCCVariable<double> massExchTerm; 
    constCCVariable<double> press_equil, press_CC;
    
    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(press_equil,          lb->press_equil_CCLabel,  0,patch,gn,0);
    new_dw->get(press_CC,             lb->press_CCLabel,        0,patch,gn,0);
    new_dw->get(sumKappa,             lb->sumKappaLabel,        0,patch,gn,0);
    new_dw->get(massExchTerm,         lb->term2Label,           0,patch,gn,0);
             
    new_dw->allocateAndPut(delP_Dilatate,lb->delP_DilatateLabel,0, patch);
    new_dw->allocateAndPut(delP_MassX,   lb->delP_MassXLabel,   0, patch);
    new_dw->allocateAndPut(sum_rho_CC,   lb->sum_rho_CCLabel,   0, patch);
    new_dw->allocateAndPut(initialGuess, lb->initialGuessLabel, 0, patch);
    
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
      delP_MassX[c]    = massExchTerm[c]/sumKappa[c];
      delP_Dilatate[c] = (press_CC[c] - delP_MassX[c]) - press_equil[c];
      initialGuess[c]  = delP_Dilatate[c];
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
 
/*___________________________________________________________________ 
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
  cout_doing<<"Doing implicitPressureSolve "<<"\t\t\t\t ICE \tL-" 
            << level->getIndex()<< endl;

  //__________________________________
  // define Matl sets and subsets
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  MaterialSubset* one_matl    = d_press_matl;
  MaterialSet* press_matlSet  = scinew MaterialSet();
  press_matlSet->add(0);
  press_matlSet->addReference(); 
  
  //__________________________________
  //  turn off parentDW scrubbing
  DataWarehouse::ScrubMode ParentOldDW_scrubmode =
                           ParentOldDW->setScrubbing(DataWarehouse::ScrubNone);
  DataWarehouse::ScrubMode ParentNewDW_scrubmode =
                           ParentNewDW->setScrubbing(DataWarehouse::ScrubNone);
  //__________________________________
  // create a new subscheduler
  SchedulerP subsched = sched->createSubScheduler();
  subsched->setRestartable(true); 
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
  //  Move data from parentOldDW to subSchedNewDW.
  delt_vartype dt;
  subNewDW = subsched->get_dw(3);
  ParentOldDW->get(dt, d_sharedState->get_delt_label());
  subNewDW->put(dt, d_sharedState->get_delt_label());
   
  max_vartype max_RHS_old;
  ParentNewDW->get(max_RHS_old, lb->max_RHSLabel);
  subNewDW->put(   max_RHS_old, lb->max_RHSLabel);
  
  subNewDW->transferFrom(ParentNewDW,lb->press_CCLabel,     patch_sub, d_press_matl); 
  subNewDW->transferFrom(ParentNewDW,lb->rhsLabel,          patch_sub, one_matl);
  //subNewDW->transferFrom(ParentOldDW,lb->initialGuessLabel, patch_sub, one_matl);  
  //__________________________________
  //  Iteration Loop
  max_vartype max_RHS = 1/d_SMALL_NUM;
  double smallest_max_RHS_sofar = max_RHS; 
  int counter = 0;
  bool restart   = false;
  bool recursion = true;
  bool firstIter = true;
  //const VarLabel* whichInitialGuess = lb->initialGuessLabel;
  const VarLabel* whichInitialGuess = NULL;
  
  while( counter < d_max_iter_implicit && max_RHS > d_outer_iter_tolerance) {
  
    //__________________________________
    // schedule the tasks
    subsched->initialize(3, 1, ParentOldDW, ParentNewDW);
   


    scheduleSetupMatrix(    subsched, level,  patch_set,  one_matl, 
                                                          all_matls,
                                                          firstIter);

    solver->scheduleSolve(level, subsched, press_matlSet,
                          lb->matrixLabel,   Task::NewDW,
                          lb->imp_delPLabel, false,
                          lb->rhsLabel,      Task::OldDW,
                          whichInitialGuess, Task::OldDW,
			     solver_parameters);

    scheduleUpdatePressure( subsched,  level, patch_set,  ice_matls,
                                                          mpm_matls, 
                                                          d_press_matl,  
                                                          all_matls);
                                                          
    scheduleImplicitVel_FC( subsched,         patch_set,  ice_matls,
                                                          mpm_matls, 
                                                          d_press_matl, 
                                                          all_matls,
                                                          recursion);

    scheduleSetupRHS(       subsched,         patch_set,  one_matl, 
                                                          all_matls,
                                                          recursion,
                                                          "computes");

    subsched->compile();
    //__________________________________
    //  - move subNewDW to subOldDW
    //  - scrub the subScheduler
    //  - execute the tasks
    subsched->advanceDataWarehouse(grid); 
    subOldDW = subsched->get_dw(2);
    subNewDW = subsched->get_dw(3);
    subOldDW->setScrubbing(DataWarehouse::ScrubComplete);
    subNewDW->setScrubbing(DataWarehouse::ScrubNone);
    
    subsched->execute();
    
    counter ++;
    firstIter = false;
    whichInitialGuess = NULL;
    
    //__________________________________
    // diagnostics
    subNewDW->get(max_RHS,     lb->max_RHSLabel);
    subOldDW->get(max_RHS_old, lb->max_RHSLabel);
    
    if(pg->myrank() == 0) {
      cout << "Outer iteration " << counter
           << " max_rhs before solve "<< max_RHS_old
           << " after solve " << max_RHS<< endl;
    }
    
    //__________________________________
    // restart timestep
                                          //  too many outer iterations
    if (counter > d_iters_before_timestep_restart ){
      restart = true;
      if(pg->myrank() == 0)
        cout <<"\nWARNING: max iterations befor timestep restart reached\n"<<endl;
    }
                                          //  solver has requested a restart
    if (subsched->get_dw(3)->timestepRestarted() ) {
      if(pg->myrank() == 0)
        cout << "\nWARNING: Solver had requested a restart\n" <<endl;
      restart = true;
    }
    
                                           //  solution is diverging
    if(max_RHS < smallest_max_RHS_sofar){
      smallest_max_RHS_sofar = max_RHS;
    }
    if(((max_RHS - smallest_max_RHS_sofar) > 100.0*smallest_max_RHS_sofar) ){
      if(pg->myrank() == 0)
        cout << "\nWARNING: outer interation is diverging now "
             << "restarting the timestep"
             << " Max_RHS " << max_RHS 
             << " smallest_max_RHS_sofar "<< smallest_max_RHS_sofar<< endl;
      restart = true;
    }
    if(restart){
      ParentNewDW->abortTimestep();
      ParentNewDW->restartTimestep();
      return;
    }
  }  // outer iteration loop
  
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
    throw ConvergenceFailure(s.str(),counter,max_RHS,d_outer_iter_tolerance,__FILE__,__LINE__);
  }

  //__________________________________
  // Move products of iteration (only) from sub_new_dw -> parent_new_dw
  subNewDW  = subsched->get_dw(3);
  bool replace = true;
  const MaterialSubset* all_matls_sub = all_matls->getUnion();
  ParentNewDW->transferFrom(subNewDW,         // press
                    lb->press_CCLabel,       patch_sub,  d_press_matl, replace); 
  ParentNewDW->transferFrom(subNewDW,         // term2
                    lb->term2Label,          patch_sub,  one_matl,     replace);    
  ParentNewDW->transferFrom(subNewDW,         // uvel_FC
                    lb->uvel_FCMELabel,      patch_sub,  all_matls_sub,replace); 
  ParentNewDW->transferFrom(subNewDW,         // vvel_FC
                    lb->vvel_FCMELabel,      patch_sub,  all_matls_sub,replace); 
  ParentNewDW->transferFrom(subNewDW,         // wvel_FC
                    lb->wvel_FCMELabel,      patch_sub,  all_matls_sub,replace); 
  ParentNewDW->transferFrom(subNewDW,        // vol_fracX_FC
                    lb->vol_fracX_FCLabel,   patch_sub, all_matls_sub,replace); 
  ParentNewDW->transferFrom(subNewDW,         // vol_fracY_FC
                    lb->vol_fracY_FCLabel,   patch_sub, all_matls_sub,replace); 
  ParentNewDW->transferFrom(subNewDW,         // vol_fracZ_FC
                    lb->vol_fracZ_FCLabel,   patch_sub, all_matls_sub,replace);                 
    
  //__________________________________
  //  Turn scrubbing back on
  ParentOldDW->setScrubbing(ParentOldDW_scrubmode);
  ParentNewDW->setScrubbing(ParentNewDW_scrubmode);  

  //__________________________________
  // clean up memory  
  if(press_matlSet->removeReference()){
    delete press_matlSet;
  }
}
