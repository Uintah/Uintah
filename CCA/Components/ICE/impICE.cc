#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h> 
#include <Core/Util/DebugStream.h>

using namespace SCIRun;
using namespace Uintah;

static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false);


/* ---------------------------------------------------------------------
 Function~  ICE::scheduleSetupMatrix--
_____________________________________________________________________*/
void ICE::scheduleSetupMatrix(  SchedulerP& sched,
                               const LevelP& level,
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
  t->requires( Task::ParentNewDW, lb->speedSound_CCLabel, gn,0);    
  t->requires( Task::ParentNewDW, lb->sp_vol_CCLabel,     gac,1);    
  t->requires( Task::ParentNewDW, lb->vol_frac_CCLabel,   gac,1);    
  t->requires( Task::ParentNewDW, lb->uvel_FCMELabel,     gac,2);    
  t->requires( Task::ParentNewDW, lb->vvel_FCMELabel,     gac,2);    
  t->requires( Task::ParentNewDW, lb->wvel_FCMELabel,     gac,2);    
  
  t->computes(lb->betaLabel,    one_matl);
  t->computes(lb->matrixLabel,  one_matl);
  sched->addTask(t, patches, all_matls);                     
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleSetupRHS--
_____________________________________________________________________*/
void ICE::scheduleSetupRHS(  SchedulerP& sched,
                                          const LevelP& level,
                                          const PatchSet* patches,
                                          const MaterialSubset* one_matl,
                                          const MaterialSet* all_matls)
{
  Task* t;
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
   
  //__________________________________
  //  Form the RHS
  cout_doing << "ICE::scheduleSetupRHS" << endl;
  t = scinew Task("ICE::setupRHS", this, &ICE::setupRHS);
  
  t->requires( Task::ParentOldDW, lb->delTLabel);
  t->requires( Task::ParentNewDW, lb->burnedMass_CCLabel,        gn,0);
  t->requires( Task::ParentNewDW, lb->sp_vol_CCLabel,            gn,0);
  t->requires( Task::ParentNewDW, lb->vol_frac_CCLabel,          gac,1); 
  t->requires( Task::ParentNewDW, lb->uvel_FCMELabel,            gac,2); 
  t->requires( Task::ParentNewDW, lb->vvel_FCMELabel,            gac,2); 
  t->requires( Task::ParentNewDW, lb->wvel_FCMELabel,            gac,2); 

//  t->requires( Task::OldDW,       lb->imp_delPLabel,    one_matl,gn,0);
  t->requires( Task::NewDW,       lb->betaLabel,        one_matl,gn,0);

  t->computes(lb->term2Label, one_matl);
  t->computes(lb->rhsLabel);
  t->computes(lb->initialGuessLabel);
  
  sched->addTask(t, patches, all_matls);                     
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleUpdatePressure--
_____________________________________________________________________*/
void ICE::scheduleUpdatePressure(  SchedulerP& sched,
                                   const LevelP& level,
                                   const PatchSet* patches,
                                   const MaterialSubset* one_matl,
                                   const MaterialSubset* press_matl,
                                   const MaterialSet* all_matls)
{
  Task* t;
   
  Ghost::GhostType  gn  = Ghost::None;
   
  //__________________________________
  // update the pressure
  cout_doing << "ICE::scheduleUpdatePressure" << endl;
  t = scinew Task("ICE::updatePressure", this, &ICE::updatePressure);

  t->requires(Task::ParentNewDW, lb->press_equil_CCLabel,  press_matl,  gn);  
  t->requires(Task::ParentNewDW, lb->sp_vol_CCLabel,                    gn);
  t->requires(Task::ParentNewDW, lb->rho_CCLabel,                       gn);
  t->requires(Task::NewDW,       lb->imp_delPLabel,        press_matl,  gn); 
  t->requires(Task::NewDW,       lb->betaLabel,            one_matl,    gn);
  t->requires(Task::NewDW,       lb->term2Label,           one_matl,    gn);

  t->computes(lb->delP_DilatateLabel, press_matl);
  t->computes(lb->delP_MassXLabel,    press_matl);  
  t->computes(lb->press_CCLabel,      press_matl); 
  t->computes(lb->sum_rho_CCLabel,    one_matl);  // only one mat subset     
  
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
                                          const MaterialSet* all_matls)
{
  cout_doing << "ICE::scheduleImplicitPressureSolve" << endl;
  
  Task* t = scinew Task("ICE::implicitPressureSolve", 
                   this, &ICE::implicitPressureSolve,
                   level, sched.get_rep());
 
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
  t->requires( Task::NewDW, lb->uvel_FCMELabel,     gac,2);    
  t->requires( Task::NewDW, lb->vvel_FCMELabel,     gac,2);    
  t->requires( Task::NewDW, lb->wvel_FCMELabel,     gac,2); 
  t->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,1); 
  t->requires( Task::NewDW, lb->sp_vol_CCLabel,     gac,1);
  //__________________________________
  // SetupMatrix
  t->requires( Task::NewDW, lb->speedSound_CCLabel, gn,0);    
  t->requires( Task::NewDW, lb->sp_vol_CCLabel,     gac,1); 
   
  //__________________________________
  // SetupRHS
  t->requires( Task::OldDW, lb->delTLabel);
  t->requires( Task::NewDW, lb->burnedMass_CCLabel, gn,0);       

  //__________________________________
  // Update Pressure
  t->requires(Task::NewDW, lb->press_equil_CCLabel,  press_matl,  gn);
  t->requires(Task::NewDW, lb->rho_CCLabel,                       gn);           

  t->computes(lb->delP_DilatateLabel, press_matl);
  t->computes(lb->delP_MassXLabel,    press_matl);  
  t->computes(lb->press_CCLabel,      press_matl); 
  t->computes(lb->sum_rho_CCLabel,    one_matl);  // only one mat subset

  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perproc_patches =  loadBal->createPerProcessorPatchSet(level, d_myworld);
  sched->addTask(t, perproc_patches, all_matls);
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
    CCVariable<double> beta;
   
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->allocateAndPut(A,    lb->matrixLabel,  0, patch, gn, 0);    
    new_dw->allocateAndPut(beta, lb->betaLabel,    0, patch, gn, 0);
    IntVector right, left, top, bottom, front, back;
    IntVector R_CC, L_CC, T_CC, B_CC, F_CC, BK_CC;
    
    //__________________________________
    //  Initialize beta and A
    beta.initialize(0.0);
    for(CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      A[c].p = 1.0;
      A[c].n = 0.0;   A[c].s = 0.0;
      A[c].e = 0.0;   A[c].w = 0.0;  // extra cell = 1.0
      A[c].t = 0.0;   A[c].b = 0.0;
    } 
    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      A[c].p = 0.0;
      A[c].n = 0.0;   A[c].s = 0.0;
      A[c].e = 0.0;   A[c].w = 0.0;   // Interior cells = 0.0;
      A[c].t = 0.0;   A[c].b = 0.0;
    }
  
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;
      constCCVariable<double> vol_frac, sp_vol_CC, speedSound;     

      parent_new_dw->get(uvel_FC,   lb->uvel_FCMELabel,        indx,patch,gac, 2);  
      parent_new_dw->get(vvel_FC,   lb->vvel_FCMELabel,        indx,patch,gac, 2);  
      parent_new_dw->get(wvel_FC,   lb->wvel_FCMELabel,        indx,patch,gac, 2);  
      parent_new_dw->get(vol_frac,  lb->vol_frac_CCLabel,      indx,patch,gac, 1);  
      parent_new_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,        indx,patch,gac, 1);  
      parent_new_dw->get(speedSound,lb->speedSound_CCLabel,    indx,patch,gn,  0);       
      
      //__________________________________
      // Sum (<upwinded volfrac> * sp_vol on faces)
      // +x -x +y -y +z -z
      //  e, w, n, s, t, b
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        right    = c + IntVector(1,0,0);    left     = c + IntVector(0,0,0);
        top      = c + IntVector(0,1,0);    bottom   = c + IntVector(0,0,0);
        front    = c + IntVector(0,0,1);    back     = c + IntVector(0,0,0);
        
        R_CC = right;   L_CC  = c - IntVector(1,0,0);  // Left, right
        T_CC = top;     B_CC  = c - IntVector(0,1,0);  // top, bottom
        F_CC = front;   BK_CC = c - IntVector(0,0,1);  // front, back
        //__________________________________
        //  T H I S   I S   G O I N G   T O   B E   S L O W   
        //__________________________________
        //  use the upwinded vol_frac
        IntVector upwnd; 
        double sp_vol_brack_R = 2.0*(sp_vol_CC[c] * sp_vol_CC[R_CC])/      
                                    (sp_vol_CC[c] + sp_vol_CC[R_CC]);      
                                                                         
        double sp_vol_brack_L = 2.0*(sp_vol_CC[c] * sp_vol_CC[L_CC])/      
                                    (sp_vol_CC[c] + sp_vol_CC[L_CC]);      
                                                                         
        double sp_vol_brack_T = 2.0*(sp_vol_CC[c] * sp_vol_CC[T_CC])/      
                                    (sp_vol_CC[c] + sp_vol_CC[T_CC]);                      
                                                                         
        double sp_vol_brack_B = 2.0*(sp_vol_CC[c] * sp_vol_CC[B_CC])/      
                                    (sp_vol_CC[c] + sp_vol_CC[B_CC]);      
 
        double sp_vol_brack_F = 2.0*(sp_vol_CC[c] * sp_vol_CC[F_CC])/      
                                    (sp_vol_CC[c] + sp_vol_CC[F_CC]);      
                                                                         
        double sp_vol_brack_BK= 2.0*(sp_vol_CC[c] * sp_vol_CC[BK_CC])/     
                                    (sp_vol_CC[c] + sp_vol_CC[BK_CC]);                        
                                  
        upwnd   = upwindCell_X(c, uvel_FC[right],  1.0);     
        A[c].e += vol_frac[upwnd] * sp_vol_brack_R;               
                       
        upwnd   = upwindCell_X(c, uvel_FC[left],   0.0);     
        A[c].w += vol_frac[upwnd] * sp_vol_brack_L;               

        upwnd   = upwindCell_Y(c, vvel_FC[top],    1.0);     
        A[c].n += vol_frac[upwnd] * sp_vol_brack_T;               
              
        upwnd   = upwindCell_Y(c, vvel_FC[bottom], 0.0);
        A[c].s += vol_frac[upwnd] * sp_vol_brack_B;

        upwnd   = upwindCell_Z(c, wvel_FC[front],  1.0);
        A[c].t += vol_frac[upwnd] * sp_vol_brack_F;
        
        upwnd   = upwindCell_Z(c, wvel_FC[back],   0.0);
        A[c].b += vol_frac[upwnd] * sp_vol_brack_BK;
      }
      //__________________________________
      // sum beta = sum ( vol_frac * sp_vol/speedSound^2)
      // (THINK ABOUT PULLING OUT OF ITER LOOP)
      // (THINK ABOUT WRITING THIS IN TERMS OF COMPRESSIBLITY
      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        beta[c] += vol_frac[c] * sp_vol_CC[c]/(speedSound[c] * speedSound[c]);
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
      A[c].e *= -tmp_e_w;
      A[c].w *= -tmp_e_w;
      
      A[c].n *= -tmp_n_s;
      A[c].s *= -tmp_n_s;

      A[c].t *= -tmp_t_b;
      A[c].b *= -tmp_t_b;
      
      A[c].p = beta[c] -
                (A[c].n + A[c].s + A[c].e + A[c].w + A[c].t + A[c].b);
    }    
    //---- P R I N T   D A T A ------   
    if (switchDebug_setupMatrix) {    
      ostringstream desc;
      desc << "BOT_setupMatrix_patch_" << patch->getID();
      printStencil( 0, patch, 0, desc.str(), "A", A);
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
    CCVariable<double> q_CC, q_advected; 
    CCVariable<double> rhs, initialGuess;
    CCVariable<double> sumAdvection, massExchTerm;
    constCCVariable<double> beta, press_CC, oldPressure;
    constCCVariable<double> old_imp_delP;
    
    const IntVector gc(1,1,1);
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;   

    parent_new_dw->get(press_CC,        lb->press_equil_CCLabel,0,patch,gn,0);     
    new_dw->get(beta,                   lb->betaLabel,          0,patch,gn,0);   
    new_dw->allocateAndPut(rhs,         lb->rhsLabel,           0,patch);    
    new_dw->allocateAndPut(initialGuess,lb->initialGuessLabel,  0,patch); 
    new_dw->allocateAndPut(massExchTerm,lb->term2Label,         0,patch);
    new_dw->allocateTemporary(q_advected,       patch);
    new_dw->allocateTemporary(sumAdvection,     patch);
    new_dw->allocateTemporary(q_CC,             patch,  gac,1);    
 
    rhs.initialize(0.0);
    initialGuess.initialize(0.0);
    sumAdvection.initialize(0.0);
    massExchTerm.initialize(0.0);
  
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;
      constCCVariable<double> vol_frac, burnedMass, sp_vol_CC;

      parent_new_dw->get(uvel_FC,    lb->uvel_FCMELabel,     indx,patch,gac, 2);    
      parent_new_dw->get(vvel_FC,    lb->vvel_FCMELabel,     indx,patch,gac, 2);    
      parent_new_dw->get(wvel_FC,    lb->wvel_FCMELabel,     indx,patch,gac, 2);    
      parent_new_dw->get(vol_frac,   lb->vol_frac_CCLabel,   indx,patch,gac, 1);  
      parent_new_dw->get(burnedMass, lb->burnedMass_CCLabel, indx,patch,gn,0);
      parent_new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn,0);    
         
      //__________________________________
      // Advection preprocessing
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,indx); 

      for(CellIterator iter = patch->getCellIterator(gc); !iter.done();iter++){
       IntVector c = *iter;
        q_CC[c] = vol_frac[c] * invvol;
      }
      advector->advectQ(q_CC, patch, q_advected, new_dw);
      
      //__________________________________
      //  sum Advecton (<vol_frac> vel_FC)
      //  sum mass Exchange term
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        sumAdvection[c] += q_advected[c];
        
        massExchTerm[c] += burnedMass[c] * (sp_vol_CC[c] * invvol);
      } 
    }  //matl loop
    delete advector;  
      
    //__________________________________
    //  Form RHS
    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
/*`==========TESTING==========*/
//   You need to use this with when iterating
//      double term1 = beta[c] *  (press_CC[c] - oldPressure[c]);
      double term1 = 0.0;
/*===========TESTING==========`*/ 

      double term2 = delT * massExchTerm[c];
      rhs[c] = -term1 + term2 + sumAdvection[c];
/*`==========TESTING==========*/
     initialGuess[c] = 0.0001;  
/*===========TESTING==========`*/
    }
    //---- P R I N T   D A T A ------  
    if (switchDebug_setupRHS) {
      ostringstream desc;
      desc << "BOT_setupRHS_patch_" << patch->getID();
      printData( 0, patch, 0,desc.str(), "rhs",              rhs);
      printData( 0, patch, 0,desc.str(), "sumAdvection",     sumAdvection);
      printData( 0, patch, 0,desc.str(), "MassExchangeTerm", massExchTerm);
      printData( 0, patch, 0,desc.str(), "InitialGuess",     initialGuess);
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
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> sum_rho_CC;        
    constCCVariable<double> imp_delP;
    constCCVariable<double> pressure;
    constCCVariable<double> rho_CC;
    constCCVariable<double> beta;
    constCCVariable<double> massExchTerm;
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
    
    Ghost::GhostType  gn = Ghost::None;
    parent_new_dw->get(pressure,      lb->press_equil_CCLabel,  0,patch,gn,0);
    new_dw->get(beta,                 lb->betaLabel,            0,patch,gn,0);
    new_dw->get(imp_delP,             lb->imp_delPLabel,        0,patch,gn,0);
    new_dw->get(massExchTerm,         lb->term2Label,           0,patch,gn,0);
        
    new_dw->allocateAndPut(delP_Dilatate,lb->delP_DilatateLabel,0, patch);
    new_dw->allocateAndPut(delP_MassX,   lb->delP_MassXLabel,   0, patch);
    new_dw->allocateAndPut(press_CC,     lb->press_CCLabel,     0, patch);
    new_dw->allocateAndPut(sum_rho_CC,   lb->sum_rho_CCLabel,   0, patch);

    sum_rho_CC.initialize(0.0);
    press_CC.initialize(0.);
    delP_Dilatate.initialize(0.0);
    delP_MassX.initialize(0.0); 
         
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      parent_new_dw->get(sp_vol_CC[m],lb->sp_vol_CCLabel, indx,patch,gn,0);
      parent_new_dw->get(rho_CC,      lb->rho_CCLabel,    indx,patch,gn,0);
      //__________________________________
      //  compute sum_rho_CC used by press_FC
      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        sum_rho_CC[c] += rho_CC[c];
      } 
    }    
    //__________________________________
    //  add delP to press
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      press_CC[c] = pressure[c] + imp_delP[c];
    }
    
    //__________________________________
    // backout delP_Dilatate and delP_MassX
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      delP_MassX[c]    = massExchTerm[c]/beta[c];
      delP_Dilatate[c] = imp_delP[c] - delP_MassX[c];
    }    
    setBC(press_CC, sp_vol_CC[SURROUND_MAT],
          "sp_vol", "Pressure", patch ,d_sharedState, 0, new_dw);
    
    //---- P R I N T   D A T A ------  
    if (switchDebug_updatePressure) {
      ostringstream desc;
      desc << "BOT_updatePressure_patch_" << patch->getID();
      printData( 0, patch, 1,desc.str(), "imp_delP",      imp_delP); 
      printData( 0, patch, 1,desc.str(), "delP_Dilatate", delP_Dilatate);
      printData( 0, patch, 1,desc.str(), "delP_MassX",    delP_MassX);
      printData( 0, patch, 1,desc.str(), "Press_CC",      press_CC);    
    }  
  } // patch loop
}
 
/* --------------------------------------------------------------------- 
 Function~  ICE::implicitPressureSolve-- 
_____________________________________________________________________*/
void ICE::implicitPressureSolve(const ProcessorGroup*,
		                  const PatchSubset* patch_sub, 
		                  const MaterialSubset*,        
		                  DataWarehouse* ParentOldDW,    
                                DataWarehouse* ParentNewDW,    
		                  LevelP level,                 
                                Scheduler* sched)
{
  cout_doing<<"Doing implicitPressureSolve "<<"\t\t\t\t ICE" << endl;
  //__________________________________
  // define Matls
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  MaterialSubset* one_matl = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  MaterialSubset* press_matl = one_matl;
  //__________________________________
  // - create subScheduler
  // - turn scrubbing off
  // - assign Data Warehouse????
  SchedulerP subsched = sched->createSubScheduler();
  DataWarehouse::ScrubMode ParentOldDW_scrubmode =
                           ParentOldDW->setScrubbing(DataWarehouse::ScrubNone);
  DataWarehouse::ScrubMode ParentNewDW_scrubmode =
                           ParentNewDW->setScrubbing(DataWarehouse::ScrubNone);
  subsched->initialize(3, 1, ParentOldDW, ParentNewDW);
  subsched->clearMappings();
  subsched->mapDataWarehouse(Task::ParentOldDW, 0);
  subsched->mapDataWarehouse(Task::ParentNewDW, 1);
  subsched->mapDataWarehouse(Task::OldDW, 2);
  subsched->mapDataWarehouse(Task::NewDW, 3); 
  
  GridP grid = level->getGrid();
  subsched->advanceDataWarehouse(grid);
  
  const PatchSet* patch_set = level->eachPatch();
    
  //__________________________________
  // Create the tasks

  scheduleSetupMatrix(    subsched, level,  patch_set,  one_matl, 
                                                        all_matls);
  
  scheduleSetupRHS(       subsched, level,  patch_set,  one_matl, 
                                                        all_matls);

  solver->scheduleSolve(level, subsched, all_matls,
                        lb->matrixLabel, lb->imp_delPLabel, false,
                        lb->rhsLabel,    lb->initialGuessLabel,
                        Task::NewDW,     solver_parameters);
                        
  scheduleUpdatePressure( subsched,  level, patch_set,  one_matl,  
                                                        press_matl,  
                                                        all_matls);

  subsched->compile(d_myworld);      
                                                    
  //__________________________________                                                
  // Pull data from parent and put in sub_sched_old dw. Use:
   delt_vartype dt;
   subsched->get_dw(0)->get(dt, d_sharedState->get_delt_label());
   subsched->get_dw(3)->put(dt, d_sharedState->get_delt_label());
 
  //__________________________________
  //  Iteration Loop
  int counter = 0;
  while(counter < 1) {
    cerr << "Iteration = " << counter++ << "\n";
    subsched->advanceDataWarehouse(grid);
    subsched->get_dw(2)->setScrubbing(DataWarehouse::ScrubComplete);
    subsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);
    subsched->execute(d_myworld);
  }  
  //__________________________________
  // Move products of iteration (only) from sub_new_dw -> parent_new_dw 
//  ParentNewDW->transferFrom(subsched->get_dw(3), lb->delP_DilatateLabel, patch_sub, press_matl);
  ParentNewDW->transferFrom(subsched->get_dw(3), lb->delP_DilatateLabel, patch_sub, press_matl);
  ParentNewDW->transferFrom(subsched->get_dw(3), lb->delP_MassXLabel,    patch_sub, press_matl);
  ParentNewDW->transferFrom(subsched->get_dw(3), lb->press_CCLabel,      patch_sub, press_matl); 
  ParentNewDW->transferFrom(subsched->get_dw(3), lb->sum_rho_CCLabel,    patch_sub, one_matl);
  //__________________________________
  //  Turn scrubbing back on???
  ParentOldDW->setScrubbing(ParentOldDW_scrubmode);
  ParentNewDW->setScrubbing(ParentNewDW_scrubmode);
}
