#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
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
   
  MaterialSubset* one_matl = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  //__________________________________
  //  Form the matrix
  cout_doing << "ICE::scheduleSetupMatrix" << endl;
  t = scinew Task("setupMatrix", this, &ICE::setupMatrix);
  t->requires( Task::OldDW, lb->delTLabel);
  t->requires( Task::NewDW, lb->speedSound_CCLabel, gn,0);    
  t->requires( Task::NewDW, lb->sp_vol_CCLabel,     gac,1);    
  t->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,1);    
  t->requires( Task::NewDW, lb->uvel_FCMELabel,     gac,2);    
  t->requires( Task::NewDW, lb->vvel_FCMELabel,     gac,2);    
  t->requires( Task::NewDW, lb->wvel_FCMELabel,     gac,2);    
  
  t->computes(lb->betaLabel,    one_matl);
  t->computes(lb->matrixLabel,  one_matl);
  sched->addTask(t, patches, all_matls);
  
  
  //__________________________________
  //  Form the RHS
  cout_doing << "ICE::scheduleSetupRHS" << endl;
  t = scinew Task("setupRHS", this, &ICE::setupRHS);
  t->requires( Task::OldDW, lb->delTLabel);
  t->requires( Task::NewDW, lb->betaLabel,        one_matl,gn,0);
  t->requires( Task::NewDW, lb->burnedMass_CCLabel,        gn,0);
  t->requires( Task::NewDW, lb->sp_vol_CCLabel,            gn,0);
  t->requires( Task::NewDW, lb->vol_frac_CCLabel,          gac,1); 
  t->requires( Task::NewDW, lb->uvel_FCMELabel,            gac,2); 
  t->requires( Task::NewDW, lb->vvel_FCMELabel,            gac,2); 
  t->requires( Task::NewDW, lb->wvel_FCMELabel,            gac,2); 
  
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
  t->requires(Task::NewDW, lb->rho_CCLabel,                       gn);
    
  t->computes(lb->delP_DilatateLabel, press_matl);
  t->computes(lb->delP_MassXLabel,    press_matl);  
  t->computes(lb->press_CCLabel,      press_matl); 
  t->computes(lb->sum_rho_CCLabel,    one_matl);  // only one mat subset     
      
  sched->addTask(t, patches, all_matls);
  
  if (one_matl->removeReference())
    delete one_matl;                      
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
    double vol = dx.x() * dx.y() * dx.z();
    CCVariable<Stencil7> A; 
    CCVariable<double> beta;
   
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->allocateAndPut(A,    lb->matrixLabel,  0, patch);    
    new_dw->allocateAndPut(beta, lb->betaLabel,    0, patch);
    IntVector right, left, top, bottom, front, back;
    IntVector R_CC, L_CC, T_CC, B_CC, F_CC, BK_CC;
    
    beta.initialize(0.0);
    for(CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      A[c].p = 0.0;
      A[c].n = 0.0;
      A[c].s = 0.0;
      A[c].e = 0.0;
      A[c].w = 0.0;
      A[c].t = 0.0;
      A[c].b = 0.0;
    } 
    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;
      constCCVariable<double> vol_frac, sp_vol_CC, speedSound;     

      new_dw->get(uvel_FC,   lb->uvel_FCMELabel,        indx,patch,gac, 2);  
      new_dw->get(vvel_FC,   lb->vvel_FCMELabel,        indx,patch,gac, 2);  
      new_dw->get(wvel_FC,   lb->wvel_FCMELabel,        indx,patch,gac, 2);  
      new_dw->get(vol_frac,  lb->vol_frac_CCLabel,      indx,patch,gac, 1);  
      new_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,        indx,patch,gac, 1);  
      new_dw->get(speedSound,lb->speedSound_CCLabel,    indx,patch,gn,  0);       
      
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
    double areaX_inv_dx = dx.y() * dx.z()/dx.x();
    double areaY_inv_dy = dx.x() * dx.z()/dx.y();
    double areaZ_inv_dz = dx.x() * dx.y()/dx.z();
        
    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      A[c].e *= delT_2 * areaX_inv_dx;
      A[c].w *= delT_2 * areaX_inv_dx;
      
      A[c].n *= delT_2 * areaY_inv_dy;
      A[c].s *= delT_2 * areaY_inv_dy;

      A[c].t *= delT_2 * areaZ_inv_dz;
      A[c].b *= delT_2 * areaZ_inv_dz;
      
      A[c].p = beta[c] * vol + 
                (A[c].n + A[c].s + A[c].e + A[c].w + A[c].t + A[c].b);
    }    
/*`==========TESTING==========*/
    //__________________________________
    //  setup up the matrix an rhs
    // DUMMY matrix for right now
    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      A[c].p = 1.0;
      A[c].n = 1.0;
      A[c].s = 1.0;
      A[c].e = 1.0;
      A[c].w = 1.0;
      A[c].t = 1.0;
      A[c].b = 1.0;
    } 
/*===========TESTING==========`*/       
  }
}

/* --------------------------------------------------------------------- 
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
    CCVariable<double> sumAdvection, massExchangeTerm;
    constCCVariable<double> beta, press_CC, oldPressure;
    
    const IntVector gc(1,1,1);
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;   
    
    new_dw->get(beta,                   lb->betaLabel,          0,patch,gn,0);   
    new_dw->get(press_CC,               lb->press_equil_CCLabel,0,patch,gn,0);     
    new_dw->allocateAndPut(rhs,         lb->rhsLabel,           0,patch);    
    new_dw->allocateAndPut(initialGuess,lb->initialGuessLabel,  0,patch);  
    new_dw->allocateTemporary(q_advected,       patch);
    new_dw->allocateTemporary(sumAdvection,     patch);
    new_dw->allocateTemporary(massExchangeTerm, patch);
    new_dw->allocateTemporary(q_CC,             patch,  gac,1);    
 
    sumAdvection.initialize(0.0);
    massExchangeTerm.initialize(0.0);
  
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;
      constCCVariable<double> vol_frac, burnedMass, sp_vol_CC;

      new_dw->get(uvel_FC,    lb->uvel_FCMELabel,     indx,patch,gac, 2);    
      new_dw->get(vvel_FC,    lb->vvel_FCMELabel,     indx,patch,gac, 2);    
      new_dw->get(wvel_FC,    lb->wvel_FCMELabel,     indx,patch,gac, 2);    
      new_dw->get(vol_frac,   lb->vol_frac_CCLabel,   indx,patch,gac, 1);  
      new_dw->get(burnedMass, lb->burnedMass_CCLabel, indx,patch,gn,0);
      new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn,0);    
         
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
        sumAdvection[c]     += q_advected[c];
        
        massExchangeTerm[c] += burnedMass[c] * (sp_vol_CC[c]/vol);
        
      } 
    }  //matl loop
    delete advector;  
      
    //__________________________________
    //  Form RHS
    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
/*`==========TESTING==========*/
//      double term1 = beta[c] * vol*(press_CC[c] - oldPressure[c]);
/*===========TESTING==========`*/ 
      double term1 = 0.0;
      double term2 = delT * massExchangeTerm[c];
      rhs[c] = -term1 + term2 -sumAdvection[c];
/*`==========TESTING==========*/
      rhs[c] = 1; 
/*===========TESTING==========`*/ 
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
