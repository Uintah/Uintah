#include "ICE.h"
#include <Packages/Uintah/CCA/Components/ICE/MathToolbox.h>
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

using namespace SCIRun;
using namespace Uintah;

static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false);

/* ---------------------------------------------------------------------
 Function~ ICE::scheduleComputeFCPressDiffRF
_____________________________________________________________________*/
void ICE::scheduleComputeFCPressDiffRF(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSubset* ice_matls,
                                        const MaterialSubset* mpm_matls,
                                        const MaterialSubset* press_matl,
                                        const MaterialSet* matls)
{
  cout_doing << "ICE::scheduleComputeFCPressDiffRF" << endl;
  Task* t = scinew Task("ICE::computeFCPressDiffRF",
                     this, &ICE::computeFCPressDiffRF);
  
  t->requires(Task::OldDW,lb->delTLabel);
  t->requires(Task::NewDW,lb->rho_CCLabel,            Ghost::AroundCells,1);
  t->requires(Task::NewDW,lb->sp_vol_CCLabel,         Ghost::AroundCells,1);  
  t->requires(Task::NewDW,lb->speedSound_CCLabel,     Ghost::AroundCells,1); 
  t->requires(Task::NewDW,lb->matl_press_CCLabel,     Ghost::AroundCells,1);
  t->requires(Task::NewDW,lb->vel_CCLabel,mpm_matls,  Ghost::AroundCells,1);
  t->requires(Task::OldDW,lb->vel_CCLabel,ice_matls,  Ghost::AroundCells,1);
  t->requires(Task::NewDW,lb->press_equil_CCLabel,press_matl,
                                                      Ghost::AroundCells,1);

  t->computes(lb->press_diffX_FCLabel);
  t->computes(lb->press_diffY_FCLabel);
  t->computes(lb->press_diffZ_FCLabel);

  sched->addTask(t, patches, matls);
}

/* --------------------------------------------------------------------- 
 Function~  ICE::computeRateFormPressure-- 
 Reference: A Multifield Model and Method for Fluid Structure
            Interaction Dynamics
_____________________________________________________________________*/
void ICE::computeRateFormPressure(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* ,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing<<"Doing computeRateFormPressure on patch "
              << patch->getID() <<"\t\t ICE" << endl;

    double tmp;
    int numMatls = d_sharedState->getNumICEMatls();
    Vector dx       = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    
    StaticArray<double> dp_drho(numMatls),dp_de(numMatls);
    StaticArray<double> mat_volume(numMatls);
    StaticArray<double> mat_mass(numMatls);
    StaticArray<double> cv(numMatls);
    StaticArray<double> gamma(numMatls);
    StaticArray<double> compressibility(numMatls);
    StaticArray<CCVariable<double> > vol_frac(numMatls);
    StaticArray<CCVariable<double> > rho_micro(numMatls);
    StaticArray<CCVariable<double> > rho_CC_new(numMatls);    
    StaticArray<CCVariable<double> > speedSound_new(numMatls);
    StaticArray<CCVariable<double> > f_theta(numMatls);
    StaticArray<CCVariable<double> > matl_press(numMatls);
    StaticArray<CCVariable<double> > sp_vol_new(numMatls);   
    StaticArray<constCCVariable<double> > rho_CC(numMatls);
    StaticArray<constCCVariable<double> > Temp(numMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);

    constCCVariable<double> press;
    CCVariable<double> press_new; 

    old_dw->get(press,         lb->press_CCLabel, 0,patch,Ghost::None, 0); 
    new_dw->allocate(press_new,lb->press_equil_CCLabel, 0,patch);
    
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* matl = d_sharedState->getICEMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(Temp[m],   lb->temp_CCLabel,  indx,patch,Ghost::None,0);
      old_dw->get(rho_CC[m], lb->rho_CCLabel,   indx,patch,Ghost::None,0);
      old_dw->get(sp_vol_CC[m],
                             lb->sp_vol_CCLabel,indx,patch,Ghost::None,0);
      new_dw->allocate(sp_vol_new[m],lb->sp_vol_CCLabel,    indx, patch); 
      new_dw->allocate(rho_CC_new[m],lb->rho_CCLabel,       indx, patch);
      new_dw->allocate(vol_frac[m],  lb->vol_frac_CCLabel,  indx, patch);
      new_dw->allocate(f_theta[m],   lb->f_theta_CCLabel,   indx, patch);
      new_dw->allocate(matl_press[m],lb->matl_press_CCLabel,indx, patch);
      new_dw->allocate(rho_micro[m], lb->rho_micro_CCLabel, indx, patch);
      new_dw->allocate(speedSound_new[m],lb->speedSound_CCLabel,indx,patch);
      speedSound_new[m].initialize(0.0);
      cv[m] = matl->getSpecificHeat();
      gamma[m] = matl->getGamma();
    }
    
    press_new.initialize(0.0);

    //__________________________________
    // Compute matl_press, speedSound, total_mat_vol
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      double total_mat_vol = 0.0;
      for (int m = 0; m < numMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);

        rho_micro[m][c] = 1.0/sp_vol_CC[m][c];
        ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m],
                                             cv[m],Temp[m][c],
                                             matl_press[m][c],dp_drho[m],dp_de[m]);

        mat_volume[m] = (rho_CC[m][c] * cell_vol) * sp_vol_CC[m][c];

        tmp = dp_drho[m] + dp_de[m] * 
           (matl_press[m][c] * (sp_vol_CC[m][c] * sp_vol_CC[m][c]));            

        total_mat_vol += mat_volume[m];
/*`==========TESTING==========*/
        speedSound_new[m][c] = sqrt(tmp)/gamma[m];  // Isothermal speed of sound
        speedSound_new[m][c] = sqrt(tmp);           // Isentropic speed of sound
        compressibility[m] = sp_vol_CC[m][c]/ 
                            (speedSound_new[m][c] * speedSound_new[m][c]); 
/*==========TESTING==========`*/
       } 
      //__________________________________
      // Compute 1/f_theta
       double f_theta_denom = 0.0;
       for (int m = 0; m < numMatls; m++) {
         vol_frac[m][c] = mat_volume[m]/total_mat_vol;
         f_theta_denom += vol_frac[m][c]*compressibility[m];
       }
       //__________________________________
       // Compute press_new
       for (int m = 0; m < numMatls; m++) {
         f_theta[m][c] = vol_frac[m][c]*compressibility[m]/f_theta_denom;
         press_new[c] += f_theta[m][c]*matl_press[m][c];
       }
    } // for(CellIterator...)

    //__________________________________
    //  Set BCs matl_press, press
    for (int m = 0; m < numMatls; m++)   {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      setBC(matl_press[m], rho_micro[SURROUND_MAT],
            "rho_micro", "Pressure", patch, d_sharedState, indx, new_dw);
    }  
    setBC(press_new,  rho_micro[SURROUND_MAT],
         "rho_micro", "Pressure", patch, d_sharedState, 0, new_dw);
    //__________________________________
    // carry rho_cc forward for MPMICE
    // carry sp_vol_CC forward for ICE:computeEquilibrationPressure
    for (int m = 0; m < numMatls; m++)   {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      rho_CC_new[m].copyData(rho_CC[m]);
      sp_vol_new[m].copyData(sp_vol_CC[m]);
      new_dw->put( sp_vol_new[m],    lb->sp_vol_CCLabel,     indx, patch); 
      new_dw->put( vol_frac[m],      lb->vol_frac_CCLabel,   indx, patch);
      new_dw->put( f_theta[m],       lb->f_theta_CCLabel,    indx, patch);
      new_dw->put( matl_press[m],    lb->matl_press_CCLabel, indx, patch);
      new_dw->put( speedSound_new[m],lb->speedSound_CCLabel, indx, patch);
      new_dw->put( rho_CC_new[m],    lb->rho_CCLabel,        indx, patch);
    }
    new_dw->put(press_new,lb->press_equil_CCLabel,0,patch);
    
   //---- P R I N T   D A T A ------   
    if (switchDebug_EQ_RF_press) {
      ostringstream desc;
      desc << "BOT_computeRFPress_patch_" << patch->getID();
      printData( patch, 1, desc.str(), "Press_CC_RF", press_new);

     for (int m = 0; m < numMatls; m++)  {
       ICEMaterial* matl = d_sharedState->getICEMaterial( m );
       int indx = matl->getDWIndex(); 
       ostringstream desc;
       desc << "BOT_computeRFPress_Mat_" << indx << "_patch_"<< patch->getID();
       printData( patch, 1, desc.str(), "rho_CC",       rho_CC[m]);
       printData( patch, 1, desc.str(), "sp_vol_CC",    sp_vol_new[m]);
       printData( patch, 1, desc.str(), "rho_micro_CC", rho_micro[m]);
       printData( patch, 1, desc.str(), "vol_frac_CC",  vol_frac[m]);
     }
    }
  } // patches
}

/* --------------------------------------------------------------------- 
 Function~  ICE::computeFCPressDiffRF-- 
 Reference: A Multifield Model and Method for Fluid Structure
            Interaction Dynamics.  
 Note:      This only computes the isotropic part of the stress difference term
             Eq 4.13b
_____________________________________________________________________*/
void ICE::computeFCPressDiffRF(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* /*matls*/,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing computeFCPressDiffRF on patch " << patch->getID()
         << "\t\t\t ICE" << endl;

    int numMatls = d_sharedState->getNumMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx      = patch->dCell();

    StaticArray<CCVariable<double>      > scratch(3); 
    StaticArray<constCCVariable<double> > rho_CC(numMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
    StaticArray<constCCVariable<double> > speedSound(numMatls);
    StaticArray<constCCVariable<double> > matl_press(numMatls);
    StaticArray<constCCVariable<Vector> > vel_CC(numMatls);
    StaticArray<SFCXVariable<double>    > press_diffX_FC(numMatls);
    StaticArray<SFCYVariable<double>    > press_diffY_FC(numMatls);
    StaticArray<SFCZVariable<double>    > press_diffZ_FC(numMatls);
    constCCVariable<double> press_CC;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(press_CC,lb->press_equil_CCLabel,0,patch,gac,1);

    for(int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      new_dw->get(rho_CC[m],       lb->rho_CCLabel,       indx,patch, gac,1);
      new_dw->get(sp_vol_CC[m],    lb->sp_vol_CCLabel,    indx,patch, gac,1);
      new_dw->get(speedSound[m],   lb->speedSound_CCLabel,indx,patch, gac,1); 
      new_dw->get(matl_press[m],   lb->matl_press_CCLabel,indx,patch, gac,1);
      if(ice_matl){
        old_dw->get(vel_CC[m],lb->vel_CCLabel,indx,patch,gac,1);      
      }
      if(mpm_matl){
        new_dw->get(vel_CC[m],lb->vel_CCLabel,indx,patch,gac,1);      
      }
      new_dw->allocate(press_diffX_FC[m],lb->press_diffX_FCLabel, indx, patch);
      new_dw->allocate(press_diffY_FC[m],lb->press_diffY_FCLabel, indx, patch);
      new_dw->allocate(press_diffZ_FC[m],lb->press_diffZ_FCLabel, indx, patch);
      press_diffX_FC[m].initialize(0.0);
      press_diffY_FC[m].initialize(0.0);
      press_diffZ_FC[m].initialize(0.0);
    }
    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces
    
    for(int dir=0; dir < 3; dir ++) {
      new_dw->allocate(scratch[dir], lb->scratchLabel, 0, patch); 
    } 
   //---- P R I N T   D A T A ------    
    if (switchDebug_PressDiffRF ) {
      for(int m = 0; m < numMatls; m++)  {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc << "TOP_computeFCPressDiffRF_"<< indx <<"_patch_"<< patch->getID();
        if(m ==0 ) {
          printData( patch,1, desc.str(), "press_CC", press_CC);
        }
        printData(  patch, 1, desc.str(), "rho_CC",     rho_CC[m]);
        printData(  patch, 1, desc.str(), "matl_press", matl_press[m]);
        printVector(patch, 1, desc.str(), "uvel_CC", 0, vel_CC[m]); 
        printVector(patch, 1, desc.str(), "vvel_CC", 1, vel_CC[m]); 
        printVector(patch, 1, desc.str(), "wvel_CC", 2, vel_CC[m]); 
      }
    }
    //______________________________________________________________________
    for(int m = 0; m < numMatls; m++) {
      //__________________________________
      //  For each face compute the press_Diff
      for (int dir= 0; dir < 3; dir++) {
        for(CellIterator iter=patch->getSFCIterator(dir);!iter.done(); iter++){
          IntVector R = *iter;
          IntVector L = R + adj_offset[dir]; 

//__________________________________
//WARNING: We're currently using the 
// isentropic compressibility instead of 
// its cousin the isothermal compressiblity.  
          double kappa_R = sp_vol_CC[m][R]/
                          ( speedSound[m][R] * speedSound[m][R] );
          double kappa_L = sp_vol_CC[m][L]/
                          ( speedSound[m][L] * speedSound[m][L] );
                          
          double rho_brack = (rho_CC[m][R]*rho_CC[m][L])/
                             (rho_CC[m][R]+rho_CC[m][L]);           
          
          double term1 = 
                (matl_press[m][R] - press_CC[R]) * sp_vol_CC[m][R] +
                (matl_press[m][L] - press_CC[L]) * sp_vol_CC[m][L];
                
          double term2 = -2.0 * delT *
                         ((sp_vol_CC[m][R] + sp_vol_CC[m][L])/
                         (kappa_R          + kappa_L) ) *
                         ((vel_CC[m][R](dir) - vel_CC[m][L](dir))/dx(dir)); 
 
          scratch[dir][R] = rho_brack * (term1 + term2);
        }
      } 
      //__________________________________
      //  Extract the different components
      for(CellIterator iter=patch->getSFCXIterator();!iter.done(); iter++){
        IntVector cur = *iter;
        press_diffX_FC[m][cur] = scratch[0][cur];   
      }
      for(CellIterator iter=patch->getSFCYIterator();!iter.done(); iter++){
        IntVector cur = *iter;
        press_diffY_FC[m][cur] = scratch[1][cur];   
      }
      for(CellIterator iter=patch->getSFCZIterator();!iter.done(); iter++){
        IntVector cur = *iter;
        press_diffZ_FC[m][cur] = scratch[2][cur];   
      }     
    }  // for(numMatls)
    
    for(int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->put(press_diffX_FC[m],lb->press_diffX_FCLabel, indx, patch);
      new_dw->put(press_diffY_FC[m],lb->press_diffY_FCLabel, indx, patch);
      new_dw->put(press_diffZ_FC[m],lb->press_diffZ_FCLabel, indx, patch);
    }
    //---- P R I N T   D A T A ------ 
    if (switchDebug_PressDiffRF ) {
      for(int m = 0; m < numMatls; m++)  {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc << "BOT_computeFCPressDiffRF_"<< indx <<"_patch_"<< patch->getID();
        printData_FC( patch,1, desc.str(), "press_diffX_FC", press_diffX_FC[m]);
        printData_FC( patch,1, desc.str(), "press_diffY_FC", press_diffY_FC[m]);
        printData_FC( patch,1, desc.str(), "press_diffZ_FC", press_diffZ_FC[m]);
      }
    }
  } // patches
}

/* ---------------------------------------------------------------------
 Function~  ICE::computeFaceCenteredVelocitiesRF--
 Purpose~   compute the face centered velocities minus the exchange
            contribution.  See Kashiwa Feb. 2001, 4.10a for description
            of term1-term4.
_____________________________________________________________________*/
template<class T> void ICE::computeVelFaceRF(int dir, CellIterator it,
                                       IntVector adj_offset,double dx,
                                       double delT, double gravity,
                                       constCCVariable<double>& sp_vol_CC,
                                       constCCVariable<Vector>& vel_CC,
                                       constCCVariable<double>& vol_frac,
                                       constCCVariable<double>& matl_press_CC,
                                       constCCVariable<double>& press_CC,
                                       const T& sig_bar_FC,
                                       T& vel_FC)
{
  double term1, term2, term3, term4, sp_vol_brack, sig_L, sig_R;
  for(;!it.done(); it++){
    IntVector R = *it;
    IntVector L = R + adj_offset; 
    sp_vol_brack = 2.*(sp_vol_CC[L] * sp_vol_CC[R])/
                      (sp_vol_CC[L] + sp_vol_CC[R]);
    //__________________________________
    // interpolation to the face           
    term1 = (vel_CC[L](dir) * sp_vol_CC[R] +
             vel_CC[R](dir) * sp_vol_CC[L])/
             (sp_vol_CC[R] + sp_vol_CC[L]);

    //__________________________________
    // pressure term
    term2 = sp_vol_brack * (delT/dx) * 
                           (press_CC[R] - press_CC[L]);
    //__________________________________
    // gravity term
    term3 =  delT * gravity;

    // stress difference term
    sig_L = vol_frac[L] * (matl_press_CC[L] - press_CC[L]);
    sig_R = vol_frac[R] * (matl_press_CC[R] - press_CC[R]);
    term4 = (sp_vol_brack * delT/dx) * (
            (sig_bar_FC[R] - sig_L)/vol_frac[L] +
            (sig_R - sig_bar_FC[R])/vol_frac[R]);

    // Todd, I think that term4 should be a negative, since Bucky
    // has a positive, but since sigma = -p*I.  What do you think?
    vel_FC[R] = term1 - term2 + term3 - term4;
  } 
}
//______________________________________________________________________
//
void ICE::computeFaceCenteredVelocitiesRF(const ProcessorGroup*,  
                                         const PatchSubset* patches,
                                         const MaterialSubset* /*matls*/,
                                         DataWarehouse* old_dw, 
                                         DataWarehouse* new_dw)
{
  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing compute_face_centered_velocitiesRF on patch " 
         << patch->getID() << "\t ICE" << endl;
    int numMatls = d_sharedState->getNumMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    delt_vartype doMechOld;
    old_dw->get(doMechOld, lb->doMechLabel);
    Vector dx      = patch->dCell();
    Vector gravity = d_sharedState->getGravity();

    constCCVariable<double> press_CC;
    constCCVariable<double> matl_press_CC;
    Ghost::GhostType  gac = Ghost::AroundCells; 
    new_dw->get(press_CC,lb->press_equil_CCLabel, 0,patch,gac,1);

    // Compute the face centered velocities
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      constCCVariable<double> rho_CC,vol_frac, sp_vol_CC;
      constCCVariable<Vector> vel_CC;
      constSFCXVariable<double> p_dXFC;
      constSFCYVariable<double> p_dYFC;
      constSFCZVariable<double> p_dZFC;
      if(ice_matl){
        new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch, gac,1);
        old_dw->get(vel_CC, lb->vel_CCLabel, indx, patch, gac,1);
      } else {
        new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch, gac,1);
        new_dw->get(vel_CC, lb->vel_CCLabel, indx, patch, gac,1);
      }                                                  
      new_dw->get(sp_vol_CC,     lb->sp_vol_CCLabel,      indx,patch,gac,1);
      new_dw->get(p_dXFC,        lb->press_diffX_FCLabel, indx,patch,gac,1);  
      new_dw->get(p_dYFC,        lb->press_diffY_FCLabel, indx,patch,gac,1); 
      new_dw->get(p_dZFC,        lb->press_diffZ_FCLabel, indx,patch,gac,1);  
      new_dw->get(matl_press_CC, lb->matl_press_CCLabel,  indx,patch,gac,1);
      new_dw->get(vol_frac,      lb->vol_frac_CCLabel,    indx,patch,gac,1);

   //---- P R I N T   D A T A ------ 
      if (switchDebug_vel_FC ) {
        ostringstream desc;
        desc << "TOP_vel_FC_Mat_" << indx << "_patch_" << patch->getID(); 
        printData(    patch,1, desc.str(), "rho_CC",     rho_CC);
        printData(    patch,1, desc.str(), "sp_vol_CC",  sp_vol_CC);
        printVector(  patch,1, desc.str(), "uvel_CC", 0, vel_CC);
        printVector(  patch,1, desc.str(), "vvel_CC", 1, vel_CC);
        printVector(  patch,1, desc.str(), "wvel_CC", 2, vel_CC);
        printData_FC( patch,1, desc.str(), "p_dXFC",     p_dXFC);
        printData_FC( patch,1, desc.str(), "p_dYFC",     p_dYFC);
        printData_FC( patch,1, desc.str(), "p_dZFC",     p_dZFC);
      }

      SFCXVariable<double> uvel_FC;
      SFCYVariable<double> vvel_FC;
      SFCZVariable<double> wvel_FC;
      new_dw->allocate(uvel_FC, lb->uvel_FCLabel, indx, patch);
      new_dw->allocate(vvel_FC, lb->vvel_FCLabel, indx, patch);
      new_dw->allocate(wvel_FC, lb->wvel_FCLabel, indx, patch);
      IntVector lowIndex(patch->getSFCXLowIndex());
      uvel_FC.initialize(0.0, lowIndex,patch->getSFCXHighIndex());
      vvel_FC.initialize(0.0, lowIndex,patch->getSFCYHighIndex());
      wvel_FC.initialize(0.0, lowIndex,patch->getSFCZHighIndex());

      vector<IntVector> adj_offset(3);
      adj_offset[0] = IntVector(-1, 0, 0);    // X faces
      adj_offset[1] = IntVector(0, -1, 0);    // Y faces
      adj_offset[2] = IntVector(0,  0, -1);   // Z faces  
      int offset=0; // 0=Compute all faces in computational domain
                    // 1=Skip the faces at the border between interior and gc      
      if(doMechOld < -1.5){
      //__________________________________
      //  Compute vel_FC for each face
      computeVelFaceRF<SFCXVariable<double> >(0, patch->getSFCXIterator(offset),
                                       adj_offset[0], dx(0), delT, gravity(0),
                                       sp_vol_CC, vel_CC, vol_frac,
                                       matl_press_CC, press_CC,
                                       p_dXFC,    uvel_FC);

      computeVelFaceRF<SFCYVariable<double> >(1, patch->getSFCYIterator(offset),
                                       adj_offset[1], dx(1), delT, gravity(1),
                                       sp_vol_CC, vel_CC, vol_frac, 
                                       matl_press_CC, press_CC,
                                       p_dYFC,    vvel_FC);

      computeVelFaceRF<SFCZVariable<double> >(2, patch->getSFCZIterator(offset),
                                       adj_offset[2], dx(2), delT, gravity(2),
                                       sp_vol_CC, vel_CC, vol_frac,
                                       matl_press_CC, press_CC,
                                       p_dZFC,    wvel_FC); 
      }  // if doMech

      //__________________________________
      // (*)vel_FC BC are updated in 
      // ICE::addExchangeContributionToFCVel()
      new_dw->put(uvel_FC, lb->uvel_FCLabel, indx, patch);
      new_dw->put(vvel_FC, lb->vvel_FCLabel, indx, patch);
      new_dw->put(wvel_FC, lb->wvel_FCLabel, indx, patch);

      //---- P R I N T   D A T A ------ 
      if (switchDebug_vel_FC ) {
        ostringstream desc;
        desc << "bottom_of_vel_FC_Mat_"<< indx <<"_patch_"<< patch->getID();
        printData_FC( patch,1, desc.str(), "uvel_FC", uvel_FC);
        printData_FC( patch,1, desc.str(), "vvel_FC", vvel_FC);
        printData_FC( patch,1, desc.str(), "wvel_FC", wvel_FC);
      }
    } // matls loop
  }  // patch loop
}

/*---------------------------------------------------------------------
 Function~  ICE::addExchangeToMomentumAndEnergyRF--
   This task adds the  exchange contribution to the 
   existing cell-centered momentum and internal energy
            
                   (A)                              (X)
| (1+b12 + b13)     -b12          -b23          |   |del_data_CC[1]  |    
|                                               |   |                |    
| -b21              (1+b21 + b23) -b32          |   |del_data_CC[2]  |    
|                                               |   |                | 
| -b31              -b32          (1+b31 + b32) |   |del_data_CC[2]  |

                        =
                        
                        (B)
| b12( data_CC[2] - data_CC[1] ) + b13 ( data_CC[3] -data_CC[1])    | 
|                                                                   |
| b21( data_CC[1] - data_CC[2] ) + b23 ( data_CC[3] -data_CC[2])    | 
|                                                                   |
| b31( data_CC[1] - data_CC[3] ) + b32 ( data_CC[2] -data_CC[3])    | 

 Steps for each cell;
    1) Comute the beta coefficients
    2) Form and A matrix and B vector
    3) Solve for X[*]
    4) Add X[*] to the appropriate Lagrangian data
 - apply Boundary conditions to vel_CC and Temp_CC

 References: see "A Cell-Centered ICE method for multiphase flow simulations"
 by Kashiwa, above equation 4.13.
 ---------------------------------------------------------------------  */
void ICE::addExchangeToMomentumAndEnergyRF(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing doCCMomExchange on patch "<< patch->getID()
               <<"\t\t\t ICE" << endl;

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numALLMatls = numMPMMatls + numICEMatls;
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
//    Vector zero(0.,0.,0.);
    // Create arrays for the grid data
    constCCVariable<double>  press_CC, delP_Dilatate;
    StaticArray<CCVariable<double> > Temp_CC(numALLMatls);  
    StaticArray<constCCVariable<double> > vol_frac_CC(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numALLMatls);
    StaticArray<constCCVariable<Vector> > mom_L(numALLMatls);
    StaticArray<constCCVariable<double> > int_eng_L(numALLMatls);

    // Create variables for the results
    StaticArray<CCVariable<Vector> > mom_L_ME(numALLMatls);
    StaticArray<CCVariable<Vector> > vel_CC(numALLMatls);
    StaticArray<CCVariable<double> > int_eng_L_ME(numALLMatls);
    StaticArray<CCVariable<double> > Tdot(numALLMatls);
    StaticArray<constCCVariable<double> > mass_L(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC(numALLMatls);
    StaticArray<constCCVariable<double> > old_temp(numALLMatls);
    StaticArray<constCCVariable<double> > f_theta(numALLMatls);
    StaticArray<constCCVariable<double> > speedSound(numALLMatls);
    StaticArray<constCCVariable<double> > int_eng_source(numALLMatls);
    
    vector<double> b(numALLMatls);
    vector<double> sp_vol(numALLMatls);
    vector<double> cv(numALLMatls);
    vector<double> X(numALLMatls);
    vector<double> e_prime_v(numALLMatls);
    vector<Vector> del_vel_CC(numALLMatls, Vector(0,0,0));
    vector<double> if_mpm_matl_ignore(numALLMatls);
    
    double tmp, sumBeta, alpha, term2, term3, kappa, sp_vol_source, delta_KE;
/*`==========TESTING==========*/
// I've included this term but it's turned off -Todd
    double Joule_coeff   = 0.0;         // measure of "thermal imperfection" 
/*==========TESTING==========`*/
    DenseMatrix beta(numALLMatls,numALLMatls),acopy(numALLMatls,numALLMatls);
    DenseMatrix K(numALLMatls,numALLMatls),H(numALLMatls,numALLMatls);
    DenseMatrix a(numALLMatls,numALLMatls), a_inverse(numALLMatls,numALLMatls);
    DenseMatrix phi(numALLMatls,numALLMatls);
    beta.zero();
    acopy.zero();
    K.zero();
    H.zero();
    a.zero();
 
    getExchangeCoefficients( K, H);
    Ghost::GhostType  gn = Ghost::None;
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      if_mpm_matl_ignore[m] = 1.0;
      if(mpm_matl){                 // M P M
        new_dw->get(old_temp[m],     lb->temp_CCLabel,     indx, patch,gn,0);
        new_dw->allocate(vel_CC[m],  MIlb->vel_CC_scratchLabel, indx, patch);
        new_dw->allocate(Temp_CC[m], MIlb->temp_CC_scratchLabel,indx, patch);
        cv[m] = mpm_matl->getSpecificHeat();
        if_mpm_matl_ignore[m] = 0.0;
      }
      if(ice_matl){                 // I C E
        old_dw->get(old_temp[m],    lb->temp_CCLabel,    indx, patch,gn,0);
        new_dw->allocate(vel_CC[m], lb->vel_CCLabel,     indx, patch);
        new_dw->allocate(Temp_CC[m],lb->temp_CCLabel,    indx, patch); 
        cv[m] = ice_matl->getSpecificHeat();
      }                             // A L L  M A T L S
      new_dw->get(mass_L[m],        lb->mass_L_CCLabel,    indx, patch,gn, 0); 
      new_dw->get(sp_vol_CC[m],     lb->sp_vol_CCLabel,    indx, patch,gn, 0);
      new_dw->get(mom_L[m],         lb->mom_L_CCLabel,     indx, patch,gn, 0); 
      new_dw->get(int_eng_L[m],     lb->int_eng_L_CCLabel, indx, patch,gn, 0); 
      new_dw->get(vol_frac_CC[m],   lb->vol_frac_CCLabel,  indx, patch,gn, 0);
      new_dw->get(speedSound[m],    lb->speedSound_CCLabel,indx, patch,gn, 0);
      new_dw->get(f_theta[m],       lb->f_theta_CCLabel,   indx, patch,gn, 0);
      new_dw->get(rho_CC[m],        lb->rho_CCLabel,       indx, patch,gn, 0);
      new_dw->get(int_eng_source[m],lb->int_eng_source_CCLabel,    
                                                           indx, patch,gn, 0);  
      new_dw->allocate( Tdot[m],    lb->Tdot_CCLabel,      indx, patch);        
      new_dw->allocate(mom_L_ME[m], lb->mom_L_ME_CCLabel,  indx, patch);       
      new_dw->allocate(int_eng_L_ME[m],lb->int_eng_L_ME_CCLabel,indx,patch);
      e_prime_v[m] = Joule_coeff * cv[m];
      Tdot[m].initialize(0.0);
    }
    new_dw->get(press_CC,           lb->press_CCLabel,     0,  patch,gn, 0);
    new_dw->get(delP_Dilatate,      lb->delP_DilatateLabel,0,  patch,gn, 0);       
    
    // Convert momenta to velocities and internal energy to Temp
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] = int_eng_L[m][c]/(mass_L[m][c]*cv[m]);
        vel_CC[m][c]  = mom_L[m][c]/mass_L[m][c];
      }
    }
    //---- P R I N T   D A T A ------ 
    if (switchDebugMomentumExchange_CC ) 
    {
      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc<<"TOP_addExchangeToMomentumAndEnergy_RF"<<indx<<"_patch_"
            <<patch->getID();
        printData(   patch,1, desc.str(),"Temp_CC",    Temp_CC[m]);     
        printData(   patch,1, desc.str(),"int_eng_L",  int_eng_L[m]);   
        printData(   patch,1, desc.str(),"mass_L",     mass_L[m]);      
        printVector( patch,1, desc.str(),"vel_CC", 0,  vel_CC[m]);            
      }
    }
    //---------- M O M E N T U M   E X C H A N G E                  
    //   Form BETA matrix (a), off diagonal terms                   
    //   beta and (a) matrix are common to all momentum exchanges   
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;

      for(int m = 0; m < numALLMatls; m++)  {
        tmp = sp_vol_CC[m][c];
        for(int n = 0; n < numALLMatls; n++) {
          beta[m][n] = delT * vol_frac_CC[n][c]  * K[n][m] * tmp;
          a[m][n]    = -beta[m][n];
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numALLMatls; m++) {
        a[m][m] = 1.0;
        for(int n = 0; n < numALLMatls; n++) {
          a[m][m] +=  beta[m][n];
        }
      }
      matrixInverse(numALLMatls, a, a_inverse);
      
      for (int dir = 0; dir <3; dir++) {  //loop over all three directons
        for(int m = 0; m < numALLMatls; m++) {
          b[m] = 0.0;
          for(int n = 0; n < numALLMatls; n++) {
           b[m] += beta[m][n] * (vel_CC[n][c](dir) - vel_CC[m][c](dir));
          }
        }
        
        multiplyMatrixAndVector(numALLMatls,a_inverse,b,X);
        
        for(int m = 0; m < numALLMatls; m++) {
          vel_CC[m][c](dir) =  vel_CC[m][c](dir) + X[m];
          del_vel_CC[m](dir) = X[m];
        }
      }
      //---------- C O M P U T E   T D O T ____________________________________   
      //  Reference: "Solution of \Delta T"
      //  For mpm matls ignore thermal expansion term alpha
      //  compute phi and alpha
      sumBeta = 0.0;
      for(int m = 0; m < numALLMatls; m++) {
        tmp = sp_vol_CC[m][c] / cv[m];
        for(int n = 0; n < numALLMatls; n++)  {
          beta[m][n]  = delT * vol_frac_CC[n][c] * H[n][m]*tmp;
          sumBeta    += beta[m][n];
          alpha       = if_mpm_matl_ignore[n] * 1.0/Temp_CC[n][c];
          phi[m][n]   = (f_theta[m][c] * vol_frac_CC[n][c]* alpha)/rho_CC[m][c]; 
        }
      }  
      //  off diagonal terms, matrix A    
      for(int m = 0; m < numALLMatls; m++) {
        for(int n = 0; n < numALLMatls; n++)  {
          a[m][n] = -( (1.0 - press_CC[c]) * phi[m][n] + beta[m][n]);
        }
      }
      //  diagonal terms, matrix A
      for(int m = 0; m < numALLMatls; m++) {
        term2   = ((1.0 + press_CC[c]) * phi[m][m] )/ f_theta[m][c];
        term3   =  (1.0 - press_CC[c]) * phi[m][m];
        a[m][m] = cv[m] + term2 - term3 + sumBeta - beta[m][m];
      }

      // -  F O R M   R H S   (b)         
      for(int m = 0; m < numALLMatls; m++)  {
        kappa         = sp_vol_CC[m][c]/(speedSound[m][c] * speedSound[m][c]);   
        sp_vol_source = -cell_vol * vol_frac_CC[m][c] * kappa*delP_Dilatate[c];
        delta_KE      = 0.5 * del_vel_CC[m].length() * del_vel_CC[m].length();     
        b[m]          = int_eng_source[m][c] - (e_prime_v[m] * sp_vol_source) 
                      - delta_KE;

        for(int n = 0; n < numALLMatls; n++) {
          b[m] += beta[m][n] * (Temp_CC[n][c] - Temp_CC[m][c]);
        }
      }
      //     S O L V E  and backout Tdot and Temp_CC
      matrixSolver(numALLMatls,a,b,X);
      
      for(int m = 0; m < numALLMatls; m++) {
        Tdot[m][c]    = X[m]/delT;
        Temp_CC[m][c] = Temp_CC[m][c] + X[m];
      }
    }  //CellIterator loop

    //__________________________________
    //  Set the Boundary conditions 
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int dwindex = matl->getDWIndex();
      setBC(vel_CC[m], "Velocity",   patch,dwindex);
      setBC(Temp_CC[m],"Temperature",patch, d_sharedState, dwindex);
    }
    //__________________________________
    // Convert vars. primitive-> flux 
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        int_eng_L_ME[m][c] = Temp_CC[m][c] * cv[m] * mass_L[m][c] 
                           + e_prime_v[m]  * sp_vol_source;
        mom_L_ME[m][c]     = vel_CC[m][c]          * mass_L[m][c];
      }
    }
    //---- P R I N T   D A T A ------ 
    if (switchDebugMomentumExchange_CC ) {
      for(int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc<<"addExchangeToMomentumAndEnergy_RF"<<indx<<"_patch_"
            <<patch->getID();
        printVector(patch,1, desc.str(), "mom_L_ME", 0,mom_L_ME[m]);
        printData(  patch,1, desc.str(),"int_eng_L_ME",int_eng_L_ME[m]);
        printData(  patch,1, desc.str(),"Tdot",        Tdot[m]);
      }
    }
    //__________________________________
    //    Put into new_dw
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->put(mom_L_ME[m],        lb->mom_L_ME_CCLabel,       indx,patch);
      new_dw->put(int_eng_L_ME[m],    lb->int_eng_L_ME_CCLabel,   indx,patch);   
      new_dw->put(Tdot[m],            lb->Tdot_CCLabel,           indx,patch);
    }  
  } //patches
}

/* ---------------------------------------------------------------------
 Function~  ICE::computeLagrangianSpecificVolumeRF--
 ---------------------------------------------------------------------  */
void ICE::computeLagrangianSpecificVolumeRF(const ProcessorGroup*,  
                                          const PatchSubset* patches,
                                          const MaterialSubset* /*matls*/,
                                          DataWarehouse* old_dw, 
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing computeLagrangianSpecificVolumeRF " <<
      patch->getID() << "\t\t ICE" << endl;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    int numALLMatls = d_sharedState->getNumMatls();
    Vector  dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();    

    StaticArray<constCCVariable<double> > Tdot(numALLMatls);
    StaticArray<constCCVariable<double> > vol_frac(numALLMatls);
    StaticArray<constCCVariable<double> > Temp_CC(numALLMatls);
    constCCVariable<double> rho_CC, rho_micro, f_theta,sp_vol_CC;
    constCCVariable<double> delP_Dilatate, press_CC, speedSound;
    CCVariable<double> sum_therm_exp;
    vector<double> if_mpm_matl_ignore(numALLMatls);

    new_dw->get(press_CC,        lb->press_CCLabel,      0,patch,Ghost::None,0);
    new_dw->get(delP_Dilatate,   lb->delP_DilatateLabel, 0,patch,Ghost::None,0);
    new_dw->allocate(sum_therm_exp,lb->SumThermExpLabel, 0,patch);

    sum_therm_exp.initialize(0.);
    //__________________________________
    // Sum of thermal expansion
    // alpha is hardwired for ideal gases
    // ignore contributions from mpm_matls
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      new_dw->get(Tdot[m],    lb->Tdot_CCLabel,    indx,patch,Ghost::None,0);  
      new_dw->get(vol_frac[m],lb->vol_frac_CCLabel,indx,patch,Ghost::None,0);  
      old_dw->get(Temp_CC[m], lb->temp_CCLabel,    indx,patch,Ghost::None,0); 

      if_mpm_matl_ignore[m] = 1.0; 
      if ( mpm_matl) {       
        if_mpm_matl_ignore[m] = 0.0; 
      } 
      for(CellIterator iter=patch->getExtraCellIterator();
                                                        !iter.done();iter++){
        IntVector c = *iter;
        double alpha =  if_mpm_matl_ignore[m] * 1.0/Temp_CC[m][c];
        sum_therm_exp[c] += vol_frac[m][c]*alpha*Tdot[m][c];
      }  
    }

    //__________________________________ 
    //  Compute spec_vol_L[m] = Mass[m] * sp_vol[m]
    //  this is consistent with 4.8c.
    //  
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      CCVariable<double> spec_vol_L, spec_vol_source;
      new_dw->allocate(spec_vol_L,     lb->spec_vol_L_CCLabel,     indx,patch);
      new_dw->allocate(spec_vol_source,lb->spec_vol_source_CCLabel,indx,patch);

      new_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,    indx,patch,Ghost::None, 0);
      spec_vol_source.initialize(0.);
      new_dw->get(rho_CC,    lb->rho_CCLabel,       indx,patch,Ghost::None, 0);
      new_dw->get(speedSound,lb->speedSound_CCLabel,indx,patch,Ghost::None, 0);
      new_dw->get(f_theta,   lb->f_theta_CCLabel,   indx,patch,Ghost::None, 0);

      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
       IntVector c = *iter;
       
       double kappa = sp_vol_CC[c]/(speedSound[c] * speedSound[c]); 
       double alpha = 1.0/Temp_CC[m][c];  // HARDWRIED FOR IDEAL GAS
       
       double term1 = -vol * vol_frac[m][c] * kappa * delP_Dilatate[c];
       double term2 = delT * vol * (vol_frac[m][c] * alpha *  Tdot[m][c] -
                                   f_theta[c] * sum_therm_exp[c]);
                                   
        // This is actually mass * sp_vol
       spec_vol_source[c] = term1 + if_mpm_matl_ignore[m] * term2;
 
       spec_vol_L[c] = (rho_CC[c] * vol)*sp_vol_CC[c] + spec_vol_source[c];
     }

      //  Set Neumann = 0 if symmetric Boundary conditions
      setBC(spec_vol_L, "set_if_sym_BC",patch, d_sharedState, indx); 

      //---- P R I N T   D A T A ------ 
      if (switchDebugLagrangianSpecificVol ) {
        ostringstream desc;
        desc <<"BOT_Lagrangian_VolRF_Mat_"<<indx<< "_patch_"<<patch->getID();
        printData(  patch,1, desc.str(), "Temp",          Temp_CC[m]);
        printData(  patch,1, desc.str(), "vol_frac[m]",   vol_frac[m]);
        printData(  patch,1, desc.str(), "rho_CC",        rho_CC);
        printData(  patch,1, desc.str(), "speedSound",    speedSound);
        printData(  patch,1, desc.str(), "sp_vol_CC",     sp_vol_CC);
        printData(  patch,1, desc.str(), "Tdot",          Tdot[m]);
        printData(  patch,1, desc.str(), "f_theta",       f_theta);
        printData(  patch,1, desc.str(), "delP_Dilatate", delP_Dilatate); 
        printData(  patch,1, desc.str(), "sum_therm_exp", sum_therm_exp);
        printData(  patch,1, desc.str(), "spec_vol_source",spec_vol_source);
        printData(  patch,1, desc.str(), "spec_vol_L",     spec_vol_L);
      }
      //____ B U L L E T   P R O O F I N G----
      IntVector neg_cell;
      if (!areAllValuesPositive(spec_vol_L, neg_cell)) {
        cout << "matl            "<< indx << endl;
        cout << "sum_thermal_exp "<< sum_therm_exp[neg_cell] << endl;
        cout << "delP_Dilatate   "<< delP_Dilatate[neg_cell] << endl;
        cout << "spec_vol_source "<< spec_vol_source[neg_cell] << endl;
        cout << "sp_vol_L        "<< spec_vol_L[neg_cell] << endl;
        cout << "mass "<< (rho_CC[neg_cell]*vol*sp_vol_CC[neg_cell]) << endl;
        ostringstream warn;
        warn<<"ERROR ICE::computeLagrangianSpecificVolumeRF, mat "<<indx
            << " cell " <<neg_cell << " spec_vol_L is negative\n";
        throw InvalidValue(warn.str());
     }  
     new_dw->put(spec_vol_L,     lb->spec_vol_L_CCLabel,     indx,patch);
     new_dw->put(spec_vol_source,lb->spec_vol_source_CCLabel,indx,patch);
    }  // end numALLMatl loop
  }  // patch loop
}
