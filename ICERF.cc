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
    new_dw->get(press_CC,lb->press_equil_CCLabel,0,patch,Ghost::AroundCells,1);

    for(int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      new_dw->get(rho_CC[m],       lb->rho_CCLabel,       indx,patch,
                                                          Ghost::AroundCells,1);
      new_dw->get(sp_vol_CC[m],    lb->sp_vol_CCLabel,    indx,patch,
                                                          Ghost::AroundCells,1);
      new_dw->get(speedSound[m],   lb->speedSound_CCLabel,indx,patch,
                                                          Ghost::AroundCells,1); 
      new_dw->get(matl_press[m],   lb->matl_press_CCLabel,indx,patch,
                                                          Ghost::AroundCells,1);
      if(ice_matl){
        old_dw->get(vel_CC[m],lb->vel_CCLabel,indx,patch,Ghost::AroundCells,1);      
      }
      if(mpm_matl){
        new_dw->get(vel_CC[m],lb->vel_CCLabel,indx,patch,Ghost::AroundCells,1);      
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
          IntVector cur = *iter;
          IntVector adj = cur + adj_offset[dir]; 

//__________________________________
//WARNING: We're currently using the 
// isentropic compressibility instead of 
// its cousin the isothermal compressiblity.  
          double inv_kappa_cur = ( speedSound[m][cur] * speedSound[m][cur] )/
                                 ( sp_vol_CC[m][cur]);
          double inv_kappa_adj = ( speedSound[m][adj] * speedSound[m][adj] )/
                                 ( sp_vol_CC[m][adj]);

          double rho_brack = (rho_CC[m][cur]*rho_CC[m][adj])/
                             (rho_CC[m][cur]+rho_CC[m][adj]);           
          
          double term1 = 
                (matl_press[m][cur] - press_CC[cur]) * sp_vol_CC[m][cur] +
                (matl_press[m][adj] - press_CC[adj]) * sp_vol_CC[m][adj];
                
          double term2 = -2.0 * delT *
                         ((sp_vol_CC[m][cur] + sp_vol_CC[m][adj])/
                         (inv_kappa_cur + inv_kappa_adj) ) *
                         ((vel_CC[m][adj](dir) - vel_CC[m][cur](dir))/dx(dir));
 
          scratch[dir][cur] = rho_brack * (term1 + term2);
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
    IntVector cur = *it;
    IntVector adj = cur + adj_offset; 
    sp_vol_brack = 2.*(sp_vol_CC[adj] * sp_vol_CC[cur])/
                      (sp_vol_CC[adj] + sp_vol_CC[cur]);
    //__________________________________
    // interpolation to the face           
    term1 = (vel_CC[adj](dir) * sp_vol_CC[cur] +
             vel_CC[cur](dir) * sp_vol_CC[adj])/
             (sp_vol_CC[cur] + sp_vol_CC[adj]);

    //__________________________________
    // pressure term
    term2 = sp_vol_brack * (delT/dx) * 
                           (press_CC[cur] - press_CC[adj]);
    //__________________________________
    // gravity term
    term3 =  delT * gravity;

    // stress difference term
    sig_L = vol_frac[adj] * (matl_press_CC[adj] - press_CC[adj]);
    sig_R = vol_frac[cur] * (matl_press_CC[cur] - press_CC[cur]);
    term4 = (sp_vol_brack * delT/dx) * (
            (sig_bar_FC[cur] - sig_L)/vol_frac[adj] +
            (sig_R - sig_bar_FC[cur])/vol_frac[cur]);

    // Todd, I think that term4 should be a negative, since Bucky
    // has a positive, but since sigma = -p*I.  What do you think?
    vel_FC[cur] = term1 - term2 + term3 - term4;
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
    new_dw->get(press_CC,lb->press_equil_CCLabel, 0,patch,Ghost::AroundCells,1);

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
        new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch, Ghost::AroundCells,1);
        old_dw->get(vel_CC, lb->vel_CCLabel, indx, patch, Ghost::AroundCells,1);
      } else {
        new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch, Ghost::AroundCells,1);
        new_dw->get(vel_CC, lb->vel_CCLabel, indx, patch, Ghost::AroundCells,1);
      }                                                  
      new_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,indx,patch,
                                                         Ghost::AroundCells, 1);
      new_dw->get(p_dXFC, lb->press_diffX_FCLabel, indx, patch,
                                                          Ghost::AroundCells,1);
      new_dw->get(p_dYFC, lb->press_diffY_FCLabel, indx, patch,
                                                          Ghost::AroundCells,1);
      new_dw->get(p_dZFC, lb->press_diffZ_FCLabel, indx, patch,
                                                          Ghost::AroundCells,1);
      new_dw->get(matl_press_CC,
                        lb->matl_press_CCLabel,indx,patch,Ghost::AroundCells,1);
      new_dw->get(vol_frac,
                        lb->vol_frac_CCLabel,  indx,patch,Ghost::AroundCells,1);

   //---- P R I N T   D A T A ------ 
      if (switchDebug_vel_FC ) {
#if 0
        ostringstream desc;
        desc << "TOP_vel_FC_Mat_" << indx << "_patch_" << patch->getID(); 
        printData(  patch, 1, desc.str(), "rho_CC",     rho_CC);
        printData(  patch, 1, desc.str(), "sp_vol_CC",  sp_vol_CC);
        printVector( patch,1, desc.str(), "uvel_CC", 0, vel_CC);
        printVector( patch,1, desc.str(), "vvel_CC", 1, vel_CC);
        printVector( patch,1, desc.str(), "wvel_CC", 2, vel_CC);
#endif
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

    new_dw->get(press_CC,        lb->press_CCLabel,      0,patch,Ghost::None,0);
    new_dw->get(delP_Dilatate,   lb->delP_DilatateLabel, 0,patch,Ghost::None,0);
    new_dw->allocate(sum_therm_exp,lb->SumThermExpLabel, 0,patch);

    sum_therm_exp.initialize(0.);

    // The following assumes that only the fluids have a thermal
    // expansivity, which is true at this time. (11-08-01)
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      if(ice_matl){
       int indx = matl->getDWIndex();
       new_dw->get(Tdot[m],     lb->Tdot_CCLabel,    indx,patch,Ghost::None, 0);
       new_dw->get(vol_frac[m], lb->vol_frac_CCLabel,indx,patch,Ghost::None, 0);
       old_dw->get(Temp_CC[m],  lb->temp_CCLabel,    indx,patch,Ghost::None, 0);
       for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
         IntVector c = *iter;
         // the following assumes an ideal gas and is getting alpha from the
         // average of the old temp and the new temp.
         double alpha = 1.0/Temp_CC[m][c];
         sum_therm_exp[c] += vol_frac[m][c]*alpha*Tdot[m][c];
       }
      }
    }

    //__________________________________ 
    //  Compute the Lagrangian specific volume
    for(int m = 0; m < numALLMatls; m++) {
     Material* matl = d_sharedState->getMaterial( m );
     int indx = matl->getDWIndex();
     ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
     CCVariable<double> spec_vol_L, spec_vol_source;
     new_dw->allocate(spec_vol_L,     lb->spec_vol_L_CCLabel,     indx,patch);
     new_dw->allocate(spec_vol_source,lb->spec_vol_source_CCLabel,indx,patch);
     spec_vol_L.initialize(0.);
     spec_vol_source.initialize(0.);
     if(ice_matl){
       new_dw->get(rho_CC,    lb->rho_CCLabel,       indx,patch,Ghost::None, 0);
       new_dw->get(speedSound,lb->speedSound_CCLabel,indx,patch,Ghost::None, 0);
       new_dw->get(f_theta,   lb->f_theta_CCLabel,   indx,patch,Ghost::None, 0);
       new_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,    indx,patch,Ghost::None, 0);

       for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        // Note that at this point, spec_vol_L is actually just a
        // volume of material, this is consistent with 4.8c

        double kappa = sp_vol_CC[c]/(speedSound[c] * speedSound[c]); 
        // the following assumes an ideal gas and is getting alpha from the
        // average of the old temp and the new temp.
        double alpha = 1.0/Temp_CC[m][c];

        double term1 = -vol * vol_frac[m][c] * kappa * delP_Dilatate[c];
        double term2 = delT * vol * (vol_frac[m][c]*alpha*Tdot[m][c] -
                                        f_theta[c]*sum_therm_exp[c]);

        spec_vol_source[c] = term1 + term2;
        spec_vol_L[c] = (rho_CC[c]*vol*sp_vol_CC[c]) + spec_vol_source[c];

#if 0
        if(spec_vol_L[c] < 0.){
          cout << "Cell = " << c << endl;
          cout << "vol  = " << vol << endl;
          cout << "sp_vol_CC[c] = " << sp_vol_CC[c] << endl;
          cout << "(rho_CC[c]*vol*sp_vol_CC[c]) = " << (rho_CC[c]*vol*sp_vol_CC[c]) << endl;
          cout << "spec_vol_L[c] = " << spec_vol_L[c] << endl;
          cout << "spec_vol_source[c] = " << spec_vol_source[c] << endl;
          cout << "vol_frac[m][c] = " << vol_frac[m][c] << endl;
          cout << "kappa = " << kappa << endl;
          cout << "delP_Dilatate[c] = " << delP_Dilatate[c] << endl;
          cout << "rho_CC[c] = " << rho_CC[c] << endl;
          cout << "Tdot = " << Tdot[m][c] << endl;
          cout << "f_theta[c] = " << f_theta[c] << endl;

        }
#endif
       }
     }
     new_dw->put(spec_vol_L,     lb->spec_vol_L_CCLabel,     indx,patch);
     new_dw->put(spec_vol_source,lb->spec_vol_source_CCLabel,indx,patch);
    }  // end numALLMatl loop
  }  // patch loop
}
