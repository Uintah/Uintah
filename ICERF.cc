#include <stdio.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>

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
  t->requires(Task::NewDW,lb->rho_micro_CCLabel,      Ghost::AroundCells,1);
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
      old_dw->get(rho_CC[m],lb->rho_CC_top_cycleLabel,
                                                indx,patch,Ghost::None,0);
      old_dw->get(sp_vol_CC[m],
                             lb->sp_vol_CCLabel,indx,patch,Ghost::None,0);
      
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

         compressibility[m]=
                      ice_matl->getEOS()->getCompressibility(matl_press[m][c]);

         mat_volume[m] = (rho_CC[m][c] * cell_vol) * sp_vol_CC[m][c];

         tmp = dp_drho[m] + dp_de[m] * 
           (matl_press[m][c]/(rho_micro[m][c]*rho_micro[m][c]));               

        total_mat_vol += mat_volume[m];
        speedSound_new[m][c] = sqrt(tmp);
       } 
      //__________________________________
      // Compute 1/f_theta, rho_CC, 
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
    //  Set BCs on density, matl_press, press
    for (int m = 0; m < numMatls; m++)   {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      setBC(matl_press[m],rho_micro[SURROUND_MAT],"Pressure",patch,indx);
    }  
    setBC(press_new, rho_micro[SURROUND_MAT], "Pressure",patch,0);
    //__________________________________
    // carry rho_cc forward 
    // In MPMICE was compute rho_CC_new and 
    // therefore need the machinery here
    //__________________________________
    for (int m = 0; m < numMatls; m++)   {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      rho_CC_new[m].copyData(rho_CC[m]);
      new_dw->put( vol_frac[m],      lb->vol_frac_CCLabel,   indx, patch);
      new_dw->put( f_theta[m],       lb->f_theta_CCLabel,    indx, patch);
      new_dw->put( matl_press[m],    lb->matl_press_CCLabel, indx, patch);
      new_dw->put( speedSound_new[m],lb->speedSound_CCLabel, indx, patch);
      new_dw->put( rho_micro[m],     lb->rho_micro_CCLabel,  indx, patch);
      new_dw->put( rho_CC_new[m],    lb->rho_CCLabel,        indx, patch);
    }
    new_dw->put(press_new,lb->press_equil_CCLabel,0,patch);

  } // patches

}
/* --------------------------------------------------------------------- 
 Function~  ICE::computeFCPressDiffRF-- 
 Reference: A Multifield Model and Method for Fluid Structure
            Interaction Dynamics
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

    StaticArray<constCCVariable<double> > rho_CC(numMatls);
    StaticArray<constCCVariable<double> > rho_micro(numMatls);
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
      new_dw->get(rho_micro[m],    lb->rho_micro_CCLabel, indx,patch,
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
    }
    //______________________________________________________________________
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      double Kcur, Kadj;
      if(mpm_matl){
        Kcur = mpm_matl->getConstitutiveModel()->getCompressibility();
        Kadj = mpm_matl->getConstitutiveModel()->getCompressibility();
      }
      //__________________________________
      //  B O T T O M   F A C E
      for(CellIterator iter=patch->getSFCYIterator();!iter.done();iter++){
        IntVector curcell = *iter;
        IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z());
        double rho_brack     = (rho_CC[m][curcell]*rho_CC[m][adjcell])/
                               (rho_CC[m][curcell]+rho_CC[m][adjcell]);
        if(ice_matl){
          Kcur = ice_matl->getEOS()->getCompressibility(press_CC[curcell]);
          Kadj = ice_matl->getEOS()->getCompressibility(press_CC[adjcell]);
        }
        double deltaP = 2.*delT*(vel_CC[m][adjcell].y()-vel_CC[m][curcell].y())/
                        ((Kcur + Kadj)*dx.y());
        press_diffY_FC[m][curcell] = rho_brack*
             ((matl_press[m][curcell]-press_CC[curcell])/rho_micro[m][curcell] +
              (matl_press[m][adjcell]-press_CC[adjcell])/rho_micro[m][adjcell] +
              (1./rho_micro[m][curcell] + 1./rho_micro[m][adjcell])*deltaP);
      }
      //__________________________________
      //  L E F T   F A C E
      for(CellIterator iter=patch->getSFCXIterator();!iter.done();iter++){
        IntVector curcell = *iter;
        IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z());

        double  rho_brack   = (rho_CC[m][curcell]*rho_CC[m][adjcell])/
                              (rho_CC[m][curcell]+rho_CC[m][adjcell]);
        if(ice_matl){
          Kcur = ice_matl->getEOS()->getCompressibility(press_CC[curcell]);
          Kadj = ice_matl->getEOS()->getCompressibility(press_CC[adjcell]);
        }
        double deltaP = 2.*delT*(vel_CC[m][adjcell].x()-vel_CC[m][curcell].x())/
                        ((Kcur + Kadj)*dx.x());
        press_diffX_FC[m][curcell] = rho_brack*
             ((matl_press[m][curcell]-press_CC[curcell])/rho_micro[m][curcell] +
              (matl_press[m][adjcell]-press_CC[adjcell])/rho_micro[m][adjcell] +
              (1./rho_micro[m][curcell] + 1./rho_micro[m][adjcell])*deltaP);
      }
      //__________________________________
      //     B A C K   F A C E
      for(CellIterator iter=patch->getSFCZIterator();!iter.done();iter++){
        IntVector curcell = *iter;
        IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1);
        double rho_brack     = (rho_CC[m][curcell]*rho_CC[m][adjcell])/
                               (rho_CC[m][curcell]+rho_CC[m][adjcell]);
        if(ice_matl){
          Kcur = ice_matl->getEOS()->getCompressibility(press_CC[curcell]);
          Kadj = ice_matl->getEOS()->getCompressibility(press_CC[adjcell]);
        }
        double deltaP = 2.*delT*(vel_CC[m][adjcell].z()-vel_CC[m][curcell].z())/
                        ((Kcur + Kadj)*dx.z());
        press_diffZ_FC[m][curcell] = rho_brack*
             ((matl_press[m][curcell]-press_CC[curcell])/rho_micro[m][curcell] +
              (matl_press[m][adjcell]-press_CC[adjcell])/rho_micro[m][adjcell] +
              (1./rho_micro[m][curcell] + 1./rho_micro[m][adjcell])*deltaP);
      }
    }
    for(int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->put(press_diffX_FC[m],lb->press_diffX_FCLabel, indx, patch);
      new_dw->put(press_diffY_FC[m],lb->press_diffY_FCLabel, indx, patch);
      new_dw->put(press_diffZ_FC[m],lb->press_diffZ_FCLabel, indx, patch);
    }
  } // patches
}
/* ---------------------------------------------------------------------
 Function~  ICE::computeFaceCenteredVelocitiesRF--
 Purpose~   compute the face centered velocities minus the exchange
            contribution.  See Kashiwa Feb. 2001, 4.10a for description
            of term1-term4.
_____________________________________________________________________*/
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
      constCCVariable<double> rho_CC, rho_micro_CC,vol_frac;
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
      new_dw->get(rho_micro_CC, lb->rho_micro_CCLabel,indx,patch,
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
        ostringstream description;
        description << "TOP_vel_FC_Mat_" << indx << "_patch_" 
                    << patch->getID(); 
        printData(  patch, 1, description.str(), "rho_CC",      rho_CC);
        printData(  patch, 1, description.str(), "rho_micro_CC",rho_micro_CC);
        printVector( patch,1, description.str(), "uvel_CC", 0, vel_CC);
        printVector( patch,1, description.str(), "vvel_CC", 1, vel_CC);
        printVector( patch,1, description.str(), "wvel_CC", 2, vel_CC);
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

      double term1, term2, term3, term4, sp_vol_brack, sig_L, sig_R;
      
      if(doMechOld < -1.5){
      //__________________________________
      //   B O T T O M   F A C E S 
      int offset=1; // 0=Compute all faces in computational domain
                    // 1=Skip the faces at the border between interior and gc
      for(CellIterator iter=patch->getSFCYIterator(offset);!iter.done();iter++){
         IntVector curcell = *iter;
         IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z()); 

         sp_vol_brack = 2.*(1./(rho_micro_CC[adjcell]*rho_micro_CC[curcell]))/
                           (1./rho_micro_CC[adjcell]+1./rho_micro_CC[curcell]);
         //__________________________________
         // interpolation to the face
         term1 = (vel_CC[adjcell].y()/rho_micro_CC[curcell] +
                  vel_CC[curcell].y()/rho_micro_CC[adjcell])/
                 (rho_micro_CC[curcell] + rho_micro_CC[adjcell]);
         //__________________________________
         // pressure term
         term2 = sp_vol_brack*(delT/dx.y()) * 
                                (press_CC[curcell] - press_CC[adjcell]);
         //__________________________________
         // gravity term
         term3 =  delT * gravity.y();

         // stress difference term
         sig_L = vol_frac[adjcell]*(matl_press_CC[adjcell] - press_CC[adjcell]);
         sig_R = vol_frac[curcell]*(matl_press_CC[curcell] - press_CC[curcell]);
         term4 = (sp_vol_brack*delT/dx.y())*(
                 (p_dYFC[curcell] - sig_L)/vol_frac[adjcell] +
                 (sig_R - p_dYFC[curcell])/vol_frac[curcell]);

         // Todd, I think that term4 should be a negative, since Bucky
         // has a positive, but since sigma = -p*I.  What do you think?
         vvel_FC[curcell] = term1 - term2 + term3 - term4;
      }

      //__________________________________
      //  L E F T   F A C E 
      for(CellIterator iter=patch->getSFCXIterator(offset);!iter.done();iter++){
         IntVector curcell = *iter;
         IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z()); 
         //__________________________________
         // interpolation to the face
         term1 = (vel_CC[adjcell].x()/rho_micro_CC[curcell] +
                  vel_CC[curcell].x()/rho_micro_CC[adjcell])/
                 (rho_micro_CC[curcell] + rho_micro_CC[adjcell]);
         //__________________________________
         // pressure term
         term2 = sp_vol_brack*(delT/dx.x()) * 
                                (press_CC[curcell] - press_CC[adjcell]);
         //__________________________________
         // gravity term
         term3 =  delT * gravity.x();

         // stress difference term
         sig_L = vol_frac[adjcell]*(matl_press_CC[adjcell] - press_CC[adjcell]);
         sig_R = vol_frac[curcell]*(matl_press_CC[curcell] - press_CC[curcell]);
         term4 = (sp_vol_brack*delT/dx.x())*(
                 (p_dXFC[curcell] - sig_L)/vol_frac[adjcell] +
                 (sig_R - p_dXFC[curcell])/vol_frac[curcell]);

         // Todd, I think that term4 should be a negative, since Bucky
         // has a positive, but since sigma = -p*I.  What do you think?
         uvel_FC[curcell] = term1 - term2 + term3 - term4;
      }
      
      //__________________________________
      //  B A C K    F A C E
      for(CellIterator iter=patch->getSFCZIterator(offset);!iter.done();iter++){
         IntVector curcell = *iter;
         IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1); 
         //__________________________________
         // interpolation to the face
         term1 = (vel_CC[adjcell].z()/rho_micro_CC[curcell] +
                  vel_CC[curcell].z()/rho_micro_CC[adjcell])/
                 (rho_micro_CC[curcell] + rho_micro_CC[adjcell]);
         //__________________________________
         // pressure term
         term2 = sp_vol_brack*(delT/dx.z()) * 
                                (press_CC[curcell] - press_CC[adjcell]);
         //__________________________________
         // gravity term
         term3 =  delT * gravity.z();

         // stress difference term
         sig_L = vol_frac[adjcell]*(matl_press_CC[adjcell] - press_CC[adjcell]);
         sig_R = vol_frac[curcell]*(matl_press_CC[curcell] - press_CC[curcell]);
         term4 = (sp_vol_brack*delT/dx.z())*(
                 (p_dZFC[curcell] - sig_L)/vol_frac[adjcell] +
                 (sig_R - p_dZFC[curcell])/vol_frac[curcell]);

         // Todd, I think that term4 should be a negative, since Bucky
         // has a positive, but since sigma = -p*I.  What do you think?
         wvel_FC[curcell] = term1 - term2 + term3 - term4;
      }
      }  // if doMech

      //__________________________________
      // (*)vel_FC BC are updated in 
      // ICE::addExchangeContributionToFCVel()

      new_dw->put(uvel_FC, lb->uvel_FCLabel, indx, patch);
      new_dw->put(vvel_FC, lb->vvel_FCLabel, indx, patch);
      new_dw->put(wvel_FC, lb->wvel_FCLabel, indx, patch);

   //---- P R I N T   D A T A ------ 
      if (switchDebug_vel_FC ) {
        char description[50];
        sprintf(description, "bottom_of_vel_FC_Mat_%d_patch_%d ",
                indx, patch->getID());
        printData_FC( patch,1, description, "uvel_FC", uvel_FC);
        printData_FC( patch,1, description, "vvel_FC", vvel_FC);
        printData_FC( patch,1, description, "wvel_FC", wvel_FC);
      }
    } // matls loop
  }  // patch loop
}

#if 0
/*---------------------------------------------------------------------
 Function~  ICE::computeDelPressAndUpdatePressCCRF--
 Purpose~
   This function calculates the change in pressure explicitly. 
 Note:  Units of delp_Dilatate and delP_MassX are [Pa]
 Reference:  Multimaterial Formalism eq. 1.5
 ---------------------------------------------------------------------  */
void ICE::computeDelPressAndUpdatePressCCRF(const ProcessorGroup*,  
					  const PatchSubset* patches,
					  const MaterialSubset* /*matls*/,
					  DataWarehouse* old_dw, 
					  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << "Doing explicit delPress on patch (RF) " << patch->getID() 
         <<  "\t\t\t ICE" << endl;

    int numMatls  = d_sharedState->getNumMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx     = patch->dCell();

    double vol    = dx.x()*dx.y()*dx.z();
    double invvol = 1./vol;
    double compressibility;

    CCVariable<double> q_CC,      q_advected;
    Advector* advector = d_advector->clone(new_dw,patch);

    constCCVariable<double> pressure;
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> press_CC;
    constCCVariable<double> burnedMass;
   
    const IntVector gc(1,1,1);

    new_dw->get(pressure,       lb->press_equil_CCLabel,0,patch,Ghost::None,0);
    new_dw->allocate(delP_Dilatate,lb->delP_DilatateLabel,0, patch);
    new_dw->allocate(delP_MassX,lb->delP_MassXLabel,    0, patch);
    new_dw->allocate(press_CC,  lb->press_CCLabel,      0, patch);
    new_dw->allocate(q_CC,      lb->q_CCLabel,          0, patch,
		     Ghost::AroundCells,1);
    new_dw->allocate(q_advected,lb->q_advectedLabel,    0, patch);

    StaticArray<constCCVariable<double> > rho_micro(numMatls);

    CCVariable<double> term1, term2, term3;
    new_dw->allocate(term1, lb->term1Label, 0, patch);
    new_dw->allocate(term2, lb->term2Label, 0, patch);
    new_dw->allocate(term3, lb->term3Label, 0, patch);

    term1.initialize(0.);
    term2.initialize(0.);
    term3.initialize(0.);
    delP_Dilatate.initialize(0.0);
    delP_MassX.initialize(0.0);

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
//      constCCVariable<double>   speedSound;
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;
      constCCVariable<double> vol_frac;
      constCCVariable<Vector> vel_CC;
      constCCVariable<double> rho_CC;

      new_dw->get(uvel_FC, lb->uvel_FCMELabel,   indx,  patch,
		  Ghost::AroundCells, 2);
      new_dw->get(vvel_FC, lb->vvel_FCMELabel,   indx,  patch,
		  Ghost::AroundCells, 2);
      new_dw->get(wvel_FC, lb->wvel_FCMELabel,   indx,  patch,
		  Ghost::AroundCells, 2);
      new_dw->get(vol_frac,lb->vol_frac_CCLabel, indx,  patch,
		  Ghost::AroundCells,1);
      new_dw->get(rho_CC,      lb->rho_CCLabel,      indx,patch,Ghost::None,0);
      new_dw->get(rho_micro[m],lb->rho_micro_CCLabel,indx,patch,Ghost::None,0);
//      new_dw->get(speedSound, lb->speedSound_CCLabel,indx,patch,Ghost::None,0);
      new_dw->get(burnedMass, lb->burnedMass_CCLabel,indx,patch,Ghost::None,0);
      if(ice_matl) {
        old_dw->get(vel_CC,   lb->vel_CCLabel,       indx,patch,Ghost::None,0);
      }
      if(mpm_matl) {
        new_dw->get(vel_CC,   lb->vel_CCLabel,       indx,patch,Ghost::None,0);
      }

      //__________________________________
      // Advection preprocessing
      // - divide vol_frac_cc/vol


      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch);

      for(CellIterator iter = patch->getCellIterator(gc); !iter.done();
	  iter++) {
	IntVector c = *iter;
        q_CC[c] = vol_frac[c] * invvol;
      }
      //__________________________________
      //   First order advection of q_CC
      advector->advectQ(q_CC,patch,q_advected);


      //---- P R I N T   D A T A ------  
      if (switchDebug_explicit_press ) {
        ostringstream description;
	description << "middle_of_explicit_Pressure_Mat_" << indx << "_patch_"
		    <<  patch->getID();
        printData_FC( patch,1, description.str(), "uvel_FC", uvel_FC);
        printData_FC( patch,1, description.str(), "vvel_FC", vvel_FC);
        printData_FC( patch,1, description.str(), "wvel_FC", wvel_FC);
      }
      
      if(mpm_matl) {
           compressibility =
                mpm_matl->getConstitutiveModel()->getCompressibility();
      }  
                    
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
	IntVector c = *iter;
        //__________________________________
        //   Contributions from reactions
        //   to be filled in Be very careful with units
        term1[c] += (burnedMass[c]/delT)/(rho_micro[m][c] * vol);

        //__________________________________
        //   Divergence of velocity * face area
        //   Be very careful with the units
        //   do the volume integral to check them
        //   See journal pg 171
        //   You need to divide by the cell volume
        //
        //  Note that sum(div (theta_k U^f_k) 
        //          =
        //  Advection(theta_k, U^f_k)
        //  This subtle point is discussed on pg
        //  190 of my Journal
        term2[c] -= q_advected[c];

         if(ice_matl) {
            compressibility =
			ice_matl->getEOS()->getCompressibility(pressure[*iter]);
        }

        // New term3 comes from Kashiwa 4.11
        term3[*iter] += vol_frac[*iter]*compressibility;
      }  //iter loop
    }  //matl loop
    delete advector;
    press_CC.initialize(0.);
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      delP_MassX[c]    = (delT * term1[c])/term3[c];
      delP_Dilatate[c] = -term2[c]/term3[c];
      press_CC[c]      = pressure[c] + 
                             delP_MassX[c] + delP_Dilatate[c];    
    }
    setBC(press_CC, rho_micro[SURROUND_MAT], "Pressure",patch,0);

    new_dw->put(delP_Dilatate, lb->delP_DilatateLabel, 0, patch);
    new_dw->put(delP_MassX,    lb->delP_MassXLabel,    0, patch);
    new_dw->put(press_CC,      lb->press_CCLabel,      0, patch);

   //---- P R I N T   D A T A ------  
    if (switchDebug_explicit_press) {
      ostringstream description;
      description << "Bottom_of_explicit_Pressure_patch_" << patch->getID();
      printData( patch, 1,description.str(), "delP_Dilatate", delP_Dilatate);
      //printData( patch, 1,description.str(), "delP_MassX",    delP_MassX);
      printData( patch, 1,description.str(), "Press_CC",      press_CC);
    }
  }  // patch loop
}
#endif
/* ---------------------------------------------------------------------
 Function~  ICE::accumulateMomentumSourceSinksRF--
 Purpose~   This function accumulates all of the sources/sinks of momentum
 ---------------------------------------------------------------------  */
void ICE::accumulateMomentumSourceSinksRF(const ProcessorGroup*,  
                                        const PatchSubset* patches,
                                        const MaterialSubset* /*matls*/,
                                        DataWarehouse* old_dw, 
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing accumulate_momentum_source_sinks_RF on patch " <<
      patch->getID() << "\t ICE" << endl;

    int indx;
    int numMatls  = d_sharedState->getNumMatls();

    IntVector right, left, top, bottom, front, back;
    Vector dx, gravity;
    double pressure_source, mass, vol;
    double viscous_source;
    double press_diff_source;
    double viscosity;
    double include_term;

    delt_vartype delT; 
    old_dw->get(delT, d_sharedState->get_delt_label());
    delt_vartype doMechOld;
    old_dw->get(doMechOld, lb->doMechLabel);
 
    dx      = patch->dCell();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    gravity = d_sharedState->getGravity();
    vol     = delX * delY * delZ;
    constCCVariable<double>   rho_CC;
    constCCVariable<Vector>   vel_CC;
    constCCVariable<double>   vol_frac;
    constSFCXVariable<double> pressX_FC,press_diffX_FC;
    constSFCYVariable<double> pressY_FC,press_diffY_FC;
    constSFCZVariable<double> pressZ_FC,press_diffZ_FC;

    new_dw->get(pressX_FC,lb->pressX_FCLabel, 0, patch,Ghost::AroundCells, 1);
    new_dw->get(pressY_FC,lb->pressY_FCLabel, 0, patch,Ghost::AroundCells, 1);
    new_dw->get(pressZ_FC,lb->pressZ_FCLabel, 0, patch,Ghost::AroundCells, 1);

  //__________________________________
  //  Matl loop 
    for(int m = 0; m < numMatls; m++) {
      Material* matl        = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      indx = matl->getDWIndex();
      new_dw->get(rho_CC,  lb->rho_CCLabel,        indx,patch,Ghost::None, 0);
      new_dw->get(vol_frac,lb->vol_frac_CCLabel,   indx,patch,Ghost::None, 0);
      new_dw->get(press_diffX_FC,
                   lb->press_diffX_FCLabel,indx,patch,Ghost::AroundCells, 1);
      new_dw->get(press_diffY_FC,
                   lb->press_diffY_FCLabel,indx,patch,Ghost::AroundCells, 1);
      new_dw->get(press_diffZ_FC,
                   lb->press_diffZ_FCLabel,indx,patch,Ghost::AroundCells, 1);

      CCVariable<Vector>   mom_source;
      new_dw->allocate(mom_source,  lb->mom_source_CCLabel,  indx, patch);
      mom_source.initialize(Vector(0.,0.,0.));

      if(doMechOld < -1.5){
      //__________________________________
      // Compute Viscous Terms 
      SFCXVariable<Vector> tau_X_FC;
      SFCYVariable<Vector> tau_Y_FC;
      SFCZVariable<Vector> tau_Z_FC;  
      // note tau_*_FC is the same size as press(*)_FC
      tau_X_FC.allocate(pressX_FC.getLowIndex(), pressX_FC.getHighIndex());
      tau_Y_FC.allocate(pressY_FC.getLowIndex(), pressY_FC.getHighIndex());
      tau_Z_FC.allocate(pressZ_FC.getLowIndex(), pressZ_FC.getHighIndex());
      
      tau_X_FC.initialize(Vector(0.,0.,0.));
      tau_Y_FC.initialize(Vector(0.,0.,0.));
      tau_Z_FC.initialize(Vector(0.,0.,0.));
      viscosity = 0.0;
      if(ice_matl){
        old_dw->get(vel_CC, lb->vel_CCLabel,  indx,patch,Ghost::None, 0);
        viscosity = ice_matl->getViscosity();
        if(viscosity != 0.0){  
          computeTauX_Components( patch, vel_CC, viscosity, dx, tau_X_FC);
          computeTauY_Components( patch, vel_CC, viscosity, dx, tau_Y_FC);
          computeTauZ_Components( patch, vel_CC, viscosity, dx, tau_Z_FC);
        }
        include_term = 1.0;
        // This multiplies terms that are only included in the ice_matls
      }
      if(mpm_matl){
        include_term = 0.0;
      }
      //__________________________________
      //  accumulate sources
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        mass = rho_CC[c] * vol;
        right    = c + IntVector(1,0,0);
        left     = c + IntVector(0,0,0);
        top      = c + IntVector(0,1,0);
        bottom   = c + IntVector(0,0,0);
        front    = c + IntVector(0,0,1);
        back     = c + IntVector(0,0,0);

        // TODD:  Note that here, as in computeFCVelocities, I'm putting
        // a negative sign in front of press_diff_source, since sigma=-p*I



        //__________________________________
        //    X - M O M E N T U M 
        pressure_source = (pressX_FC[right]-pressX_FC[left]) * vol_frac[c];

        press_diff_source = (press_diffX_FC[right]-press_diffX_FC[left]);

        viscous_source = (tau_X_FC[right].x() - tau_X_FC[left].x())  *delY*delZ +
                         (tau_Y_FC[top].x()   - tau_Y_FC[bottom].x())*delX*delZ +
                         (tau_Z_FC[front].x() - tau_Z_FC[back].x())  *delX*delY;

        mom_source[c].x( (-pressure_source * delY * delZ +
                               viscous_source -
                               press_diff_source * delY * delZ * include_term +
                               mass * gravity.x() * include_term) * delT);
        //__________________________________
        //    Y - M O M E N T U M
        pressure_source = (pressY_FC[top]-pressY_FC[bottom])* vol_frac[c];

        press_diff_source = (press_diffY_FC[top]-press_diffY_FC[bottom]);

        viscous_source = (tau_X_FC[right].y() - tau_X_FC[left].y())  *delY*delZ+
                         (tau_Y_FC[top].y()   - tau_Y_FC[bottom].y())*delX*delZ+
                         (tau_Z_FC[front].y() - tau_Z_FC[back].y())  *delX*delY;

        mom_source[c].y( (-pressure_source * delX * delZ +
                               viscous_source -
                               press_diff_source * delX * delZ * include_term +
                               mass * gravity.y() * include_term) * delT );
        //__________________________________
        //    Z - M O M E N T U M
        pressure_source = (pressZ_FC[front]-pressZ_FC[back]) * vol_frac[c];
        
        press_diff_source = (press_diffZ_FC[front]-press_diffZ_FC[back]);

        viscous_source = (tau_X_FC[right].z() - tau_X_FC[left].z())  *delY*delZ+
                         (tau_Y_FC[top].z()   - tau_Y_FC[bottom].z())*delX*delZ+
                         (tau_Z_FC[front].z() - tau_Z_FC[back].z())  *delX*delY;

        mom_source[c].z( (-pressure_source * delX * delY +
                               viscous_source - 
                               press_diff_source * delX * delY * include_term +
                               mass * gravity.z() * include_term) * delT );
                               
      }
      } // if doMechOld
      new_dw->put(mom_source, lb->mom_source_CCLabel, indx, patch);
      new_dw->put(doMechOld,  lb->doMechLabel);

      //---- P R I N T   D A T A ------ 
      if (switchDebugSource_Sink) {
        ostringstream description;
         description << "sources/sinks_Mat_" << indx << "_patch_" 
                    <<  patch->getID();
        printVector(patch, 1, description.str(), "xmom_source", 0, mom_source);
        printVector(patch, 1, description.str(), "ymom_source", 1, mom_source);
        printVector(patch, 1, description.str(), "zmom_source", 2, mom_source);
      }
    }
  }
}

/* --------------------------------------------------------------------- 
 Function~  ICE::accumulateEnergySourceSinksRF--
 Purpose~   This function accumulates all of the sources/sinks of energy 
 Currently the kinetic energy isn't included.
 ---------------------------------------------------------------------  */
void ICE::accumulateEnergySourceSinksRF(const ProcessorGroup*,  
                                      const PatchSubset* patches,
                                  const MaterialSubset* /*matls*/,
                                      DataWarehouse* old_dw, 
                                      DataWarehouse* new_dw)
{
#ifdef ANNULUSICE
  static int n_iter;
  n_iter ++;
#endif

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing accumulate_energy_source_sinksRF on patch " 
         << patch->getID() << "\t ICE" << endl;

    int numMatls = d_sharedState->getNumMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx = patch->dCell();
    double A, B, vol=dx.x()*dx.y()*dx.z();

    constCCVariable<double> rho_micro_CC;
    constCCVariable<double> speedSound;
    constCCVariable<double> vol_frac;
    constCCVariable<double> press_CC;
    constCCVariable<double> delP_Dilatate;

    new_dw->get(press_CC,     lb->press_CCLabel,      0, patch,Ghost::None, 0);
    new_dw->get(delP_Dilatate,lb->delP_DilatateLabel, 0, patch,Ghost::None, 0);

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx    = matl->getDWIndex();   
      CCVariable<double> int_eng_source;
      new_dw->get(rho_micro_CC,lb->rho_micro_CCLabel, indx,patch,Ghost::None,0);
      new_dw->get(speedSound,  lb->speedSound_CCLabel,indx,patch,Ghost::None,0);
      new_dw->get(vol_frac,    lb->vol_frac_CCLabel,  indx,patch,Ghost::None,0);

#ifdef ANNULUSICE
      CCVariable<double> rho_CC;
      new_dw->get(rho_CC,      lb->rho_CCLabel,       indx,patch,Ghost::None,0);                                            
#endif
      new_dw->allocate(int_eng_source,  lb->int_eng_source_CCLabel, indx,patch);

      //__________________________________
      //   Compute source from volume dilatation
      //   Exclude contribution from delP_MassX
      int_eng_source.initialize(0.);
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
         IntVector c = *iter;
        A = vol * vol_frac[c] * press_CC[c];
        B = rho_micro_CC[c]   * speedSound[c] * speedSound[c];
        int_eng_source[c] = (A/B) * delP_Dilatate[c];
      }

#ifdef ANNULUSICE
      if(n_iter <= 1.e10){
        if(m==2){
          for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
             IntVector c = *iter;
            int_eng_source[c] += 1.e10 * delT * rho_CC[c] * vol;
          }
        }
      }
#endif

      //---- P R I N T   D A T A ------ 
      if (switchDebugSource_Sink) {
        ostringstream description;
        description <<  "sources/sinks_Mat_" << indx << "_patch_" 
                    <<  patch->getID();
        printData(patch,1,description.str(),"int_eng_source", int_eng_source);
      }

      new_dw->put(int_eng_source, lb->int_eng_source_CCLabel, indx,patch);
    }  // matl loop
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

    constCCVariable<double> rho_CC, rho_micro, Tdot;
    constCCVariable<double> delP_Dilatate, press_CC;
    constCCVariable<double> f_theta,vol_frac, Temp_CC;
    CCVariable<double> sum_therm_exp;

    new_dw->get(press_CC,        lb->press_CCLabel,      0,patch,Ghost::None,0);
    new_dw->get(delP_Dilatate,   lb->delP_DilatateLabel, 0,patch,Ghost::None,0);
    new_dw->allocate(sum_therm_exp,lb->SumThermExpLabel, 0,patch);

#if 0
    double alpha;
    // The following assumes that only the fluids have a thermal
    // expansivity, which is true at this time. (11-08-01)
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      if(ice_matl){
        int indx = matl->getDWIndex();
        new_dw->get(Tdot,      lb->Tdot_CCLabel,     indx,patch,Ghost::None, 0);
        new_dw->get(rho_micro, lb->rho_micro_CCLabel,indx,patch,Ghost::None, 0);
        new_dw->get(vol_frac,  lb->vol_frac_CCLabel, indx,patch,Ghost::None, 0);
        old_dw->get(Temp_CC,   lb->temp_CCLabel,     indx,patch,Ghost::None, 0);
        for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          alpha = 1.0/Temp_CC[c];  // this assumes an ideal gas
          sum_therm_exp[c] += vol_frac[c]*alpha*Tdot[c];
        }
      }
    }
#endif

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
       new_dw->get(rho_CC,   lb->rho_CCLabel,      indx,patch,Ghost::None, 0);
       new_dw->get(rho_micro,lb->rho_micro_CCLabel,indx,patch,Ghost::None, 0);
       new_dw->get(vol_frac, lb->vol_frac_CCLabel, indx,patch,Ghost::None, 0);
       new_dw->get(Tdot,     lb->Tdot_CCLabel,     indx,patch,Ghost::None, 0);
       new_dw->get(f_theta,  lb->f_theta_CCLabel,  indx,patch,Ghost::None, 0);

       for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        // Note that at this point, spec_vol_L is actually just a
        // volume of material, this is consistent with 4.8c
        double K = ice_matl->getEOS()->getCompressibility(press_CC[c]);
        double term1 = -vol * vol_frac[c] * K * delP_Dilatate[c];
#if 0
        alpha = 1.0/Temp_CC[c];  // this assumes an ideal gas
        double term2 = delT * vol * (vol_frac[c]*alpha*Tdot[c] -
                                        vol_frac[c]*sum_therm_exp[c]);
        spec_vol_source[c] = term1 + term2;
        spec_vol_source[c] = term1;
#endif
        spec_vol_source[c] = term1;
        spec_vol_L[c] = (rho_CC[c]*vol)/rho_micro[c] + term1;
       }
     }
     new_dw->put(spec_vol_L,     lb->spec_vol_L_CCLabel,     indx,patch);
     new_dw->put(spec_vol_source,lb->spec_vol_source_CCLabel,indx,patch);
    }  // end numALLMatl loop
  }  // patch loop
}
