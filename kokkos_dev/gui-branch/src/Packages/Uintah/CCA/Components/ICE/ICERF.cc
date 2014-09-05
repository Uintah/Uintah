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
