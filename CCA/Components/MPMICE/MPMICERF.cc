
/* --------------------------------------------------------------------- 
 Function~  MPMICE::computeRateFormPressure-- 
 Reference: A Multifield Model and Method for Fluid Structure
            Interaction Dynamics
_____________________________________________________________________*/
void MPMICE::computeRateFormPressure(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing<<"Doing computeRateFormPressure on patch "
              << patch->getID() <<"\t\t MPMICE" << endl;

    double    tmp;
    double press_ref= d_sharedState->getRefPress();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numALLMatls = numICEMatls + numMPMMatls;

    Vector dx       = patch->dCell(); 
    double cell_vol = dx.x()*dx.y()*dx.z();

    StaticArray<double> delVol_frac(numALLMatls),press_eos(numALLMatls);
    StaticArray<double> dp_drho(numALLMatls),dp_de(numALLMatls);
    StaticArray<double> mat_volume(numALLMatls);
    StaticArray<double> mat_mass(numALLMatls);
    StaticArray<double> cv(numALLMatls);
    StaticArray<double> gamma(numALLMatls);
    StaticArray<double> compressibility(numALLMatls);
    StaticArray<CCVariable<double> > vol_frac(numALLMatls);
    StaticArray<CCVariable<double> > rho_micro(numALLMatls);
    StaticArray<CCVariable<double> > sp_vol_new(numALLMatls);
    StaticArray<CCVariable<double> > rho_CC(numALLMatls);
    StaticArray<CCVariable<double> > speedSound_new(numALLMatls);
    StaticArray<CCVariable<double> > f_theta(numALLMatls);
    StaticArray<CCVariable<double> > matl_press(numALLMatls);

    StaticArray<constCCVariable<double> > Temp(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numALLMatls);
    StaticArray<constCCVariable<double> > mat_vol(numALLMatls);
    StaticArray<constCCVariable<double> > rho_top(numALLMatls);
    StaticArray<constCCVariable<double> > mass_CC(numALLMatls);
    CCVariable<double> press_new; 

    new_dw->allocate(press_new,Ilb->press_equil_CCLabel, 0,patch);
    
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(ice_matl){                    // I C E
        old_dw->get(Temp[m],   Ilb->temp_CCLabel,  indx,patch,Ghost::None,0);
        old_dw->get(rho_top[m],Ilb->rho_CC_top_cycleLabel,
                                                   indx,patch,Ghost::None,0);
        old_dw->get(sp_vol_CC[m],
                               Ilb->sp_vol_CCLabel,indx,patch,Ghost::None,0);
        cv[m]    = ice_matl->getSpecificHeat();
        gamma[m] = ice_matl->getGamma();
      }
      if(mpm_matl){                    // M P M    
        new_dw->get(Temp[m],   MIlb->temp_CCLabel,indx, patch,Ghost::None,0);
        new_dw->get(mat_vol[m],MIlb->cVolumeLabel,indx, patch,Ghost::None,0);
        new_dw->get(mass_CC[m],MIlb->cMassLabel,  indx, patch,Ghost::None,0);
        cv[m] = mpm_matl->getSpecificHeat();
      }
      new_dw->allocate(sp_vol_new[m],Ilb->sp_vol_CCLabel,    indx, patch);
      new_dw->allocate(rho_CC[m],    Ilb->rho_CCLabel,       indx, patch);
      new_dw->allocate(vol_frac[m],  Ilb->vol_frac_CCLabel,  indx, patch);
      new_dw->allocate(f_theta[m],   Ilb->f_theta_CCLabel,   indx, patch);
      new_dw->allocate(matl_press[m],Ilb->matl_press_CCLabel,indx, patch);
      new_dw->allocate(rho_micro[m], Ilb->rho_micro_CCLabel, indx, patch);
      new_dw->allocate(speedSound_new[m],Ilb->speedSound_CCLabel,indx,patch);
      speedSound_new[m].initialize(0.0);
    }
    
    press_new.initialize(0.0);

    //__________________________________
    // Compute rho_micro,matl_press, total_vol, speedSound
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      double total_mat_vol = 0.0;
      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

        if(ice_matl){                // I C E
         rho_micro[m][c] = 1.0/sp_vol_CC[m][c];
         ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m],
                                         cv[m],Temp[m][c],
                                         press_eos[m],dp_drho[m],dp_de[m]);
         
         compressibility[m]=
                     ice_matl->getEOS()->getCompressibility(press_eos[m]);

         mat_mass[m]   = rho_top[m][c] * cell_vol;
         mat_volume[m] = mat_mass[m] * sp_vol_CC[m][c];

         tmp = dp_drho[m] + dp_de[m] * 
           (press_eos[m]*(sp_vol_CC[m][c] * sp_vol_CC[m][c]));  
        } 

        if(mpm_matl){                //  M P M
          rho_micro[m][c] = mass_CC[m][c]/mat_vol[m][c];
          mat_mass[m]     = mass_CC[m][c];

          compressibility[m]=
              mpm_matl->getConstitutiveModel()->getCompressibility();

          mpm_matl->getConstitutiveModel()->
            computePressEOSCM(rho_micro[m][c],press_eos[m], press_ref,
                              dp_drho[m], tmp,mpm_matl);

          mat_volume[m] = mat_vol[m][c];
        }              
        matl_press[m][c] = press_eos[m];
        total_mat_vol += mat_volume[m];
        speedSound_new[m][c] = sqrt(tmp);
       }  // for ALLMatls...
       
       
      //__________________________________
      // Compute 1/f_theta
       double f_theta_denom = 0.0;
       for (int m = 0; m < numALLMatls; m++) {
         vol_frac[m][c] = mat_volume[m]/total_mat_vol;
         f_theta_denom += vol_frac[m][c]*compressibility[m];
       }
       //__________________________________
       // Compute press_new
       for (int m = 0; m < numALLMatls; m++) {
         f_theta[m][c] = vol_frac[m][c]*compressibility[m]/f_theta_denom;
         press_new[c] += f_theta[m][c]*matl_press[m][c];
       }
    } // for(CellIterator...)
    
    //__________________________________
    // Now change how rho_CC is defined to 
    // rho_CC = mass/cell_volume  NOT mass/mat_volume 
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(ice_matl){
        for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          rho_CC[m][c] = rho_top[m][c];
        }
      }
      if(mpm_matl){
        for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          rho_CC[m][c] = mass_CC[m][c]/cell_vol;
        }
      }
    } 
    
    //__________________________________
    //  Update boundary conditions
    for (int m = 0; m < numALLMatls; m++)   {
       Material* matl = d_sharedState->getMaterial( m );
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       int dwi = matl->getDWIndex();
       if(ice_matl){
         d_ice->setBC(rho_CC[m],   "Density" ,patch, dwi);
       }  
       d_ice->setBC(matl_press[m],rho_micro[SURROUND_MAT],
                    "rho_micro", "Pressure", patch, 0, new_dw);
    }  
    d_ice->setBC(press_new, rho_micro[SURROUND_MAT], 
                "rho_micro", "Pressure", patch, 0,  new_dw);

    //__________________________________
    // compute sp_vol_CC
    for (int m = 0; m < numALLMatls; m++)   {
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
        IntVector c = *iter;
        sp_vol_new[m][c] = 1.0/rho_micro[m][c];

      }
    } 
    //__________________________________
    //    carry sp_vol_CC forward for MPMICE:computeEquilibrationPressure
    for (int m = 0; m < numALLMatls; m++)   {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->put( sp_vol_new[m], Ilb->sp_vol_CCLabel,     indx, patch); 
      new_dw->put( vol_frac[m],      Ilb->vol_frac_CCLabel,   indx, patch);
      new_dw->put( f_theta[m],       Ilb->f_theta_CCLabel,    indx, patch);
      new_dw->put( matl_press[m],    Ilb->matl_press_CCLabel, indx, patch);
      new_dw->put( speedSound_new[m],Ilb->speedSound_CCLabel, indx, patch);
      new_dw->put( rho_CC[m],        Ilb->rho_CCLabel,        indx, patch);
    }
    new_dw->put(press_new,Ilb->press_equil_CCLabel,0,patch);

  } // patches
}
