/* ---------------------------------------------------------------------
 Function~  MPMICE::scheduleComputeNonEquilibrationPressureRF--
 Note:  This similar to ICE::scheduleComputeEquilibrationPressure
         with the addition of MPM matls
_____________________________________________________________________*/
void MPMICE::scheduleComputeNonEquilibrationPressureRF(SchedulerP& sched,
					    const PatchSet* patches,
                                       const MaterialSubset* ice_matls,
                                       const MaterialSubset* mpm_matls,
                                       const MaterialSubset* press_matl,
					    const MaterialSet*    all_matls)
{
  cout_doing << "MPMICE::scheduleComputeNonEquilibrationPressureRF" << endl;

  Task* t = scinew Task("MPMICE::computeNonEquilibrationPressureRF",
                     this, &MPMICE::computeNonEquilibrationPressureRF);

//  t->requires(Task::OldDW,Ilb->press_CCLabel, press_matl, Ghost::None);
                 // I C E
  t->requires(Task::OldDW,Ilb->temp_CCLabel,            ice_matls, Ghost::None);
  t->requires(Task::OldDW,Ilb->rho_CC_top_cycleLabel,   ice_matls, Ghost::None);
  t->requires(Task::OldDW,Ilb->sp_vol_CCLabel,          ice_matls, Ghost::None);
                // M P M
  t->requires(Task::NewDW,MIlb->temp_CCLabel, mpm_matls,  Ghost::None);
  t->requires(Task::NewDW,MIlb->cVolumeLabel, mpm_matls,  Ghost::None);
  t->requires(Task::NewDW,MIlb->cMassLabel,   mpm_matls,  Ghost::None);  

                //  A L L _ M A T L S
  t->computes(Ilb->speedSound_CCLabel);
  t->computes(Ilb->rho_micro_CCLabel);
  t->computes(Ilb->vol_frac_CCLabel);
  t->computes(Ilb->rho_CCLabel);
  t->computes(Ilb->matl_press_CCLabel);
  t->computes(Ilb->f_theta_CCLabel);

  t->computes(Ilb->press_equil_CCLabel, press_matl);

  sched->addTask(t, patches, all_matls);
}
//______________________________________________________________________
//
void MPMICE::scheduleCCMomExchangeRF(SchedulerP& sched,
				   const PatchSet* patches,
                               const MaterialSubset* ice_matls,
                               const MaterialSubset* mpm_matls,
				   const MaterialSet* all_matls)
{

  cout_doing << "MPMICE::scheduleCCMomExchangeRF" << endl;
 
  Task* t=scinew Task("MPMICE::doCCMomExchangeRF",
		  this, &MPMICE::doCCMomExchangeRF);
  t->requires(Task::OldDW, d_sharedState->get_delt_label());
                                 // I C E
  t->computes(Ilb->mom_L_ME_CCLabel,     ice_matls);
  t->computes(Ilb->int_eng_L_ME_CCLabel, ice_matls);
  t->requires(Task::NewDW, Ilb->mass_L_CCLabel, ice_matls, Ghost::None);
  t->requires(Task::OldDW, Ilb->temp_CCLabel,   ice_matls, Ghost::None);

                                 // M P M
  t->computes(MIlb->dTdt_CCLabel, mpm_matls);
  t->computes(MIlb->dvdt_CCLabel, mpm_matls);
  t->requires(Task::NewDW,  Ilb->rho_CCLabel,   mpm_matls, Ghost::None);
  t->requires(Task::NewDW,  Ilb->temp_CCLabel,  mpm_matls, Ghost::None);

                                // A L L  M A T L S
  t->requires(Task::NewDW,  Ilb->mom_L_CCLabel,     Ghost::None);
  t->requires(Task::NewDW,  Ilb->int_eng_L_CCLabel, Ghost::None);
  t->requires(Task::NewDW,  Ilb->rho_micro_CCLabel, Ghost::None);
  t->requires(Task::NewDW,  Ilb->vol_frac_CCLabel,  Ghost::None);
  t->computes(Ilb->Tdot_CCLabel);


  sched->addTask(t, patches, all_matls);
}


/* --------------------------------------------------------------------- 
 Function~  MPMICE::computeNonEquilibrationPressureRF-- 
 Reference: A Multifield Model and Method for Fluid Structure
            Interaction Dynamics
_____________________________________________________________________*/
void MPMICE::computeNonEquilibrationPressureRF(const ProcessorGroup*,
			 		     const PatchSubset* patches,
					     const MaterialSubset* ,
					     DataWarehouse* old_dw,
					     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing<<"Doing computeNonEquilibrationPressure(RF) on patch "
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
      int dwindex = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(ice_matl){                    // I C E
        old_dw->get(Temp[m],   Ilb->temp_CCLabel, dwindex, patch,Ghost::None,0);
        old_dw->get(rho_top[m],Ilb->rho_CC_top_cycleLabel,
						  dwindex, patch,Ghost::None,0);
        old_dw->get(sp_vol_CC[m],
			       Ilb->sp_vol_CCLabel,dwindex,patch,Ghost::None,0);
        cv[m]    = ice_matl->getSpecificHeat();
        gamma[m] = ice_matl->getGamma();
      }
      if(mpm_matl){                    // M P M    
        new_dw->get(Temp[m],   MIlb->temp_CCLabel,dwindex, patch,Ghost::None,0);
        new_dw->get(mat_vol[m],MIlb->cVolumeLabel,dwindex, patch,Ghost::None,0);
        new_dw->get(mass_CC[m],MIlb->cMassLabel,  dwindex, patch,Ghost::None,0);
        cv[m] = mpm_matl->getSpecificHeat();
      }
      new_dw->allocate(rho_CC[m],    Ilb->rho_CCLabel,       dwindex, patch);
      new_dw->allocate(vol_frac[m],  Ilb->vol_frac_CCLabel,  dwindex, patch);
      new_dw->allocate(f_theta[m],   Ilb->f_theta_CCLabel,   dwindex, patch);
      new_dw->allocate(matl_press[m],Ilb->matl_press_CCLabel,dwindex, patch);
      new_dw->allocate(rho_micro[m], Ilb->rho_micro_CCLabel, dwindex, patch);
      new_dw->allocate(speedSound_new[m],Ilb->speedSound_CCLabel,dwindex,patch);
      speedSound_new[m].initialize(0.0);
    }
    
    press_new.initialize(0.0);

    // Compute rho_micro, speedSound, volfrac, rho_CC
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      double total_mat_vol = 0.0;
      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

        if(ice_matl){                // I C E
	  rho_micro[m][*iter] = 1.0/sp_vol_CC[m][*iter];
	  ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma[m],
					      cv[m],Temp[m][*iter],
					      press_eos[m],dp_drho[m],dp_de[m]);
	  
	  compressibility[m]=
			ice_matl->getEOS()->getCompressibility(press_eos[m]);

          mat_mass[m]   = rho_top[m][*iter] * cell_vol;
	  mat_volume[m] = mat_mass[m] * sp_vol_CC[m][*iter];

	  tmp = dp_drho[m] + dp_de[m] * 
	    (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
        } 

        if(mpm_matl){                //  M P M
	   rho_micro[m][*iter] = mass_CC[m][*iter]/mat_vol[m][*iter];
           mat_mass[m]   = mass_CC[m][*iter];

	   compressibility[m]=
		mpm_matl->getConstitutiveModel()->getCompressibility();

	   mpm_matl->getConstitutiveModel()->
	     computePressEOSCM(rho_micro[m][*iter],press_eos[m], press_ref,
                              dp_drho[m], tmp,mpm_matl);

	   mat_volume[m] = mat_vol[m][*iter];
        }              

	matl_press[m][*iter] = press_eos[m];
        total_mat_vol += mat_volume[m];
        speedSound_new[m][*iter] = sqrt(tmp);

       }  // for ALLMatls...

       double f_theta_denom = 0.0;
       for (int m = 0; m < numALLMatls; m++) {
         vol_frac[m][*iter] = mat_volume[m]/total_mat_vol;
         rho_CC[m][*iter] = vol_frac[m][*iter]*rho_micro[m][*iter];
	 f_theta_denom += vol_frac[m][*iter]*compressibility[m];
       }
       for (int m = 0; m < numALLMatls; m++) {
	f_theta[m][*iter] = vol_frac[m][*iter]*compressibility[m]/f_theta_denom;
	press_new[*iter] += f_theta[m][*iter]*matl_press[m][*iter];
	rho_CC[m][*iter]  = mat_mass[m]/cell_vol;
       }
    } // for(CellIterator...)


    //  Set BCs on density and pressure
    for (int m = 0; m < numALLMatls; m++)   {
       Material* matl = d_sharedState->getMaterial( m );
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       int dwi = matl->getDWIndex();
       if(ice_matl){
         d_ice->setBC(rho_CC[m],   "Density" ,patch, dwi);
       }  
       d_ice->setBC(matl_press[m],rho_micro[SURROUND_MAT],"Pressure",patch,dwi);
    }  

    //__________________________________
    //  Update BC
    d_ice->setBC(press_new, rho_micro[SURROUND_MAT], "Pressure",patch,0);

    //__________________________________
    //    Put all matls into new dw
    for (int m = 0; m < numALLMatls; m++)   {
      Material* matl = d_sharedState->getMaterial( m );
      int dwindex = matl->getDWIndex();
      new_dw->put( vol_frac[m],      Ilb->vol_frac_CCLabel,   dwindex, patch);
      new_dw->put( f_theta[m],       Ilb->f_theta_CCLabel,    dwindex, patch);
      new_dw->put( matl_press[m],    Ilb->matl_press_CCLabel, dwindex, patch);
      new_dw->put( speedSound_new[m],Ilb->speedSound_CCLabel, dwindex, patch);
      new_dw->put( rho_micro[m],     Ilb->rho_micro_CCLabel,  dwindex, patch);
      new_dw->put( rho_CC[m],        Ilb->rho_CCLabel,        dwindex, patch);
    }
    new_dw->put(press_new,Ilb->press_equil_CCLabel,0,patch);

  } // patches
}

/* ---------------------------------------------------------------------
 Function~  MPMICE::doCCMomExchangeRF--
_____________________________________________________________________*/
void MPMICE::doCCMomExchangeRF(const ProcessorGroup*,
                             const PatchSubset* patches,
			     const MaterialSubset* ,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing doCCMomExchange (RF) on patch "<< patch->getID()
               <<"\t\t\t MPMICE" << endl;

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numALLMatls = numMPMMatls + numICEMatls;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx = patch->dCell();
    Vector zero(0.,0.,0.);

    // Create arrays for the grid data
    StaticArray<CCVariable<double> > Temp_CC(numALLMatls);  
    StaticArray<constCCVariable<double> > vol_frac_CC(numALLMatls);
    StaticArray<constCCVariable<double> > rho_micro_CC(numALLMatls);

    StaticArray<constCCVariable<Vector> > mom_L(numALLMatls);
    StaticArray<constCCVariable<double> > int_eng_L(numALLMatls);

    // Create variables for the results
    StaticArray<CCVariable<Vector> > mom_L_ME(numALLMatls);
    StaticArray<CCVariable<Vector> > vel_CC(numALLMatls);
    StaticArray<CCVariable<Vector> > dvdt_CC(numALLMatls);
    StaticArray<CCVariable<double> > dTdt_CC(numALLMatls);
    StaticArray<NCVariable<double> > dTdt_NC(numALLMatls);
    StaticArray<CCVariable<double> > int_eng_L_ME(numALLMatls);
    StaticArray<CCVariable<double> > mass_L_temp(numALLMatls);
    StaticArray<constCCVariable<double> > mass_L(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC(numALLMatls);
    StaticArray<constCCVariable<double> > old_temp(numALLMatls);
    StaticArray<CCVariable<double> > Tdot(numALLMatls);

    vector<double> b(numALLMatls);
    vector<double> density(numALLMatls);
    vector<double> cv(numALLMatls);
    DenseMatrix beta(numALLMatls,numALLMatls),acopy(numALLMatls,numALLMatls);
    DenseMatrix K(numALLMatls,numALLMatls),H(numALLMatls,numALLMatls);
    DenseMatrix a(numALLMatls,numALLMatls), a_inverse(numALLMatls,numALLMatls);
    beta.zero();
    acopy.zero();
    K.zero();
    H.zero();
    a.zero();

    d_ice->getExchangeCoefficients( K, H);

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int dwindex = matl->getDWIndex();
      if(mpm_matl){                   // M P M
        new_dw->allocate(vel_CC[m],  MIlb->velstar_CCLabel,     dwindex, patch);
        new_dw->allocate(Temp_CC[m], MIlb->temp_CC_scratchLabel,dwindex, patch);
        new_dw->allocate(mass_L_temp[m],  Ilb->mass_L_CCLabel,  dwindex, patch);

        new_dw->get(rho_CC[m],        Ilb->rho_CCLabel,         dwindex, patch,
							        Ghost::None, 0);
        new_dw->get(old_temp[m],      Ilb->temp_CCLabel,        dwindex, patch,
							        Ghost::None, 0);
        cv[m] = mpm_matl->getSpecificHeat();
      }
      if(ice_matl){                 // I C E
        new_dw->allocate(vel_CC[m], Ilb->vel_CCLabel,     dwindex, patch);
        new_dw->allocate(Temp_CC[m],Ilb->temp_CCLabel,    dwindex, patch);
        new_dw->get(mass_L[m],      Ilb->mass_L_CCLabel,  dwindex, patch,
							  Ghost::None, 0);
        old_dw->get(old_temp[m],    Ilb->temp_CCLabel,    dwindex, patch,
							  Ghost::None, 0);
        cv[m] = ice_matl->getSpecificHeat();
      }                             // A L L  M A T L S

      new_dw->get(rho_micro_CC[m],  Ilb->rho_micro_CCLabel, dwindex, patch,
							  Ghost::None, 0);
      new_dw->get(mom_L[m],         Ilb->mom_L_CCLabel,     dwindex, patch,
							  Ghost::None, 0);
      new_dw->get(int_eng_L[m],     Ilb->int_eng_L_CCLabel, dwindex, patch,
							  Ghost::None, 0);
      new_dw->get(vol_frac_CC[m],   Ilb->vol_frac_CCLabel,  dwindex, patch,
							  Ghost::None, 0);
      new_dw->allocate(dvdt_CC[m], MIlb->dvdt_CCLabel,      dwindex, patch);
      new_dw->allocate(dTdt_CC[m], MIlb->dTdt_CCLabel,      dwindex, patch);
      new_dw->allocate(mom_L_ME[m], Ilb->mom_L_ME_CCLabel,  dwindex,patch);
      new_dw->allocate(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex,patch);
      new_dw->allocate(Tdot[m],     Ilb->Tdot_CCLabel,      dwindex,patch);

      dvdt_CC[m].initialize(zero);
      dTdt_CC[m].initialize(0.);
    }

    double vol = dx.x()*dx.y()*dx.z();
    double tmp;

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(mpm_matl){
       // Loaded rho_CC into mass_L for solid matl's, converting to mass_L
       for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
	  mass_L_temp[m][*iter] = rho_CC[m][*iter]*vol;
       }
       mass_L[m] = mass_L_temp[m];
      }
    }

    // Convert momenta to velocities.  Slightly different for MPM and ICE.
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      for (int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][*iter] = int_eng_L[m][*iter]/(mass_L[m][*iter]*cv[m]);
        vel_CC[m][*iter]  = mom_L[m][*iter]/mass_L[m][*iter];
      }
    }

    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      //   Form BETA matrix (a), off diagonal terms
      //  The beta and (a) matrix is common to all momentum exchanges
      for(int m = 0; m < numALLMatls; m++)  {
        density[m]  = rho_micro_CC[m][*iter];
        for(int n = 0; n < numALLMatls; n++) {
	  beta[m][n] = delT * vol_frac_CC[n][*iter] * K[n][m]/density[m];
	  a[m][n] = -beta[m][n];
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numALLMatls; m++) {
        a[m][m] = 1.;
        for(int n = 0; n < numALLMatls; n++) {
	  a[m][m] +=  beta[m][n];
        }
      }
      d_ice->matrixInverse(numALLMatls, a, a_inverse);
      
      //     X - M O M E N T U M  --  F O R M   R H S   (b)
      for(int m = 0; m < numALLMatls; m++) {
        b[m] = 0.0;
        for(int n = 0; n < numALLMatls; n++) {
	  b[m] += beta[m][n] * (vel_CC[n][*iter].x() - vel_CC[m][*iter].x());
        }
      }
      //     S O L V E
      //  - Add exchange contribution to orig value     
      vector<double> X(numALLMatls);
      d_ice->multiplyMatrixAndVector(numALLMatls,a_inverse,b,X);
      for(int m = 0; m < numALLMatls; m++) {
	  vel_CC[m][*iter].x( vel_CC[m][*iter].x() + X[m] );
	  dvdt_CC[m][*iter].x( X[m]/delT );
      } 

      //     Y - M O M E N T U M  --   F O R M   R H S   (b)
      for(int m = 0; m < numALLMatls; m++) {
        b[m] = 0.0;
        for(int n = 0; n < numALLMatls; n++) {
	  b[m] += beta[m][n] * (vel_CC[n][*iter].y() - vel_CC[m][*iter].y());
        }
      }

      //     S O L V E
      //  - Add exchange contribution to orig value
      d_ice->multiplyMatrixAndVector(numALLMatls,a_inverse,b,X);
      for(int m = 0; m < numALLMatls; m++)  {
	  vel_CC[m][*iter].y( vel_CC[m][*iter].y() + X[m] );
	  dvdt_CC[m][*iter].y( X[m]/delT );
      }

      //     Z - M O M E N T U M  --  F O R M   R H S   (b)
      for(int m = 0; m < numALLMatls; m++)  {
        b[m] = 0.0;
        for(int n = 0; n < numALLMatls; n++) {
	  b[m] += beta[m][n] * (vel_CC[n][*iter].z() - vel_CC[m][*iter].z());
        }
      }    

      //     S O L V E
      //  - Add exchange contribution to orig value
      d_ice->multiplyMatrixAndVector(numALLMatls,a_inverse,b,X);
      for(int m = 0; m < numALLMatls; m++)  {
	  vel_CC[m][*iter].z( vel_CC[m][*iter].z() + X[m] );
	  dvdt_CC[m][*iter].z( X[m]/delT );
      }

      //---------- E N E R G Y   E X C H A N G E
      //         
      for(int m = 0; m < numALLMatls; m++) {
        tmp = cv[m]*rho_micro_CC[m][*iter];
        for(int n = 0; n < numALLMatls; n++)  {
	  beta[m][n] = delT * vol_frac_CC[n][*iter] * H[n][m]/tmp;
	  a[m][n] = -beta[m][n];
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numALLMatls; m++) {
        a[m][m] = 1.;
        for(int n = 0; n < numALLMatls; n++)   {
	  a[m][m] +=  beta[m][n];
        }
      }
      // -  F O R M   R H S   (b)
      for(int m = 0; m < numALLMatls; m++)  {
        b[m] = 0.0;

       for(int n = 0; n < numALLMatls; n++) {
	  b[m] += beta[m][n] *
	    (Temp_CC[n][*iter] - Temp_CC[m][*iter]);
        }
      }
      //     S O L V E, Add exchange contribution to orig value
      d_ice->matrixSolver(numALLMatls,a,b,X);
      for(int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][*iter] = Temp_CC[m][*iter] + X[m];
        dTdt_CC[m][*iter] = X[m]/delT;
      }

    }  //end CellIterator loop

    //__________________________________
    //  Set the Boundary conditions 
    //   Do this for all matls even though MPM doesn't
    //   care about this.  For two identical ideal gases
    //   mom_L_ME and int_eng_L_ME should be identical and this
    //   is useful when debugging.
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int dwindex = matl->getDWIndex();
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      d_ice->setBC(vel_CC[m], "Velocity",   patch,dwindex);
      d_ice->setBC(Temp_CC[m],"Temperature",patch,dwindex);
      

      //__________________________________
      //  Symetry BC dTdt: Neumann = 0
      //             dvdt: tangent components Neumann = 0
      //                   normal component negInterior
      if(mpm_matl){
        d_ice->setBC(dTdt_CC[m], "set_if_sym_BC", patch, dwindex);
        d_ice->setBC(dvdt_CC[m], "set_if_sym_BC", patch, dwindex);
      }
    }
    //__________________________________
    // Convert vars. primitive-> flux 
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      for (int m = 0; m < numALLMatls; m++) {
          int_eng_L_ME[m][*iter] = Temp_CC[m][*iter] * cv[m] * mass_L[m][*iter];
          mom_L_ME[m][*iter]     = vel_CC[m][*iter]          * mass_L[m][*iter];
	  Tdot[m][*iter] = (Temp_CC[m][*iter] - old_temp[m][*iter])/delT;
      }
    }
    //---- P R I N T   D A T A ------ 
    if (d_ice->switchDebugMomentumExchange_CC ) {
      for(int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int dwindex = matl->getDWIndex();
        char description[50];;
        sprintf(description, "MPMICE_momExchange_CC_%d_patch_%d ", 
                dwindex, patch->getID());
        d_ice->printVector(patch,1, description, "xmom_L_ME", 0, mom_L_ME[m]);
        d_ice->printVector(patch,1, description, "ymom_L_ME", 1, mom_L_ME[m]);
        d_ice->printVector(patch,1, description, "zmom_L_ME", 2, mom_L_ME[m]);
        d_ice->printData(  patch,1, description,"int_eng_L_ME",int_eng_L_ME[m]);
        d_ice->printData(  patch,1, description, "dTdt_CC",       dTdt_CC[m]);
        d_ice->printVector(patch,1, description, "dVdt_CC.X",  0, dvdt_CC[m]);
        d_ice->printVector(patch,1, description, "dVdt_CC.Y",  1, dvdt_CC[m]);
        d_ice->printVector(patch,1, description, "dVdt_CC.Z",  2, dvdt_CC[m]);
      }
    }
    //__________________________________
    //    Put into new_dw
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int dwindex = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
       if(ice_matl){
         new_dw->put(mom_L_ME[m],    Ilb->mom_L_ME_CCLabel,    dwindex, patch);
         new_dw->put(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex, patch);
       }
       if(mpm_matl){
        new_dw->put(dvdt_CC[m],MIlb->dvdt_CCLabel,dwindex,patch);
        new_dw->put(dTdt_CC[m],MIlb->dTdt_CCLabel,dwindex,patch);
       }
       new_dw->put(Tdot[m],Ilb->Tdot_CCLabel,dwindex, patch);
    }  
  } //patches
}
