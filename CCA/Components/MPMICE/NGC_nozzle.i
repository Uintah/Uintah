//______________________________________________________________________
//  This file contains the hard wiring necessary for the
//  Northrup Grumman Corp. two stage nozzle separation simulation
//  The lower stage velocity the nozzle pressure, temperature
//  and density all vary with time. 
//
//  To turn on the hard wiring simply uncomment the 
//  
//#define running_NGC_nozzle
#undef  running_NGC_nozzle 


#ifdef running_NGC_nozzle
  //______________________________________________________________________
  //          M P M
  //__________________________________
  #ifdef rigidBody_1
     // additional variables need for the hardwired velocity
     t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);           
     t->requires(Task::OldDW, lb->pXLabel,      Ghost::None);        
  #endif
  
  
  
  //  Inside RidgidBodyContact.cc: exMomIntegrated()
  #ifdef rigidBody_2 
      //__________________________________
      //  Hardwiring for Northrup Grumman nozzle
      //  Loop over all the particles and set the velocity
      //  of the 2nd stage = to the hard coded velocity
      //  Note that the pcolor of the 2nd stage has been 
      //  initialized to a non zero value

      int dwi = 0;
      NCVariable<Vector> gvelocity_star_org;
      new_dw->getCopy(gvelocity_star_org, lb->gVelocityStarLabel,dwi,patch);
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constParticleVariable<double> pcolor;
      constParticleVariable<Point> px;
      old_dw->get(pcolor, lb->pColorLabel, pset);
      old_dw->get(px,     lb->pXLabel,     pset);

      double t = d_sharedState->getElapsedTime();
      double time = t + d_sharedState->getTimeOffset();
      vector<double> c(4);
      c[3] = 5.191382077671874e+05;
      c[2] = -6.077090505408908e+03;
      c[1] =  1.048411519176967e+01;
      c[0] =  1.606269799418405e-01;

      double vel_x = c[3]*pow(time,3) + c[2]*pow(time,2) + c[1]*time + c[0];
      Vector hardWired_velocity = Vector(vel_x,0,0);

#if 0
      if (patch->getID() ){
        cout << " hardWired_velocity " << hardWired_velocity << endl;
      }
#endif

      for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); 
                                                                          iter++){
        particleIndex idx = *iter;

        if (pcolor[idx]> 0) {          // for the second stage only

          // Get the node indices that surround the cell
          IntVector ni[8];
          double S[8];
          patch->findCellAndWeights(px[idx], ni, S);

          for(int k = 0; k < 8; k++) {
            if(patch->containsNode(ni[k])) {
              gacceleration[0][ni[k]] = (hardWired_velocity - gvelocity_star_org[ni[k]])/delT;
	      gvelocity_star[0][ni[k]]= hardWired_velocity; 
            }
          }
        }  // second stage only
      } // End of particle loop
  #endif


  //__________________________________
  //  Inside SerialMPM.cc: actuallyInitialize()
  //  Initialize pcolor of the 2nd stages particles to
  //  1.0.  Note you MUST have the initial value of the
  //  2nd stage velocity != 0
  
  #ifdef SerialMPM_1
    constParticleVariable<Vector> pvelocity;
    new_dw->get(pvelocity,  lb->pVelocityLabel, pset);

    ParticleSubset::iterator iter = pset->begin();

    for(;iter != pset->end(); iter++){
      particleIndex idx = *iter;
      if (pvelocity[idx].length() > 0.0 ) {  
        pcolor[idx] = 1.0;
      }
    }
  #endif
  //______________________________________________________________________
  //        M P M I C E
  //__________________________________
  #if 0
    //__________________________________
    // put this in equilibration pressure just before the upper printData
    // find the cells along the inner wall of the nozzle  
    // - only on the first timestep
    // - examine the vertical gradient of vol_frac for the nozzle matl (0)
    // - 
    if (dataArchiver->getCurrentTimestep() == 1) {
      Vector dx = patch->dCell();
      CellIterator iter=patch->getCellIterator();
      IntVector lo = iter.begin();
      IntVector hi = iter.end();
      const Level* level = getLevel(patches);
      Point this_cell_pos = level->getCellPosition(hi);
      
      if (this_cell_pos.x() < 0.38){
      
        int k   = 0;
        int mat = 0;
        
        for(int i = lo.x(); i < hi.x(); i++) {
          for(int j = lo.y(); j < hi.y()-1; j++) {
            IntVector c(i, j, k);
            IntVector adj(i, j+1, k);
            Point cell_pos = level->getCellPosition(c);

            // compute vertical gradient of vol_frac
            double gradY = (vol_frac[mat][adj] - vol_frac[mat][c])/dx.y();
            
            // if cell position is below a certain position

            if(vol_frac[mat][adj] > 0.001 && cell_pos.y() < 0.13){
              if (fabs(gradY)>0.1){
                cout << c << " adj " << vol_frac[mat][adj]
                          << " c " << vol_frac[mat][c]
                          << " gradY " << gradY << endl;
                j = hi.y();
              }
            }
          }
        }
      }
    }
  #endif

  //______________________________________________________________________
  //        I C E
  //__________________________________
  //  Inside ICE/BoundaryCond.cc: setBC(Pressure)
  //  hard code the pressure as a function of time
#ifdef ICEBoundaryCond_1
    BCGeomBase* bc_geom_type = patch->getBCDataArray(face)->getChild(mat_id,child);
    cmp_type<CircleBCData> nozzle;

    if(kind == "Pressure" && face == Patch::xminus && nozzle(bc_geom_type) ) {
      vector<IntVector>::const_iterator iter;
     
      //__________________________________
      //  curve fit for the pressure
      double mean = 3.2e-02;
      double std  = 1.851850425925377e-02;
      vector<double> c(8);
      c[7] = -2.233767893099373e+05;
      c[6] = -5.824923861605825e+04;
      c[5] =  1.529343908400654e+06;
      c[4] =  1.973796762592652e+05;
      c[3] =  -3.767381404747613e+06;
      c[2] =  5.939587204102841e+04;
      c[1] =  5.692124635957888e+06;    
      c[0] =  3.388928006241853e+06;
      
      
      double t = sharedState->getElapsedTime();
      double time = t + sharedState->getTimeOffset();
      
      double tbar = (time - mean)/std;
      
      double stag_press = c[7] * pow(tbar,7) + c[6]*pow(tbar,6) + c[5]*pow(tbar,5)
                       + c[4] * pow(tbar,4) + c[3]*pow(tbar,3) + c[2]*pow(tbar,2)
                       + c[1] * tbar + c[0];
      //__________________________________
      //  Isenentrop relations for A/A* = 1.88476
      double p_p0     = 0.92850;
      double static_press = p_p0 * stag_press;
#if 0
      if(patch->getID() == 0){
        cout << " time " << time 
              << " static_press " << static_press << endl;
      }
#endif
      for (iter=bound.begin(); iter != bound.end(); iter++) {
        press_CC[*iter] = static_press;
      }
      //cout << "Time " << time << " Pressure:   child " << child 
      //     <<"\t bound limits = "<< *bound.begin()<< " "<< *(bound.end()-1) << endl;
    }
#endif  // ICEBoundaryCond_1


  //__________________________________
  //  Inside ICE/BoundaryCond.cc: setBC(Temperature/Densiity)
  //  hard code the temperature and density as a function of time
#ifdef ICEBoundaryCond_2
    BCGeomBase* bc_geom_type = patch->getBCDataArray(face)->getChild(mat_id,child);
    cmp_type<CircleBCData> nozzle;

    if(face == Patch:: xminus && mat_id == 1 && nozzle(bc_geom_type)){
      vector<IntVector>::const_iterator iter;

      //__________________________________
      //  constants and curve fit for pressure
      double pascals_per_psi = 6894.4;
      
      double gamma = 1.4;
      double cv    = 716;
      
      double mean = 3.2e-02;
      double std  = 1.851850425925377e-02;
      vector<double> c(8);
      c[7] = -2.233767893099373e+05;
      c[6] = -5.824923861605825e+04;
      c[5] =  1.529343908400654e+06;
      c[4] =  1.973796762592652e+05;
      c[3] =  -3.767381404747613e+06;
      c[2] =  5.939587204102841e+04;
      c[1] =  5.692124635957888e+06;    
      c[0] =  3.388928006241853e+06;
      
      double t = sharedState->getElapsedTime();
      double time = t + sharedState->getTimeOffset();

      double tbar = (time - mean)/std;
      
      double stag_press = c[7] * pow(tbar,7) + c[6]*pow(tbar,6) + c[5]*pow(tbar,5)
                       + c[4] * pow(tbar,4) + c[3]*pow(tbar,3) + c[2]*pow(tbar,2)
                       + c[1] * tbar + c[0];
      
      double stag_temp  = (2.7988e3 +110.5*log(stag_press/pascals_per_psi) ); 
      double stag_rho   = stag_press/((gamma - 1.0) * cv * stag_temp);

      //__________________________________
      //  Isenentrop relations for A/A* = 1.88476
      double T_T0     = 0.97993;
      double rho_rho0 = 0.94839;

      double static_temp = T_T0 * stag_temp;
      double static_rho  = rho_rho0 * stag_rho;

#if 0
      if (patch->getID() ) {
        cout << " time " << time 
             << " static_temp " << static_temp
             << " static_rho " << static_rho << endl;
      } 
#endif     
      if(desc == "Temperature") {
        for (iter=bound.begin(); iter != bound.end(); iter++) {
          var_CC[*iter] = static_temp;
        }
      }
      if(desc == "Density") {
        for (iter=bound.begin(); iter != bound.end(); iter++) {
          var_CC[*iter] = static_rho;
        }
      }
    } 
  #endif  //  ICEBoundaryCond_2

  //__________________________________
  //  Inside ICE/BoundaryCond.cc: setBC(Velocity)
#ifdef ICEBoundaryCond_3
    BCGeomBase* bc_geom_type = patch->getBCDataArray(face)->getChild(mat_id,child);
    cmp_type<CircleBCData> nozzle;

    if(face == Patch:: xminus && mat_id == 1 && nozzle(bc_geom_type)){
      vector<IntVector>::const_iterator iter;

      //__________________________________
      //  constants and curve fit for pressure
      double pascals_per_psi = 6894.4;
  
      double mean = 3.2e-02;
      double std  = 1.851850425925377e-02;
      vector<double> c(8);
      c[7] = -2.233767893099373e+05;
      c[6] = -5.824923861605825e+04;
      c[5] =  1.529343908400654e+06;
      c[4] =  1.973796762592652e+05;
      c[3] =  -3.767381404747613e+06;
      c[2] =  5.939587204102841e+04;
      c[1] =  5.692124635957888e+06;    
      c[0] =  3.388928006241853e+06;
      
      double t = sharedState->getElapsedTime();
      double time = t + sharedState->getTimeOffset();

      double tbar = (time - mean)/std;
      
      double stag_press = c[7] * pow(tbar,7) + c[6]*pow(tbar,6) + c[5]*pow(tbar,5)
                       + c[4] * pow(tbar,4) + c[3]*pow(tbar,3) + c[2]*pow(tbar,2)
                       + c[1] * tbar + c[0];
      
      double stag_temp  = (2.7988e3 +110.5*log(stag_press/pascals_per_psi) ); 

      //__________________________________
      //  Isenentrop relations for A/A* = 1.88476
      double T_T0     = 0.97902;
      double Mach     = 0.32725;
      double R        = 287.0;
      double gamma    = 1.4;
      double static_temp = T_T0 * stag_temp;
      
      double velX = Mach * sqrt(gamma * R * static_temp); 
    //  cout << "nozzle Velocity " << velX << " static_temp " << static_temp << endl;     
      if(desc == "Velocity") {
        for (iter=bound.begin(); iter != bound.end(); iter++) {
          var_CC[*iter] = Vector(velX, 0.,0.);
        }
      }
    } 
  #endif  //  ICEBoundaryCond_3

#endif  //running_NGC_nozzle
