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
      
      Vector hardWired_velocity = Vector(0,0,0);    // put relationship here
      if (time > 0.012589) {                        // root of curve fit
        hardWired_velocity = Vector(20.0,0,0);
      }

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
              gacceleration[0][ni[k]]   = (hardWired_velocity - gvelocity_star_org[ni[k]])/delT;
	       gvelocity_star[0][ni[k]]  = hardWired_velocity; 
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
      double mean = 3.205751633986928e-02;
      double std  = 2.020912741291855e-02;
      vector<double> c(8);
      c[7] = -2.084666863540239e+05;
      c[6] = -2.768180990368879e+04;
      c[5] =  1.216851029348476e+06;
      c[4] =  8.665056462676427e+04;

      c[3] = -2.554361910136362e+06;
      c[2] =  6.697085748135508e+04;
      c[1] =  3.276185586871563e+06;    
      c[0] =  1.794787251771923e+06;
      
      
      double t = sharedState->getElapsedTime();
      double time = t + sharedState->getTimeOffset();
      
      double tbar = (time - mean)/std;
      
      double pressfit = c[7] * pow(tbar,7) + c[6]*pow(tbar,6) + c[5]*pow(tbar,5)
                      + c[4] * pow(tbar,4) + c[3]*pow(tbar,3) + c[2]*pow(tbar,2)
                      + c[1] * tbar + c[0];
      
      for (iter=bound.begin(); iter != bound.end(); iter++) {
        press_CC[*iter] = pressfit;
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
      double mean = 3.205751633986928e-02;
      double std  = 2.020912741291855e-02;
      double temp_static_stag =0.83333333;
      
      double gamma = 1.4;
      double cv    = 716;
      
      vector<double> c(8);
      c[7] = -2.084666863540239e+05;
      c[6] = -2.768180990368879e+04;
      c[5] =  1.216851029348476e+06;
      c[4] =  8.665056462676427e+04;

      c[3] = -2.554361910136362e+06;
      c[2] =  6.697085748135508e+04;
      c[1] =  3.276185586871563e+06;    
      c[0] =  1.794787251771923e+06;
      
      double t = sharedState->getElapsedTime();
      double time = t + sharedState->getTimeOffset();

      double tbar = (time - mean)/std;
      
      double pressfit = c[7] * pow(tbar,7) + c[6]*pow(tbar,6) + c[5]*pow(tbar,5)
                      + c[4] * pow(tbar,4) + c[3]*pow(tbar,3) + c[2]*pow(tbar,2)
                      + c[1] * tbar + c[0];
      
      double static_temp  = temp_static_stag*(2.7988e3 +110.5*log(pressfit/pascals_per_psi) ); 
      double static_rho   = pressfit/((gamma - 1.0) * cv * static_temp);

      //cout << " time " << time << " pressfit " << pressfit
      //     << " static_temp " << static_temp << " static_rho " << static_rho << endl;
      
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

#endif  //running_NGC_nozzle
