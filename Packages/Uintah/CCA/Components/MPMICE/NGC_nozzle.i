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

      double time = d_sharedState->getElapsedTime();
      //double time = t + d_sharedState->getTimeOffset();
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
#endif  //running_NGC_nozzle
