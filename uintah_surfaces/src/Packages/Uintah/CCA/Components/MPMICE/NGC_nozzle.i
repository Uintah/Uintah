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
 #ifdef RigidMPM_1
 
      //__________________________________
      //  Hardwiring for Northrup Grumman nozzle
      //  Loop over all the particles and set the velocity
      //  of the 2nd stage = to the hard coded velocity
      //  Note that the pcolor of the 2nd stage has been 
      //  initialized to a non zero value
      constParticleVariable<double> pcolor;
      old_dw->get(pcolor, lb->pColorLabel, pset);

      double time = d_sharedState->getElapsedTime();
      //double time = t + d_sharedState->getTimeOffset();
      vector<double> c(4);
      c[3] = 5.191382077671874e+05;
      c[2] = -6.077090505408908e+03;
      c[1] =  1.048411519176967e+01;
      c[0] =  1.606269799418405e-01;

      double vel_x = c[3]*pow(time,3) + c[2]*pow(time,2) + c[1]*time + c[0];
      Vector hardWired_velocity = Vector(vel_x,0,0);

      if (patch->getID() == 0){
        cout << " hardWired_velocity " << hardWired_velocity << endl;
      }

      for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); 
                                                                          iter++){
        particleIndex idx = *iter;

        if (pcolor[idx]> 0) {   
          pvelocitynew[idx] = hardWired_velocity;
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
    cout << "dataArchiver " << dataArchiver->getCurrentTimestep()<< endl;
    if (dataArchiver->getCurrentTimestep() == 1) {
   
      Vector dx = patch->dCell();
      CellIterator iter=patch->getCellIterator();
      IntVector lo = iter.begin();
      IntVector hi = iter.end();
      const Level* level = getLevel(patches);
      Point this_cell_pos = level->getCellPosition(hi);
      
      cout << "this_cell_pos " << this_cell_pos << endl;
      
      
      int k   = 0;
      int mat = 0;
      //__________________________________
      //  find the interior of the nozzle
      for(int i = lo.x(); i < hi.x(); i++) {
        for(int j = lo.y(); j < hi.y()-1; j++) {
          IntVector c(i, j, k);
          IntVector adj(i, j+1, k);
          Point cell_pos = level->getCellPosition(c);

          // compute vertical gradient of vol_frac
          double gradY = (vol_frac[mat][adj] - vol_frac[mat][c])/dx.y();

          // if cell position is below a certain position

          if(vol_frac[mat][adj] > 0.001 && 
             cell_pos.y() < 0.15 && 
             cell_pos.x() < 0.42){
            if (fabs(gradY)>0.1){
              cout << c << " adj " << vol_frac[mat][adj]
                        << " c " << vol_frac[mat][c]
                        << " gradY " << gradY
                        << " cellPos " << cell_pos << endl;
              j = hi.y();
            }
          } // if in right area
        } // j loop
      }  // i loop

      //__________________________________
      //  Find the edge of the upper dome      
      cout << "UPPER DOME" << endl;
   
      for(int j = lo.y(); j < hi.y()-1; j++) {
        for(int i = hi.x() -1; i > lo.x()+1; i--) {
          IntVector c(i, j, k);
          IntVector adj(i-1, j, k);
          Point cell_pos = level->getCellPosition(c);

          // compute vertical gradient of vol_frac
          double gradX = (vol_frac[mat][adj] - vol_frac[mat][c])/dx.x();

          // if cell position is below a certain position

          if(fabs(vol_frac[mat][adj]) > 0.001 && 
             (c.y() < 170  && c.y() > 65) &&
             (c.x() < 158 && c.x() >118) ){
           //  cout << c << " gradX " << gradX << endl;
            if (fabs(gradX)>0.1){
              cout << c << " adj " << vol_frac[mat][adj]
                        << " c " << vol_frac[mat][c]
                        << " gradX " << gradX
                        << " cellPos " << cell_pos << endl;
              i = lo.x()+1;
            }
          } // if in right area
        } // i loop
      }  // j
    }
  #endif
#endif  //running_NGC_nozzle
