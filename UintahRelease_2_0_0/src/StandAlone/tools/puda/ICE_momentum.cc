
#include <StandAlone/tools/puda/ICE_momentum.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

/*______________________________________________________________________
                   ICE:MOMENTUM
 
 This performs a momentum anaylsis on the control volume.
 From this analysis you can compute the forces on the control volume using
 
      dmom_cv/dt(i) = (mom_cv(i) - mom)_cv(i-1))/(t(i) - t(i-1));
      
      Force(i)    = dmom_cv/dt(i) + flux(i);
      
      Force  - | Time rate of change of momentum |   +   | net flow rate of momentum  |
               | inside the control volume       |       | out of the control surface |
      
      Force = body forces + surface forces
      
 This assumes that the following  variables have been saved
      <save label = "rho_CC"/>
      <save label = "vel_CC"/>
      
      <save label = "pressX_FC"/>
      <save label = "pressY_FC"/>
      <save label = "pressZ_FC"/>
      
      <save label = "uvel_FCME"/>
      <save label = "vvel_FCME"/>
      <save label = "wvel_FCME"/>      
      
      <save label = "tau_X_FC"/>
      <save label = "tau_Y_FC"/>
      <save label = "tau_Z_FC"/>

The outut is saved to the file momentumAnalysis.txt and it contains:




//______________________________________________________________________*/


void
Uintah::ICE_momentum( DataArchive * da, CommandLineFlags & clf )
{     
  vector<int> index;
  vector<double> times;
  string faceName[] = { "x-", "x+", "y-", "y+", "z-", "z+"};
  
  da->queryTimesteps(index, times);
  
  ASSERTEQ(index.size(), times.size());
  cout << "There are " << index.size() << " timesteps:\n";
  for( int i = 0; i < (int)index.size(); i++ ) {
    cout << index[i] << ": " << times[i] << endl;
  }
  
  
  string filename = "momentumAnalysis.txt";
  // write header to file
  FILE *fp;
  fp = fopen( filename.c_str(),"w");       
  fprintf(fp, "#                                                 total momentum in the control volume                                          Net convective momentum flux                                               net viscous flux                                                             pressure force on control vol.\n");                                                                      
  fprintf(fp, "#Time                    CV_mom.x                 CV_mom.y                  CV_mom.z                  momFlux.x               momFlux.y                momFlux.z                 visFlux.x                 visFlux.y                visFlux.z                 pressForce.x              pressForce.y             pressForce.z            mDot.x                    mDot.y                    mDot.z\n");
  fclose(fp);
  
      
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
  
  //______________________________________________________________________
  //  Timestep loop
  for(unsigned long t=clf.time_step_lower; t<=clf.time_step_upper; t+=clf.time_step_inc){
  
    if( t == 0 ){    // ignore the zero-th timestep
      continue;       
    }
  
    double time = times[t];
    GridP grid = da->queryGrid(t);
    int numLevels = grid->numLevels();

    //__________________________________
    //  Level loop
    for(int l=0;l<numLevels;l++){
      LevelP level = grid->getLevel(l);
      double vol = level->cellVolume();
    
      //______________________________________________________________________
      // ICE: total momentum in the control volume
      Vector totalCV_mom ( 0.0 );
      double ICE_TotalMass   = 0.0;
      
      for(Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){

        const Patch* patch = *iter;
        int matl = clf.matl;
        CCVariable<double> rho_CC;
        CCVariable<Vector> vel_CC;

        da->query( rho_CC,  "rho_CC",  matl, patch, t);
        da->query( vel_CC,  "vel_CC",  matl, patch, t);
        
        //__________________________________
        //  Sum contributions over patch        
        for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          ICE_TotalMass   += rho_CC[c] * vol;
          totalCV_mom    += rho_CC[c] * vol * vel_CC[c];
        }
      } // for patches
      
      
      //______________________________________________________________________
      //
      // momentum fluxes
      // keep track of the contributions on each face.
      map< int, double > sumMdot_map;
      map< int, Vector > mom_faceFlux;
      map< int, double > pressForce;
      map< int, Vector > viscousForce;
      
      for (int f = 0; f<Patch::numFaces; f++){
        sumMdot_map[f]    = 0.;
        pressForce[f]     = 0.;
        mom_faceFlux[f]   = Vector(0.);
        viscousForce[f]   = Vector(0.);
      }
      
      
      for(Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){

        const Patch* patch = *iter;
        Vector dx = patch->dCell();
        
        const int pressMatl = 0;
        const int matl      = clf.matl;

        //__________________________________
        //  bulletproofing
        // do the required variables exist?
        vector<string> vars;
        vector<const Uintah::TypeDescription*> types;

        da->queryVariables(vars, types);
        int vars_found = 0;

        for (unsigned int i = 0; i < vars.size(); i++) {
          if ( "rho_CC"     == vars[i] ||
               "vel_CC"     == vars[i] ||
               "uvel_FCME"  == vars[i] || "vvel_FCME"  == vars[i] || "wvel_FCME" == vars[i] ||
               "pressX_FC"  == vars[i] || "pressY_FC"  == vars[i] || "pressZ_FC" == vars[i] ||
               "tau_X_FC"   == vars[i] || "tau_Y_FC"   == vars[i] || "tau_Z_FC"  == vars[i] ) {
            vars_found +=1;
          }
        }

        if (vars_found != 11 ) {
          ostringstream warn;
          warn << "The required variables ( rho_CC, vel_CC, (u,v,w)vel_FCME, press(X,Y,Z)_FC, tau_(X,Y,Z)_FC "
               << ") were not found in the uda."  ;
          throw ProblemSetupException(warn.str(),__FILE__,__LINE__);        
        }
        
        //__________________________________
        //  Now pull data from archive
        CCVariable<double>   rho_CC;
        CCVariable<Vector>   vel_CC;
        
        SFCXVariable<double> uvel_FC, pressX_FC;
        SFCYVariable<double> vvel_FC, pressY_FC;
        SFCZVariable<double> wvel_FC, pressZ_FC;

        SFCXVariable<Vector>  tau_X_FC;
        SFCYVariable<Vector>  tau_Y_FC;
        SFCZVariable<Vector>  tau_Z_FC;
        
        da->query( rho_CC,    "rho_CC",    matl,      patch, t );
        da->query( vel_CC,    "vel_CC",    matl,      patch, t );
        
        da->query( uvel_FC,   "uvel_FCME", matl,      patch, t );
        da->query( vvel_FC,   "vvel_FCME", matl,      patch, t );
        da->query( wvel_FC,   "wvel_FCME", matl,      patch, t );

        da->query( pressX_FC, "pressX_FC", pressMatl, patch, t );
        da->query( pressY_FC, "pressY_FC", pressMatl, patch, t );
        da->query( pressZ_FC, "pressZ_FC", pressMatl, patch, t );
        
        da->query( tau_X_FC, "tau_X_FC",   matl,      patch, t );
        da->query( tau_Y_FC, "tau_Y_FC",   matl,      patch, t );
        da->query( tau_Z_FC, "tau_Z_FC",   matl,      patch, t );
                
        //__________________________________
        // Sum the momentum fluxes passing through the boundaries
        // Sum the surface forces on each face      
        vector<Patch::FaceType> bf;
        patch->getBoundaryFaces(bf);

        for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
        
          Patch::FaceType face = *itr;
          string faceName = patch->getFaceName(face );

          // define the iterator on this face 
          Patch::FaceIteratorType SFC = Patch::SFCVars;
          CellIterator iterLimits=patch->getFaceIterator(face, SFC);    
                    
          //__________________________________
          //           X faces
          if ( face == Patch::xminus || face == Patch::xplus ) {    
            double area = dx.y() * dx.z();
            
            cout << "    X_iterLimits: " << iterLimits <<  endl;

            for(CellIterator iter = iterLimits; !iter.done();iter++) {
              IntVector c = *iter; 
              
              double vel = uvel_FC[c];

              // find upwind cell
              IntVector uw = c;
              if (vel > 0 ){
                uw.x( uw.x() -1 );
              }

              // use upwinded values
              double mdot         = vel * area * rho_CC[uw];
              sumMdot_map[face]  += mdot;
              mom_faceFlux[face] += mdot * vel_CC[uw];
              pressForce[face]   += pressX_FC[c] * area;
              viscousForce[face] += ( tau_X_FC[c] * area );
              //cout << "face: " << faceName << " c: " << c << " c_FC: " << c_FC << " vel = " << vel << " mdot = " << mdot << endl;
            }
          }

          //__________________________________
          //        Y faces
          if (face == Patch::yminus || face == Patch::yplus) {    
            double area = dx.x() * dx.z();

            cout << "    Y_iterLimits: " << iterLimits  << endl;
            
            for(CellIterator iter = iterLimits; !iter.done();iter++) {
              IntVector c = *iter;
              
              double vel = vvel_FC[c];

              // find upwind cell
              IntVector uw = c;
              if (vel > 0 ){
                uw.y( uw.y() -1 );
              }
              
              // use upwinded values
              double mdot   = vel * area * rho_CC[uw];
              sumMdot_map[face]  += mdot;
              mom_faceFlux[face] += mdot * vel_CC[uw];
              pressForce[face]   += pressY_FC[c] * area;
              viscousForce[face] += ( tau_Y_FC[c] * area );

              //cout << "face: " << faceName << " c: " << c << " offset: " << offset << " vel = " << vel << " mdot = " << mdot << endl;
            }
          }

          //__________________________________
          //        Z faces
          if (face == Patch::zminus || face == Patch::zplus) {
            double area = dx.x() * dx.y();

            cout << "    Z_iterLimits: " << iterLimits  << endl;

            for(CellIterator iter = iterLimits; !iter.done();iter++) {
              IntVector c = *iter;
              
              double vel = wvel_FC[c];

              // find upwind cell
              IntVector uw = c;
              if (vel > 0 ){
                uw.z( uw.z() -1 );
              }
              
              // use upwinded values
              double mdot   = vel * area * rho_CC[uw];
              sumMdot_map[face]  += mdot;
              mom_faceFlux[face] += mdot * vel_CC[uw];
              pressForce[face]   += pressZ_FC[c] * area;
              viscousForce[face] = ( area * tau_Z_FC[c]);
            }
          }
        }  // boundary faces          
      } // for patches
      
      //__________________________________
      //  output to stdout
      for (int f = 0; f<Patch::numFaces; f++){
        cout << "    face: " << faceName[f] << " momentum faceFlux: " << mom_faceFlux[f] << " pressForce: " << pressForce[f] << "\t sumMdot; " << sumMdot_map[f] <<"\n"<< endl;
      }
      
      //__________________________________
      // write data 
      // net momentum flux through the control surfaces
      double mom_X_flux  = ( mom_faceFlux[0].x() - mom_faceFlux[1].x() ) + 
                           ( mom_faceFlux[2].x() - mom_faceFlux[3].x() ) + 
                           ( mom_faceFlux[4].x() - mom_faceFlux[5].x() );
                           
      double mom_Y_flux  = ( mom_faceFlux[0].y() - mom_faceFlux[1].y() ) + 
                           ( mom_faceFlux[2].y() - mom_faceFlux[3].y() ) + 
                           ( mom_faceFlux[4].y() - mom_faceFlux[5].y() );
                           
      double mom_Z_flux  = ( mom_faceFlux[0].z() - mom_faceFlux[1].z() ) + 
                           ( mom_faceFlux[2].z() - mom_faceFlux[3].z() ) + 
                           ( mom_faceFlux[4].z() - mom_faceFlux[5].z() );
                           
      
      // momentum flux due to viscous diffusion                   
      double vis_X_flux  = ( viscousForce[0].x() - viscousForce[1].x() ) + 
                           ( viscousForce[2].x() - viscousForce[3].x() ) + 
                           ( viscousForce[4].x() - viscousForce[5].x() );
                           
      double vis_Y_flux  = ( viscousForce[0].y() - viscousForce[1].y() ) + 
                           ( viscousForce[2].y() - viscousForce[3].y() ) + 
                           ( viscousForce[4].y() - viscousForce[5].y() );
                           
      double vis_Z_flux  = ( viscousForce[0].z() - viscousForce[1].z() ) + 
                           ( viscousForce[2].z() - viscousForce[3].z() ) + 
                           ( viscousForce[4].z() - viscousForce[5].z() );
                           
      // net mass flow rate (diagnostic)
      double mDot_X      = sumMdot_map[0] - sumMdot_map[1];
      double mDot_Y      = sumMdot_map[2] - sumMdot_map[3];
      double mDot_Z      = sumMdot_map[4] - sumMdot_map[5];
      
      // net force on control volume due to pressure forces
      double pressForceX = pressForce[0] - pressForce[1];
      double pressForceY = pressForce[2] - pressForce[3];
      double pressForceZ = pressForce[4] - pressForce[5];
      
      
      
      FILE *fp;
      fp = fopen( filename.c_str(),"a");
      fprintf(fp, "%16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,    %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E\n", 
                  time, 
                  totalCV_mom.x(),
                  totalCV_mom.y(),
                  totalCV_mom.z(),
                  mom_X_flux,
                  mom_Y_flux,
                  mom_Z_flux,
                  vis_X_flux,
                  vis_Y_flux,
                  vis_Z_flux,                  
                  pressForceX,
                  pressForceY,
                  pressForceZ,
                  mDot_X,
                  mDot_Y,
                  mDot_Z );

      fclose(fp);  
    }  // timestep loop
  } // for levels
} // end todd2()
