
#include <StandAlone/tools/puda/todd1.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/InvalidGrid.h>

#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

/*______________________________________________________________________
                   T O D D 1
 
 This performs a thermodynamic first law anaylsis on the control volume.
 From this analysis you can compute Q into the CV using
 
      dUdt(i) = (Utot(i) - Utot(i-1))/(t(i) - t(i-1));
      Q(i)    = dUdt(i) + flux(i);
 
 NOTE: the user must changed some hardwired thermodynamic constants
       in the code. 
      
 This assumes that the following MPMICE variables have been saved

    <save label="vel_CC"    material="1"/>
    <save label="rho_CC"    material="1"/>
    <save label="temp_CC"   material="1"/>
    <save label="uvel_FCME" material="1"/>
    <save label="vvel_FCME" material="1"/>
    <save label="wvel_FCME" material="1"/>

    <save label="p.mass"/>
    <save label="p.velocity"/>
    <save label="p.temperature"/>
    
The outut is saved to the file todd1.out and it contains:

#Time                      ICE_totalIntEng            MPM_totalIntEng             totalIntEng                 total_ICE_Flux                 XFace_flux                 YFace_flux                 ZFace_flux
1.000002252673294E-02      1.480199278565353E+00      1.184298599959503E+00       2.664497878524856E+00       3.128669396857262E-01      -4.391694926905313E+01       0.000000000000000E+00       4.422981620873943E+01
2.000023455777600E-02      1.480250100320916E+00      1.211657541707296E+00       2.691907642028212E+00       4.039349103612686E-01      -4.705808690814945E+01       0.000000000000000E+00       4.746202181851062E+01
3.000004353644865E-02      1.480285917855526E+00      1.230394596050221E+00       2.710680513905747E+00       5.291662033523892E-01      -4.003711816747943E+01       0.000000000000000E+00       4.056628437083164E+01


//______________________________________________________________________*/


void
Uintah::todd1( DataArchive * da, CommandLineFlags & clf )
{     
  vector<int> index;
  vector<double> times;
  
  da->queryTimesteps(index, times);
  
  ASSERTEQ(index.size(), times.size());
  cout << "There are " << index.size() << " timesteps:\n";
  for( int i = 0; i < (int)index.size(); i++ ) {
    cout << index[i] << ": " << times[i] << endl;
  }
  
  // write header to file
  FILE *fp;
  fp = fopen("todd1.out","w");
  fprintf(fp, "#Time                      ICE_totalIntEng            MPM_totalIntEng             totalIntEng                 total_ICE_Flux                 XFace_flux                 YFace_flux                 ZFace_flux\n");
  fclose(fp);
  
      
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
  
  //______________________________________________________________________
  //  Timestep loop
  for(unsigned long t=clf.time_step_lower; t<=clf.time_step_upper; t+=clf.time_step_inc){
    double time = times[t];
    GridP grid = da->queryGrid(t);
    int numLevels = grid->numLevels();

    //__________________________________
    //  Level loop
    for(int l=0;l<numLevels;l++){
      LevelP level = grid->getLevel(l);
    
      
      //__________________________________
      // MPM: total internal energy
      
      double MPM_TotalIntEng = 0.0;
      
      for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){

        const Patch* patch = *iter;
        
        int matl  = 0;                       // HARDWIRED
        double Cp = 234.0 ;                  // HARDWIRED
        
        ParticleVariable<double> pTempNew;
        ParticleVariable<double> pMass;
        
        da->query(pTempNew,  "p.temperature",  matl, patch, t);
        da->query(pMass,     "p.mass",         matl, patch, t);
        ParticleSubset* pset = pTempNew.getParticleSubset();  
        
        if( pset->numParticles() > 0 ){  // are there particles on this patch
          for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
            particleIndex idx = *iter;    

            MPM_TotalIntEng += pTempNew[idx] * pMass[idx] * Cp;
          } // particle Loop  
        }
      } // for patches
      

      //__________________________________
      // ICE: total internal energy
      double ICE_TotalIntEng = 0.0;
      
      for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){

        const Patch* patch = *iter;
        Vector dx = patch->dCell();
        double vol = dx.x() * dx.y() * dx.z();
        
        int matl = 1;                         // HARDWIRED
        double cv = 717.5;                    // HARDWIRED

        CCVariable<double> rho_CC;
        CCVariable<double> temp_CC;

        da->query(rho_CC,  "rho_CC",  matl, patch, t);
        da->query(temp_CC, "temp_CC", matl, patch, t);
        
        //__________________________________
        //  Sum contributions over patch        
        for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          ICE_TotalIntEng += rho_CC[c] * vol * cv * temp_CC[c];
        }
          
      } // for patches
      
      //__________________________________
      // ICE: fluxes
      double total_flux = 0.0;
      double d_conversion = 1.0;              // HARDWIRED
      
      // keep track of the contributions on each face.
      map< int,double > sumKE_map;
      map< int,double > sumH_map;
      map< int,double > sumMdot_map;
      map< int,double > faceFlux;
      
      for (int f = 0; f<Patch::numFaces; f++){
        sumKE_map[f]   = 0;
        sumH_map[f]    = 0;
        sumMdot_map[f] = 0;
        faceFlux[f]    = 0;
      }
      
      
      for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){

        const Patch* patch = *iter;
        Vector dx = patch->dCell();

        int matl = 1;                   // HARDWIRED
        double cv           = 717.5;    // HARDWIRED
        double gamma        = 1.4;      // HARDWIRED
        double patch_F_flux = 0;

        CCVariable<double> rho_CC;
        CCVariable<double> temp_CC;
        SFCXVariable<double> uvel_FC;
        SFCYVariable<double> vvel_FC;
        SFCZVariable<double> wvel_FC;
    
    
        da->query(rho_CC,  "rho_CC",    matl, patch, t);
        da->query(temp_CC, "temp_CC",   matl, patch, t);
        da->query(uvel_FC, "uvel_FCME", matl, patch, t);
        da->query(vvel_FC, "vvel_FCME", matl, patch, t);
        da->query(wvel_FC, "wvel_FCME", matl, patch, t);
                
        //__________________________________
        // Sum the fluxes passing through the boundaries      
        vector<Patch::FaceType> bf;
        patch->getBoundaryFaces(bf);

        for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
          Patch::FaceType face = *itr;
          string faceName = patch->getFaceName(face );

          // define the iterator on this face 
          Patch::FaceIteratorType SFC = Patch::SFCVars;
          CellIterator iterLimits=patch->getFaceIterator(face, SFC);

          IntVector axes = patch->getFaceAxes(face);
          int P_dir = axes[0];  // principal direction
          double plus_minus_one = (double) patch->faceDirection(face)[P_dir];  
          
          double sumKE   = 0.0;
          double sumH    = 0.0;
          double sumMdot = 0.0;
          double f_fluxes = 0.0;
                    
          //__________________________________
          //           X faces
          if ( face == Patch::xminus || face == Patch::xplus ) {    
            double area = dx.y() * dx.z();

            for(CellIterator iter = iterLimits; !iter.done();iter++) {
              IntVector c = *iter;
              double vel = uvel_FC[c];

              // compute the average values
              IntVector cc = c - IntVector(1,0,0);

              double mdot   = plus_minus_one * vel * area * (rho_CC[c] + rho_CC[cc])/2.0; 
              double KE     = 0.5 * vel * vel;
              double enthpy = ( temp_CC[c]  * gamma  * cv + 
                                temp_CC[cc] * gamma * cv )/2.0;

              sumKE   += mdot * KE;
              sumH    += mdot * enthpy;
              sumMdot += mdot;

              f_fluxes +=  mdot * (enthpy + KE * d_conversion);
            }
          }

          //__________________________________
          //        Y faces
          if (face == Patch::yminus || face == Patch::yplus) {    
            double area = dx.x() * dx.z();

            for(CellIterator iter = iterLimits; !iter.done();iter++) {
              IntVector c = *iter;
              double vel = vvel_FC[c];

              // compute the average values
              IntVector cc = c - IntVector(0,1,0);

              double mdot   = plus_minus_one * vel * area * (rho_CC[c] + rho_CC[cc])/2.0; 
              double KE     = 0.5 * vel * vel;
              double enthpy = ( temp_CC[c]  * gamma  * cv + 
                                temp_CC[cc] * gamma * cv )/2.0;

              sumH     += mdot * enthpy; 
              sumKE    += mdot * KE;
              sumMdot  += mdot;
              f_fluxes +=  mdot * (enthpy + KE * d_conversion);
            }
          }


          //__________________________________
          //        Z faces
          if (face == Patch::zminus || face == Patch::zplus) {
            double area = dx.x() * dx.y();
            
            for(CellIterator iter = iterLimits; !iter.done();iter++) {
              IntVector c = *iter;
              double vel = wvel_FC[c];            

              // compute the average values
              IntVector cc = c - IntVector(0,0,1);

              double mdot   = plus_minus_one * vel * area * (rho_CC[c] + rho_CC[cc])/2.0; 
              double KE     = 0.5 * vel * vel;
              double enthpy = ( temp_CC[c]  * gamma  * cv + 
                                temp_CC[cc] * gamma * cv )/2.0;

              sumH     += mdot * enthpy;
              sumKE    += mdot * KE;
              sumMdot  += mdot;
              f_fluxes +=  mdot * (enthpy + KE * d_conversion);
            }
          }
          patch_F_flux      += f_fluxes;
          sumKE_map[face]   += sumKE;
          sumH_map[face]    += sumH;
          sumMdot_map[face] += sumMdot;
          faceFlux[face]    += f_fluxes;
          
          
        }  // boundary faces
      
        total_flux += patch_F_flux;
          
      } // for patches
      
      cout << time << " Internal Energy: ICE: " << ICE_TotalIntEng << " MPM: " << MPM_TotalIntEng 
                   << " total: " << ICE_TotalIntEng + MPM_TotalIntEng << " total_flux " << total_flux <<endl;
      
      for (int f = 0; f<Patch::numFaces; f++){
        cout << "    face: " << f << " faceFlux: " << faceFlux[f] << "\t sumKE: " << sumKE_map[f] << "\t sumH: " << sumH_map[f] << "\t sumMdos; " << sumMdot_map[f] <<"\n"<< endl;
      }
      
      //__________________________________
      // write data 
      double totalIntEng = ICE_TotalIntEng + MPM_TotalIntEng;
      double X_flux      = faceFlux[0] + faceFlux[1];
      double Y_flux      = faceFlux[2] + faceFlux[3];
      double Z_flux      = faceFlux[4] + faceFlux[5];    

      FILE *fp;
      fp = fopen("todd1.out","a");
      fprintf(fp, "%16.15E      %16.15E      %16.15E       %16.15E       %16.15E      %16.15E       %16.15E       %16.15E\n", time, 
                  ICE_TotalIntEng, 
                  MPM_TotalIntEng, 
                  totalIntEng,
                  total_flux,
                  X_flux,
                  Y_flux,
                  Z_flux );

      fclose(fp);  
    }  // timestep loop
  } // for levels
} // end todd1()
