
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

using namespace SCIRun;
using namespace Uintah;

static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false); 
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
    StaticArray<double> kappa(numALLMatls);
    StaticArray<CCVariable<double> > vol_frac(numALLMatls);
    StaticArray<CCVariable<double> > rho_micro(numALLMatls);
    StaticArray<CCVariable<double> > sp_vol_new(numALLMatls);
    StaticArray<CCVariable<double> > rho_CC_new(numALLMatls);
    StaticArray<CCVariable<double> > rho_CC_scratch(numALLMatls);
    StaticArray<CCVariable<double> > speedSound_new(numALLMatls);
    StaticArray<CCVariable<double> > f_theta(numALLMatls);
    StaticArray<CCVariable<double> > matl_press(numALLMatls);

    StaticArray<constCCVariable<double> > Temp(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numALLMatls);
    StaticArray<constCCVariable<double> > mat_vol(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC(numALLMatls);
    StaticArray<constCCVariable<double> > mass_CC(numALLMatls);
    CCVariable<double> press_new, press_copy; 

    //__________________________________
    //  Implicit pressure calc. needs two copies of press 
    new_dw->allocateAndPut(press_new, Ilb->press_equil_CCLabel, 0,patch);
    new_dw->allocateAndPut(press_copy,Ilb->press_CCLabel,       0,patch);
    Ghost::GhostType  gn = Ghost::None;   
   
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(ice_matl){                    // I C E
        old_dw->get(Temp[m],      Ilb->temp_CCLabel,  indx,patch,gn,0);
        old_dw->get(rho_CC[m],    Ilb->rho_CCLabel,   indx,patch,gn,0);
        old_dw->get(sp_vol_CC[m], Ilb->sp_vol_CCLabel,indx,patch,gn,0);
        cv[m]    = ice_matl->getSpecificHeat();
        gamma[m] = ice_matl->getGamma();
      }
      if(mpm_matl){                    // M P M    
        new_dw->get(Temp[m],     MIlb->temp_CCLabel,  indx,patch,gn,0); 
        new_dw->get(mat_vol[m],  MIlb->cVolumeLabel,  indx,patch,gn,0); 
        new_dw->get(mass_CC[m],  MIlb->cMassLabel,    indx,patch,gn,0); 
        cv[m] = mpm_matl->getSpecificHeat();
      }
      new_dw->allocateTemporary(rho_CC_scratch[m], patch);
      new_dw->allocateTemporary(rho_micro[m],      patch);
      new_dw->allocateAndPut(sp_vol_new[m],Ilb->sp_vol_CCLabel,    indx,patch);
      new_dw->allocateAndPut(rho_CC_new[m],Ilb->rho_CCLabel,       indx,patch);
      new_dw->allocateAndPut(vol_frac[m],  Ilb->vol_frac_CCLabel,  indx,patch);
      new_dw->allocateAndPut(f_theta[m],   Ilb->f_theta_CCLabel,   indx,patch);
      new_dw->allocateAndPut(matl_press[m],Ilb->matl_press_CCLabel,indx,patch);
      new_dw->allocateAndPut(speedSound_new[m], 
                                           Ilb->speedSound_CCLabel,indx,patch);
      speedSound_new[m].initialize(0.0);
      if(ice_matl){                    // I C E
       rho_CC_scratch[m].copyData(rho_CC[m]);
      }
    }
    
    press_new.initialize(0.0);


    // This adjusts the amount of ice material in cells that aren't
    // identically full after initialization
    static int tstep=1;
    if(tstep==0){
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){        IntVector c = *iter;
        double total_mat_vol = 0.0;
        double total_ice_vol=0.0;
        for (int m = 0; m < numALLMatls; m++) {
          Material* matl = d_sharedState->getMaterial( m );
          ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
          MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
          if(ice_matl){                // I C E
           rho_micro[m][c]  = 1.0/sp_vol_CC[m][c];
           sp_vol_new[m][c] = sp_vol_CC[m][c];
           mat_mass[m]   = rho_CC[m][c] * cell_vol;
           mat_volume[m] = mat_mass[m] * sp_vol_CC[m][c];
           total_ice_vol+=mat_volume[m];
          }
          if(mpm_matl){                //  M P M
            rho_micro[m][c]  = mass_CC[m][c]/mat_vol[m][c];
            sp_vol_new[m][c] = 1/rho_micro[m][c];
            mat_mass[m]      = mass_CC[m][c];
            mat_volume[m] = mat_vol[m][c];
          }
          total_mat_vol += mat_volume[m];
        }
        // "Fix" the cells that aren't identically full
        if((fabs(total_mat_vol-cell_vol)/cell_vol) > .01){
          if(total_mat_vol > cell_vol){ // For cells that are too full
            double extra_vol = total_mat_vol - cell_vol;
            for (int m = 0; m < numALLMatls; m++) {
             Material* matl = d_sharedState->getMaterial( m );
             ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
             // Remove a proportional amount of each ice material's mass & vol
             if(ice_matl){                // I C E
               mat_volume[m] -= (extra_vol/total_ice_vol)*mat_volume[m];
               mat_mass[m]   -= (extra_vol/total_ice_vol)*mat_mass[m];
               rho_CC_scratch[m][c] = mat_mass[m]/cell_vol;
             }
            } // for ALL matls
           }
           if(total_mat_vol < cell_vol){ // For cells that aren't full enough
            double missing_vol = cell_vol - total_mat_vol;
            for (int m = 0; m < numALLMatls; m++) {
             Material* matl = d_sharedState->getMaterial( m );
             ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
             // Add an equal amount of each ice material's mass & vol
             if(ice_matl){                // I C E
               mat_volume[m] += missing_vol/((double) numICEMatls);
               mat_mass[m]   = mat_volume[m]*rho_micro[m][c];
               rho_CC_scratch[m][c] = mat_mass[m]/cell_vol;
             }
            } // for ALL matls
          }
        } // if cells aren't identically full
      }
    }
    tstep++;

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
         rho_micro[m][c]  = 1.0/sp_vol_CC[m][c];
         sp_vol_new[m][c] = sp_vol_CC[m][c];
         ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m],
                                         cv[m],Temp[m][c],
                                         press_eos[m],dp_drho[m],dp_de[m]);

         mat_mass[m]   = rho_CC_scratch[m][c] * cell_vol;
         mat_volume[m] = mat_mass[m] * sp_vol_CC[m][c];

         tmp = dp_drho[m] + dp_de[m] * 
           (press_eos[m]*(sp_vol_CC[m][c] * sp_vol_CC[m][c]));         
        } 
        if(mpm_matl){                //  M P M
          rho_micro[m][c]  = mass_CC[m][c]/mat_vol[m][c];
          sp_vol_new[m][c] = 1./rho_micro[m][c];
          mat_mass[m]      = mass_CC[m][c];

          mpm_matl->getConstitutiveModel()->
            computePressEOSCM(rho_micro[m][c],press_eos[m], press_ref,
                              dp_drho[m], tmp,mpm_matl);

          mat_volume[m] = mat_vol[m][c];
        }              
        matl_press[m][c] = press_eos[m];
        total_mat_vol += mat_volume[m];
/*`==========TESTING==========*/
    //  speedSound_new[m][c] = sqrt(tmp)/gamma[m];  // Isothermal speed of sound
        speedSound_new[m][c] = sqrt(tmp);           // Isentropic speed of sound
        kappa[m] = sp_vol_new[m][c]/ 
                            (speedSound_new[m][c] * speedSound_new[m][c]); 
/*==========TESTING==========`*/
       }  // for ALLMatls...
       
       
      //__________________________________
      // Compute 1/f_theta
       double f_theta_denom = 0.0;
       for (int m = 0; m < numALLMatls; m++) {
         vol_frac[m][c] = mat_volume[m]/total_mat_vol;
         f_theta_denom += vol_frac[m][c]*kappa[m];
       }
       //__________________________________
       // Compute press_new
       for (int m = 0; m < numALLMatls; m++) {
         f_theta[m][c] = vol_frac[m][c]*kappa[m]/f_theta_denom;
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
        rho_CC_new[m].copyData(rho_CC_scratch[m]);
      }
      if(mpm_matl){
        for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          rho_CC_new[m][c] = mass_CC[m][c]/cell_vol;
        }
      }
    } 
    
    //__________________________________
    //  Update boundary conditions
    // implicit pressure calc. needs two copies of the pressure
    for (int m = 0; m < numALLMatls; m++)   {
      setBC(matl_press[m],rho_micro[SURROUND_MAT],
           "rho_micro", "Pressure", patch, d_sharedState, 0, new_dw);
    }  
    setBC(press_new, rho_micro[SURROUND_MAT], 
          "rho_micro", "Pressure", patch, d_sharedState, 0,  new_dw);
          
    press_copy.copyData(press_new);
    //__________________________________
    // compute sp_vol_CC
    for (int m = 0; m < numALLMatls; m++)   {
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
        IntVector c = *iter;
        sp_vol_new[m][c] = 1.0/rho_micro[m][c];

      }
    } 
    
   //---- P R I N T   D A T A ------   
    if (d_ice->switchDebug_EQ_RF_press) {
      ostringstream desc;
      desc << "BOT_computeRFPress_patch_" << patch->getID();
      d_ice->printData( 0, patch, 1, desc.str(), "Press_CC_RF", press_new);

      for (int m = 0; m < numALLMatls; m++)  {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex(); 
        ostringstream desc;
        desc<< "BOT_computeRFPress_Mat_"<< indx << "_patch_"<<patch->getID();
    #if 0
        d_ice->printData(indx,patch,1,desc.str(),"matl_press",  matl_press[m]);         
        d_ice->printData(indx,patch,1,desc.str(),"f_theta",     f_theta[m]);            
        d_ice->printData(indx,patch,1,desc.str(),"sp_vol_CC",   sp_vol_new[m]);         
    #endif 
        d_ice->printData(indx,patch,1,desc.str(),"rho_CC",      rho_CC_new[m]);         
        d_ice->printData(indx,patch,1,desc.str(),"rho_micro_CC",rho_micro[m]);          
        d_ice->printData(indx,patch,1,desc.str(),"vol_frac_CC", vol_frac[m]);           
      }
    }
  } // patches
}
