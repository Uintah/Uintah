#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
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
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h> 

using namespace SCIRun;
using namespace Uintah;

static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false);

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputeDiv_vol_frac_vel_CC--
_____________________________________________________________________*/
void ICE::schedulecomputeDivThetaVel_CC(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSubset* ice_matls,
                                          const MaterialSubset* mpm_matls,
                                          const MaterialSet* all_matls)
{
  Task* t;
  cout_doing << "ICE::schedulecomputeDivThetaVel_CC" << endl;
  t = scinew Task("ICE::computeDivThetaVel_CC",
                   this, &ICE::computeDivThetaVel_CC); 
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW,lb->vol_frac_CCLabel,  /*all_matls*/ gac,1);
  t->requires(Task::OldDW,lb->vel_CCLabel,         ice_matls,  gac,1);
  t->requires(Task::NewDW,lb->vel_CCLabel,         mpm_matls,  gac,1);  
  t->computes(lb->DLabel);
  sched->addTask(t, patches, all_matls);    
}

/* ---------------------------------------------------------------------
 Function~  ICE::actuallyComputeStableTimestepRF--
 Reference:  See MF document pg 30 and 
_____________________________________________________________________*/
void ICE::actuallyComputeStableTimestepRF(const ProcessorGroup*,  
                                    const PatchSubset* patches,
                                    const MaterialSubset* /*matls*/,
                                    DataWarehouse* /*old_dw*/,
                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Compute Stable Timestep RF on patch " << patch->getID() 
         << "\t\t ICE" << endl;
      
    Vector dx = patch->dCell();
    double delt_CFL = 1e3, delt_cond = 1e3, delt;
    double inv_sum_invDelx_sqr = 1.0/( 1.0/(dx.x() * dx.x()) 
                                     + 1.0/(dx.y() * dx.y()) 
                                     + 1.0/(dx.z() * dx.z())  );
                                     
    constCCVariable<double> speedSound, sp_vol_CC, rho_CC;
    constCCVariable<Vector> vel_CC;
    Ghost::GhostType  gac = Ghost::AroundCells;    
    if (d_CFL > 0.5) {
      throw ProblemSetupException("CFL can't exceed 0.5 for RF problems");
    }
    
    vector<IntVector> adj_offset(3);                   
    adj_offset[0] = IntVector(-1, 0, 0);    // X 
    adj_offset[1] = IntVector(0, -1, 0);    // Y 
    adj_offset[2] = IntVector(0,  0, -1);   // Z     
    Vector delt_tmp;
    Vector include_delT= Vector(0,0,0);
    IntVector numCells(patch->getHighIndex() - patch->getLowIndex());
    //__________________________________
    // which dimensions are relevant
    int mat_id = 0;
    if ( patch->getBCValues(mat_id,"Symmetric",Patch::xminus) || 
         patch->getBCValues(mat_id,"Symmetric",Patch::xplus)  || 
         numCells(0) >3 ) {
     include_delT[0] = 1.0;
    }
    if ( patch->getBCValues(mat_id,"Symmetric",Patch::yminus) || 
         patch->getBCValues(mat_id,"Symmetric",Patch::yplus)  || 
         numCells(1) >3 ) {
     include_delT[1] = 1.0;
    }
    if ( patch->getBCValues(mat_id,"Symmetric",Patch::zminus) || 
         patch->getBCValues(mat_id,"Symmetric",Patch::zplus)  || 
         numCells(2) >3 ) {
     include_delT[2] = 1.0;
    }   
      
    for (int m = 0; m < d_sharedState->getNumMatls(); m++) {
      Material* matl = d_sharedState->getMaterial(m);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx= matl->getDWIndex(); 
        
      new_dw->get(speedSound, lb->speedSound_CCLabel, indx,patch,gac, 1);
      new_dw->get(vel_CC,     lb->vel_CCLabel,        indx,patch,gac, 1);
      new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gac, 1);     
      
      //__________________________________
      //  stability constraint due to courant Condition
      for (int dir = 0; dir <3; dir++) {  //loop over all three directions
        delt_tmp[dir] = 1000.0; 
        for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
          IntVector R = *iter;
          IntVector L = R + adj_offset[dir];

          double ave_vel= 0.5 * fabs( vel_CC[R].length() + 
                                      vel_CC[L].length() ); 

          double kappa_R= sp_vol_CC[R]/( speedSound[R] * speedSound[R] );          
          double kappa_L= sp_vol_CC[L]/( speedSound[L] * speedSound[L] );          

          double tmp    = (sp_vol_CC[L] + sp_vol_CC[R])/(kappa_L + kappa_R);                    
          double cstar  = sqrt(tmp) + fabs(vel_CC[R].length() -
                                           vel_CC[L].length());  

          double A = d_CFL * dx[dir]/(cstar + ave_vel);
          delt_tmp[dir]       = std::min(delt_tmp[dir],A);
        }
      }  //  dir loop
      
      // see page 30 and 46 of MF document
      double matl_delt = 1.0/(include_delT[0]/delt_tmp[0] + 
                              include_delT[1]/delt_tmp[1] + 
                              include_delT[2]/delt_tmp[2] );
           
      delt_CFL = std::min(delt_CFL, matl_delt);
      
//    cout << " delT Based on currant number "<< delt_CFL << endl;
      //__________________________________
      // stability constraint due to heat conduction
      //  I C E  O N L Y
      if (ice_matl) {
        double thermalCond = ice_matl->getThermalConductivity();
        if (thermalCond !=0) {
          double cv    = ice_matl->getSpecificHeat();
          double gamma = ice_matl->getGamma();
          double cp = cv * gamma;

          for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
            IntVector c = *iter;
            double inv_thermalDiffusivity = cp/(sp_vol_CC[c] * thermalCond);
            double A =  0.5 * inv_sum_invDelx_sqr * inv_thermalDiffusivity;
            delt_cond = std::min(A, delt_cond);
          }
        }  //
      }  // ice_matl     
    }  //  matl loop
//  cout << "delT based on conduction "<< delt_cond<<endl;
    delt = std::min(delt_CFL, delt_cond);
    delt = std::min(delt, d_initialDt);
    d_initialDt = 10000.0;

    //__________________________________
    //  Bullet proofing
    if(delt< 1e-20) {  
      string warn = " E R R O R \n ICE::ComputeStableTimestepRF: delT < 1e-20";
      throw InvalidValue(warn);
    }
    
    new_dw->put(delt_vartype(delt), lb->delTLabel);
  }  // patch loop
  //  update when you should dump debugging data. 
  d_dbgNextDumpTime = d_dbgOldTime + d_dbgOutputInterval;
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
    StaticArray<CCVariable<double> > sp_vol_new(numMatls);   
    StaticArray<constCCVariable<double> > rho_CC(numMatls);
    StaticArray<constCCVariable<double> > Temp(numMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);

    constCCVariable<double> press;
    CCVariable<double> press_new, press_copy; 
    Ghost::GhostType  gn = Ghost::None;
    //__________________________________
    //  Implicit pressure calc. needs two copies of press 
    old_dw->get(press,                lb->press_CCLabel, 0,patch,gn, 0); 
    new_dw->allocateAndPut(press_new, lb->press_equil_CCLabel, 0,patch);
    new_dw->allocateAndPut(press_copy,lb->press_CCLabel,       0,patch);

    
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* matl = d_sharedState->getICEMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(Temp[m],      lb->temp_CCLabel,  indx,patch,gn,0);
      old_dw->get(rho_CC[m],    lb->rho_CCLabel,   indx,patch,gn,0);
      old_dw->get(sp_vol_CC[m], lb->sp_vol_CCLabel,indx,patch,gn,0);
      new_dw->allocateAndPut(sp_vol_new[m], lb->sp_vol_CCLabel,    indx,patch);  
      new_dw->allocateAndPut(rho_CC_new[m], lb->rho_CCLabel,       indx,patch);  
      new_dw->allocateAndPut(vol_frac[m],   lb->vol_frac_CCLabel,  indx,patch);  
      new_dw->allocateAndPut(f_theta[m],    lb->f_theta_CCLabel,   indx,patch);  
      new_dw->allocateAndPut(matl_press[m], lb->matl_press_CCLabel,indx,patch);  
      new_dw->allocateAndPut(speedSound_new[m], lb->speedSound_CCLabel,
                                                                   indx,patch);
      new_dw->allocateTemporary(rho_micro[m],  patch);
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

        mat_volume[m] = (rho_CC[m][c] * cell_vol) * sp_vol_CC[m][c];

        tmp = dp_drho[m] + dp_de[m] * 
           (matl_press[m][c] * (sp_vol_CC[m][c] * sp_vol_CC[m][c]));            

        total_mat_vol += mat_volume[m];

//      speedSound_new[m][c] = sqrt(tmp)/gamma[m];  // Isothermal speed of sound
        speedSound_new[m][c] = sqrt(tmp);           // Isentropic speed of sound
        compressibility[m] = sp_vol_CC[m][c]/ 
                            (speedSound_new[m][c] * speedSound_new[m][c]); 

       } 
      //__________________________________
      // Compute 1/f_theta
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
    //  Set BCs matl_press, press
    // implicit pressure calc. needs two copies of the pressure
    for (int m = 0; m < numMatls; m++)   {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      setBC(matl_press[m], rho_micro[SURROUND_MAT],
            "rho_micro", "Pressure", patch, d_sharedState, indx, new_dw);
    }  
    setBC(press_new,  rho_micro[SURROUND_MAT],
         "rho_micro", "Pressure", patch, d_sharedState, 0, new_dw);
         
    press_copy.copyData(press_new);
    //__________________________________
    // carry rho_cc forward for MPMICE
    // carry sp_vol_CC forward for ICE:computeEquilibrationPressure
    for (int m = 0; m < numMatls; m++)   {
      rho_CC_new[m].copyData(rho_CC[m]);
      sp_vol_new[m].copyData(sp_vol_CC[m]);
    }
        
   //---- P R I N T   D A T A ------   
    if (switchDebug_EQ_RF_press) {
      ostringstream desc;
      desc << "BOT_computeRFPress_patch_" << patch->getID();
      printData(0, patch, 1, desc.str(), "Press_CC_RF", press_new);

     for (int m = 0; m < numMatls; m++)  {
       ICEMaterial* matl = d_sharedState->getICEMaterial( m );
       int indx = matl->getDWIndex(); 
       ostringstream desc;
       desc << "BOT_computeRFPress_Mat_" << indx << "_patch_"<< patch->getID();
       printData( indx, patch, 1, desc.str(), "matl_press",   matl_press[m]);
       printData( indx, patch, 1, desc.str(), "rho_CC",       rho_CC[m]);
       printData( indx, patch, 1, desc.str(), "sp_vol_CC",    sp_vol_new[m]);
       printData( indx, patch, 1, desc.str(), "rho_micro_CC", rho_micro[m]);
       printData( indx, patch, 1, desc.str(), "vol_frac_CC",  vol_frac[m]);
     }
    }
  } // patches
}

/* --------------------------------------------------------------------- 
 Function~  ICE::computeDivThetaVel_CC-- 
_____________________________________________________________________*/
void ICE::computeDivThetaVel_CC(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* ,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<<"doing computeDivThetaVel_CC on patch "
              << patch->getID() <<"\t\t\t ICE" << endl;

    Vector dx       = patch->dCell();
    Ghost::GhostType  gac = Ghost::AroundCells;
    int numMatls  = d_sharedState->getNumMatls();  
      
    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces
    
    for (int m = 0; m < numMatls; m++)   {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);

      CCVariable<Vector> D;
      constCCVariable<Vector> vel_CC;
      constCCVariable<double> vol_frac;
      new_dw->allocateAndPut(D, lb->DLabel, indx, patch);
      D.initialize(Vector(0.0, 0.0, 0.0));
      new_dw->get(vol_frac,    lb->vol_frac_CCLabel, indx, patch,gac,1);
      if(ice_matl){
        old_dw->get(vel_CC,    lb->vel_CCLabel,      indx, patch,gac,1);
      } else {
        new_dw->get(vel_CC,    lb->vel_CCLabel,      indx, patch,gac,1);
      }
      //__________________________________
      // compute D using cell-centered difference
      for (int dir=0; dir<3; dir++ ) {
        for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
          IntVector c  = *iter;
          IntVector RR = c - adj_offset[dir];
          IntVector L  = c + adj_offset[dir];
          D[c][dir] = (vol_frac[RR] * vel_CC[RR][dir] 
                    -  vol_frac[L]  * vel_CC[L][dir])/(2.*dx[dir]);
        }
      }
      
      setBC(D, "Neumann", patch, indx);
    }  // matls loop
  } // patches
} 


/* ---------------------------------------------------------------------
 Function~  ICE::vel_PressDiff_FC--
 Purpose~  compute face-centered velocity and pressure difference terms 
_____________________________________________________________________*/
template<class T> void ICE::vel_PressDiff_FC
                                      (int dir, 
                                       CellIterator it,
                                       IntVector adj_offset,double dx,
                                       double delT, double gravity,
                                       constCCVariable<double>& sp_vol_CC,
                                       constCCVariable<Vector>& vel_CC,
                                       constCCVariable<double>& vol_frac,
                                       constCCVariable<double>& rho_CC,
                                       constCCVariable<Vector>& D,
                                       constCCVariable<double>& speedSound,
                                       constCCVariable<double>& matl_press_CC,
                                       constCCVariable<double>& press_CC,
                                       T& vel_FC,
                                       T& pressDiff_FC)
{
  double term1, term2, term3, term4, sp_vol_brack, rho_brack, delP_FC;

  for(;!it.done(); it++){
    IntVector R = *it;
    IntVector L = R + adj_offset;
   
    //__________________________________
    // compute local timestep
    double divu = (vol_frac[R]*vel_CC[R][dir] - vol_frac[L]*vel_CC[L][dir])/dx;
    
    double rminus   = 2.*D[R][dir]/(divu + d_SMALL_NUM) - 1.;    // 4.16b-c
    double rplus    = 2.*D[L][dir]/(divu + d_SMALL_NUM) - 1.;    // 4.16b-c

    double rr       = min(1.0,2.0 * rplus);
    double rl       = min(1.0,2.0 * rminus);

    double phi      = 2.0 - max(0.0,rr) - max(0.0,rl);
    
    double kappa_R  = sp_vol_CC[R]/( speedSound[R] * speedSound[R] );
    double kappa_L  = sp_vol_CC[L]/( speedSound[L] * speedSound[L] );

    double tmp      = (sp_vol_CC[L] + sp_vol_CC[R])/(kappa_L + kappa_R);
    double cstar    = sqrt(tmp) + fabs(vel_CC[R].length() - 
                                       vel_CC[L].length()); 
/*`==========TESTING==========*/                                 
    phi = 0.0;  //HARDWIRED
/*==========TESTING==========`*/
    
    double local_dt = delT + .5*phi*(dx/cstar - delT);
    double dtdx     = local_dt/dx;            //4.10d

    //__________________________________
    // interpolation to the face
    term1 = (vel_CC[L][dir] * sp_vol_CC[R] +
             vel_CC[R][dir] * sp_vol_CC[L])/
             (sp_vol_CC[R] + sp_vol_CC[L]);

    //__________________________________
    // pressure term
    sp_vol_brack = 2.*(sp_vol_CC[L] * sp_vol_CC[R])/
                      (sp_vol_CC[L] + sp_vol_CC[R]);
                      
    term2 = dtdx * sp_vol_brack * (press_CC[R] - press_CC[L]);

    //__________________________________
    //  stress Difference term
    rho_brack    = (rho_CC[L] * rho_CC[R])/
                   (rho_CC[L] + rho_CC[R]);
    
    double pDiff_R   = matl_press_CC[R] - press_CC[R];
    double pDiff_L   = matl_press_CC[L] - press_CC[L];
    double stressBar = rho_brack * 
                     (sp_vol_CC[L]*pDiff_L + sp_vol_CC[R]*pDiff_R); 
    double stressBar_R = stressBar/(vol_frac[R]);
    double stressBar_L = stressBar/(vol_frac[L]); 
        
    term3 = dtdx * sp_vol_brack * 
            ( (stressBar_L - pDiff_L) + (pDiff_R - stressBar_R) );

    //__________________________________
    // gravity term
    term4 =  delT * gravity;

    vel_FC[R] = term1 - term2 - term3 + term4;
   
    //__________________________________
    // compute pressDiff  //4.13c

    delP_FC = rho_brack * phi * cstar * (vel_CC[R].length() -
                                         vel_CC[L].length() );                
              
    pressDiff_FC[R] = (stressBar - delP_FC);        
  }
} 
//______________________________________________________________________
//
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
    Vector dx      = patch->dCell();
    Vector gravity = d_sharedState->getGravity();

    constCCVariable<double> press_CC;
    constCCVariable<double> matl_press_CC;
    Ghost::GhostType  gac = Ghost::AroundCells; 
    new_dw->get(press_CC,lb->press_CCLabel, 0, patch,gac,1);

    // Compute the face centered velocities
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      constCCVariable<double> rho_CC,vol_frac, sp_vol_CC,speedSound;
      constCCVariable<Vector> vel_CC;
      constCCVariable<Vector> D;
      if(ice_matl){
        old_dw->get(vel_CC, lb->vel_CCLabel, indx, patch, gac,1);
      } else {
        new_dw->get(vel_CC, lb->vel_CCLabel, indx, patch, gac,1);
      }
      new_dw->get(D,             lb->DLabel,              indx,patch,gac,1);
      new_dw->get(rho_CC,        lb->rho_CCLabel,         indx,patch,gac,1);
      new_dw->get(sp_vol_CC,     lb->sp_vol_CCLabel,      indx,patch,gac,1);
      new_dw->get(matl_press_CC, lb->matl_press_CCLabel,  indx,patch,gac,1);
      new_dw->get(vol_frac,      lb->vol_frac_CCLabel,    indx,patch,gac,1);
      new_dw->get(speedSound,    lb->speedSound_CCLabel,  indx,patch,gac,1);

      //---- P R I N T   D A T A ------
      if (switchDebug_vel_FC ) {
        ostringstream desc;
        desc << "TOP_vel_FC_Mat_" << indx << "_patch_" << patch->getID();
        printData(  indx, patch,1, desc.str(), "rho_CC",     rho_CC);     
        printData(  indx, patch,1, desc.str(), "sp_vol_CC",  sp_vol_CC);  
        printVector(indx, patch,1, desc.str(), "vel_CC",  0, vel_CC);     
        printVector(indx, patch,1, desc.str(), "D",       0, D);          
      }

      SFCXVariable<double> uvel_FC, pressDiffX_FC;
      SFCYVariable<double> vvel_FC, pressDiffY_FC;
      SFCZVariable<double> wvel_FC, pressDiffZ_FC;
      new_dw->allocateAndPut(uvel_FC,      lb->uvel_FCLabel,       indx,patch); 
      new_dw->allocateAndPut(vvel_FC,      lb->vvel_FCLabel,       indx,patch); 
      new_dw->allocateAndPut(wvel_FC,      lb->wvel_FCLabel,       indx,patch); 
      new_dw->allocateAndPut(pressDiffX_FC,lb->press_diffX_FCLabel,indx,patch); 
      new_dw->allocateAndPut(pressDiffY_FC,lb->press_diffY_FCLabel,indx,patch); 
      new_dw->allocateAndPut(pressDiffZ_FC,lb->press_diffZ_FCLabel,indx,patch); 

      uvel_FC.initialize(0.0);     
      vvel_FC.initialize(0.0);     
      wvel_FC.initialize(0.0); 
      
      pressDiffX_FC.initialize(0.0);
      pressDiffY_FC.initialize(0.0);
      pressDiffZ_FC.initialize(0.0);
      
      vector<IntVector> adj_offset(3);
      adj_offset[0] = IntVector(-1, 0, 0);    // X faces
      adj_offset[1] = IntVector(0, -1, 0);    // Y faces
      adj_offset[2] = IntVector(0,  0, -1);   // Z faces  
      int offset=0; // 0=Compute all faces in computational domain
                    // 1=Skip the faces at the border between interior and gc      
      //__________________________________
      //  Compute vel_FC for each face
      vel_PressDiff_FC<SFCXVariable<double> >(
                                       0,patch->getSFCXIterator(offset),
                                       adj_offset[0], dx[0], delT, gravity[0],
                                       sp_vol_CC, vel_CC, vol_frac, rho_CC, D,
                                       speedSound, matl_press_CC, press_CC,
                                       uvel_FC, pressDiffX_FC);

      vel_PressDiff_FC<SFCYVariable<double> >(
                                       1,patch->getSFCYIterator(offset),
                                       adj_offset[1], dx[1], delT, gravity[1],
                                       sp_vol_CC, vel_CC, vol_frac, rho_CC, D,
                                       speedSound, matl_press_CC, press_CC,
                                       vvel_FC, pressDiffY_FC);

      vel_PressDiff_FC<SFCZVariable<double> >(
                                       2,patch->getSFCZIterator(offset),
                                       adj_offset[2], dx[2], delT, gravity[2],
                                       sp_vol_CC, vel_CC, vol_frac, rho_CC, D,
                                       speedSound, matl_press_CC, press_CC,
                                       wvel_FC, pressDiffZ_FC);

      //__________________________________
      // (*)vel_FC BC are updated in 
      // ICE::addExchangeContributionToFCVel()
      
      //---- P R I N T   D A T A ------
      if (switchDebug_vel_FC ) {
        ostringstream desc;
        desc << "BOT_vel_FC_Mat_"<< indx <<"_patch_"<< patch->getID();
        printData_FC( indx, patch,1, desc.str(), "uvel_FC",       uvel_FC);
        printData_FC( indx, patch,1, desc.str(), "vvel_FC",       vvel_FC);
        printData_FC( indx, patch,1, desc.str(), "wvel_FC",       wvel_FC);
        printData_FC( indx, patch,1, desc.str(), "pressDiffX_FC", pressDiffX_FC);
        printData_FC( indx, patch,1, desc.str(), "pressDiffY_FC", pressDiffY_FC);
        printData_FC( indx, patch,1, desc.str(), "pressDiffZ_FC", pressDiffZ_FC); 
      }
    } // matls loop
  }  // patch loop
} 

/* --------------------------------------------------------------------- 
 Function~  ICE::accumulateEnergySourceSinks_RF--
 Purpose~   This function accumulates all of the sources/sinks of energy 
 ---------------------------------------------------------------------  */
void ICE::accumulateEnergySourceSinks_RF(const ProcessorGroup*,  
                                  const PatchSubset* patches,
                                  const MaterialSubset* /*matls*/,
                                  DataWarehouse* old_dw, 
                                  DataWarehouse* new_dw)
{

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing accumulate_energy_source_sinks_RF on patch " 
         << patch->getID() << "\t ICE" << endl;

    int numMatls = d_sharedState->getNumMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx = patch->dCell();
    double areaX = dx.y() * dx.z();
    double areaY = dx.x() * dx.z();
    double areaZ = dx.x() * dx.y();
    IntVector right, left, top, bottom, front, back;
    CCVariable<double> term1, term2, term3;
    
    constCCVariable<double> sp_vol_CC, speedSound, press_CC;
    constCCVariable<double> f_theta, rho_CC, Temp_CC;
    
    constSFCXVariable<double> pressX_FC, pressDiffX_FC;
    constSFCYVariable<double> pressY_FC, pressDiffY_FC;
    constSFCZVariable<double> pressZ_FC, pressDiffZ_FC;
    
    StaticArray<constCCVariable<double>   > vol_frac(numMatls);            
    StaticArray<constCCVariable<Vector>   > vel_CC(numMatls);
    StaticArray<constSFCXVariable<double> > uvel_FC(numMatls);
    StaticArray<constSFCYVariable<double> > vvel_FC(numMatls);
    StaticArray<constSFCZVariable<double> > wvel_FC(numMatls);

    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(press_CC,     lb->press_CCLabel,  0, patch, gn,  0);
    new_dw->get(pressX_FC,    lb->pressX_FCLabel, 0, patch, gac, 1);
    new_dw->get(pressY_FC,    lb->pressY_FCLabel, 0, patch, gac, 1);
    new_dw->get(pressZ_FC,    lb->pressZ_FCLabel, 0, patch, gac, 1);
    
    new_dw->allocateTemporary(term1,  patch);
    new_dw->allocateTemporary(term2,  patch);
    new_dw->allocateTemporary(term3,  patch);
    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx    = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      new_dw->get(vol_frac[m],lb->vol_frac_CCLabel,  indx, patch, gac, 1);
      new_dw->get(uvel_FC[m], lb->uvel_FCMELabel,    indx, patch, gac, 1);
      new_dw->get(vvel_FC[m], lb->vvel_FCMELabel,    indx, patch, gac, 1);
      new_dw->get(wvel_FC[m], lb->wvel_FCMELabel,    indx, patch, gac, 1);
      if(ice_matl){
        old_dw->get(vel_CC[m],lb->vel_CCLabel,       indx, patch, gn,0);   
      } else {
        new_dw->get(vel_CC[m],lb->vel_CCLabel,       indx, patch, gn,0);   
      }
    }

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl); 
      
      int indx    = matl->getDWIndex();   
      CCVariable<double> int_eng_source;
      new_dw->get(rho_CC,        lb->rho_CCLabel,        indx,patch,gac, 1);
      new_dw->get(sp_vol_CC,     lb->sp_vol_CCLabel,     indx,patch,gac, 1);
      new_dw->get(speedSound,    lb->speedSound_CCLabel, indx,patch,gn,  0);
      new_dw->get(f_theta,       lb->f_theta_CCLabel,    indx,patch,gn,  0);
      new_dw->get(pressDiffX_FC, lb->press_diffX_FCLabel,indx,patch,gac, 1);      
      new_dw->get(pressDiffY_FC, lb->press_diffY_FCLabel,indx,patch,gac, 1);      
      new_dw->get(pressDiffZ_FC, lb->press_diffZ_FCLabel,indx,patch,gac, 1);         
      new_dw->allocateAndPut(int_eng_source, 
                                 lb->int_eng_source_CCLabel, indx,patch);
      int_eng_source.initialize(0.0);
      
      for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        double vol_frac_CC =  vol_frac[m][c];
        right    = c + IntVector(1,0,0);    left     = c;
        top      = c + IntVector(0,1,0);    bottom   = c;
        front    = c + IntVector(0,0,1);    back     = c;
             
        //__________________________________
        //  term1
        double term1_X, term1_Y, term1_Z;
        double tmp_R = 0.0, tmp_L   = 0.0;
        double tmp_T = 0.0, tmp_BOT = 0.0;
        double tmp_F = 0.0, tmp_BK  = 0.0;
      
        for(int mat = 0; mat < numMatls; mat++) {
          //   O H   T H I S   I S   G O I N G   T O   B E   S L O W   
          //__________________________________
          //  use the upwinded vol_frac
          IntVector upwc;    
          upwc     = upwindCell_X(c,       uvel_FC[mat][right],  1.0);
          tmp_R   += vol_frac[mat][upwc] * uvel_FC[mat][right];
          upwc     = upwindCell_X(c,       uvel_FC[mat][left],   0.0);
          tmp_L   += vol_frac[mat][upwc] * uvel_FC[mat][left];
          
          upwc     = upwindCell_Y(c,       vvel_FC[mat][top],    1.0);
          tmp_T   += vol_frac[mat][upwc] * vvel_FC[mat][top];
          upwc     = upwindCell_Y(c,       vvel_FC[mat][bottom], 0.0);
          tmp_BOT += vol_frac[mat][upwc] * vvel_FC[mat][bottom];
          
          upwc     = upwindCell_Z(c,       wvel_FC[mat][front],  1.0);
          tmp_F   += vol_frac[mat][upwc] * wvel_FC[mat][front];
          upwc     = upwindCell_Z(c,       wvel_FC[mat][back],   0.0);
          tmp_BK  += vol_frac[mat][upwc] * wvel_FC[mat][back];
        }


        term1_X =(tmp_R * pressX_FC[right] - tmp_L  * pressX_FC[left])  *areaX;   
        term1_Y =(tmp_T * pressY_FC[top]   - tmp_BOT* pressY_FC[bottom])*areaY;     
        term1_Z =(tmp_F * pressZ_FC[front] - tmp_BK * pressZ_FC[back])  *areaZ;
        term1[c] = f_theta[c] * (term1_X + term1_Y + term1_Z);
        //__________________________________
        // Gradient of press_FC term
        double term2_X, term2_Y, term2_Z;
        Vector U = Vector(0,0,0);
        for (int dir = 0; dir <3; dir++) {  //loop over all three directons
          for(int mat = 0; mat < numMatls; mat++) {
            U[dir] = vol_frac_CC * vel_CC[mat][c][dir];
          }
        }

        term2_X =  ( f_theta[c] * U.x() - vol_frac_CC * vel_CC[m][c].x() ) * 
                 (pressX_FC[right] - pressX_FC[left])  * areaX;
                
        term2_Y = ( f_theta[c] * U.y() - vol_frac_CC * vel_CC[m][c].y() ) * 
                 (pressY_FC[top]   - pressY_FC[bottom])* areaY;
                 
        term2_Z = ( f_theta[c] * U.z() - vol_frac_CC * vel_CC[m][c].z() ) * 
                 (pressZ_FC[front] - pressZ_FC[back])  * areaZ; 

        term2[c] = term2_X + term2_Y + term2_Z;

        
        //__________________________________
        //  Divergence of work flux
        double term3_X, term3_Y, term3_Z;
       
        term3_X = (uvel_FC[m][right]  * pressDiffX_FC[right] - 
                 uvel_FC[m][left]   * pressDiffX_FC[left] )   * areaX;
                  
        term3_Y = (vvel_FC[m][top]    * pressDiffY_FC[top]   - 
                 vvel_FC[m][bottom] * pressDiffY_FC[bottom] ) * areaY;
                   
        term3_Z = (wvel_FC[m][front]  * pressDiffZ_FC[front] - 
                 wvel_FC[m][back]   * pressDiffZ_FC[back])    * areaZ; 
                 
        term3[c] = term3_X + term3_Y + term3_Z;
  
        int_eng_source[c] = (-term1[c] + term2[c] - term3[c]) * delT;
      }  // iter loop

      //__________________________________
      //  Source due to conduction
      //   I C E   O N L Y
      if(ice_matl){
        double thermalCond = ice_matl->getThermalConductivity();
        if(thermalCond != 0.0){ 
          old_dw->get(Temp_CC, lb->temp_CCLabel, indx,patch,gac,1);
          
          SFCXVariable<double> q_X_FC;
          SFCYVariable<double> q_Y_FC;
          SFCZVariable<double> q_Z_FC;
          
          computeQ_conduction_FC( new_dw, patch, 
                                  rho_CC,  sp_vol_CC, Temp_CC, thermalCond,
                                  q_X_FC, q_Y_FC, q_Z_FC);
          
          for(CellIterator iter = patch->getCellIterator(); !iter.done(); 
                                                                    iter++){
            IntVector c = *iter;
            right  = c + IntVector(1,0,0);    left   = c ;    
            top    = c + IntVector(0,1,0);    bottom = c ;    
            front  = c + IntVector(0,0,1);    back   = c ; 

            int_eng_source[c]-=((q_X_FC[right] - q_X_FC[left])  * areaX +         
                                (q_Y_FC[top]   - q_Y_FC[bottom])* areaY +         
                                (q_Z_FC[front] - q_Z_FC[back])  * areaZ)*delT;  
          }
        } 
      }
      
      //__________________________________
      //  User specified source/sink   
      double Time= dataArchiver->getCurrentTime();  
      if (  d_add_heat &&
            Time >= d_add_heat_t_start && 
            Time <= d_add_heat_t_final ) { 
        
        double vol = dx.x() * dx.y() * dx.z();
        for (int i = 0; i<(int) d_add_heat_matls.size(); i++) {
          if(m == d_add_heat_matls[i] ){
            for(CellIterator iter = patch->getCellIterator();!iter.done();
                                                                    iter++){            
              IntVector c = *iter;
              if ( vol_frac[m][c] > 0.001) {
                int_eng_source[c] += d_add_heat_coeff[i]
                                   * delT * rho_CC[c] * vol;
              }
            }  // iter loop
          }  // if right matl
        } 
      }  // if add heat

      //---- P R I N T   D A T A ------ 
      if (switchDebugSource_Sink) {
        ostringstream desc;
        desc <<  "sources_sinks_Mat_" << indx << "_patch_"<<  patch->getID();
        printData(indx, patch,1,desc.str(),"int_eng_source_RF", int_eng_source);
#if 0
        printData(indx, patch,1,desc.str(),"term1", term1);
        printData(indx, patch,1,desc.str(),"term2", term2);
        printData(indx, patch,1,desc.str(),"term3", term3);

        if (m == 0 ){
          printData_FC( indx, patch,1, desc.str(), "pressX_FC",     pressX_FC);
          printData_FC( indx, patch,1, desc.str(), "pressY_FC",     pressY_FC);
          printData_FC( indx, patch,1, desc.str(), "pressZ_FC",     pressZ_FC);
          printData_FC( indx, patch,1, desc.str(), "pressDiffX_FC", pressDiffX_FC);
          printData_FC( indx, patch,1, desc.str(), "pressDiffY_FC", pressDiffY_FC);
          printData_FC( indx, patch,1, desc.str(), "pressDiffZ_FC", pressDiffZ_FC);
        }
        printData_FC( indx, patch,1, desc.str(), "uvel_FC",       uvel_FC[m]);
        printData_FC( indx, patch,1, desc.str(), "vvel_FC",       vvel_FC[m]);
        printData_FC( indx, patch,1, desc.str(), "wvel_FC",       wvel_FC[m]);
#endif 
      }
    }  // matl loop
  }  // patch loop
}

/*---------------------------------------------------------------------
 Function~  ICE::addExchangeToMomentumAndEnergyRF--
   This task adds the  exchange contribution to the 
   existing cell-centered momentum and internal energy
            
                   (A)                              (X)
| (1+b12 + b13)     -b12          -b23          |   |del_data_CC[1]  |    
|                                               |   |                |    
| -b21              (1+b21 + b23) -b32          |   |del_data_CC[2]  |    
|                                               |   |                | 
| -b31              -b32          (1+b31 + b32) |   |del_data_CC[2]  |

                        =
                        
                        (B)
| b12( data_CC[2] - data_CC[1] ) + b13 ( data_CC[3] -data_CC[1])    | 
|                                                                   |
| b21( data_CC[1] - data_CC[2] ) + b23 ( data_CC[3] -data_CC[2])    | 
|                                                                   |
| b31( data_CC[1] - data_CC[3] ) + b32 ( data_CC[2] -data_CC[3])    | 

 Steps for each cell;
    1) Compute the beta coefficients
    2) Form and A matrix and B vector
    3) Solve for X[*]
    4) Add X[*] to the appropriate Lagrangian data
 - apply Boundary conditions to vel_CC and Temp_CC
 ---------------------------------------------------------------------  */
void ICE::addExchangeToMomentumAndEnergyRF(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing doCCMomExchange on patch "<< patch->getID()
               <<"\t\t\t ICE" << endl;

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numALLMatls = numMPMMatls + numICEMatls;
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    // Create arrays for the grid data
    constCCVariable<double>  press_CC;
    StaticArray<CCVariable<double> > Temp_CC(numALLMatls);  
    StaticArray<constCCVariable<double> > vol_frac_CC(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numALLMatls);
    StaticArray<constCCVariable<Vector> > mom_L(numALLMatls);
    StaticArray<constCCVariable<double> > int_eng_L(numALLMatls);

    // Create variables for the results
    StaticArray<CCVariable<Vector> > mom_L_ME(numALLMatls);
    StaticArray<CCVariable<Vector> > vel_CC(numALLMatls);
    StaticArray<CCVariable<double> > tot_eng_L_ME(numALLMatls);
    StaticArray<CCVariable<double> > Tdot(numALLMatls);
    StaticArray<constCCVariable<double> > mass_L(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC(numALLMatls);
    StaticArray<constCCVariable<double> > old_Temp(numALLMatls);
    StaticArray<constCCVariable<Vector> > old_vel_CC(numALLMatls); 
    StaticArray<constCCVariable<double> > f_theta(numALLMatls);
    StaticArray<constCCVariable<double> > speedSound(numALLMatls);
    StaticArray<constCCVariable<double> > int_eng_source(numALLMatls);
    
    vector<double> b(numALLMatls);
    vector<double> sp_vol(numALLMatls);
    vector<double> cv(numALLMatls);
    vector<double> X(numALLMatls);
    vector<double> e_prime_v(numALLMatls);
    vector<double> if_mpm_matl_ignore(numALLMatls);
    
    double tmp, alpha,KE;
/*`==========TESTING==========*/
// I've included this term but it's turned off -Todd
    double Joule_coeff   = 0.0;         // measure of "thermal imperfection" 
/*==========TESTING==========`*/
    FastMatrix beta(numALLMatls, numALLMatls),acopy(numALLMatls, numALLMatls);
    FastMatrix K(numALLMatls, numALLMatls),H(numALLMatls, numALLMatls);
    FastMatrix a(numALLMatls, numALLMatls), a_inverse(numALLMatls, numALLMatls);
    FastMatrix phi(numALLMatls, numALLMatls);
    beta.zero();
    acopy.zero();
    K.zero();
    H.zero();
    a.zero();
 
    getExchangeCoefficients( K, H);
    Ghost::GhostType  gn = Ghost::None;
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      if_mpm_matl_ignore[m] = 1.0;
      if(mpm_matl){                 // M P M
        new_dw->get(old_vel_CC[m],   lb->vel_CCLabel,      indx, patch,gn,0); 
        new_dw->get(old_Temp[m],     lb->temp_CCLabel,     indx, patch,gn,0);
        new_dw->allocateTemporary(vel_CC[m],   patch);
        new_dw->allocateTemporary(Temp_CC[m],  patch);
        cv[m] = mpm_matl->getSpecificHeat();
        if_mpm_matl_ignore[m] = 0.0;
      }
      if(ice_matl){                 // I C E
        old_dw->get(old_vel_CC[m],  lb->vel_CCLabel,     indx, patch,gn,0); 
        old_dw->get(old_Temp[m],    lb->temp_CCLabel,    indx, patch,gn,0);
        new_dw->allocateTemporary(vel_CC[m],  patch);
        new_dw->allocateTemporary(Temp_CC[m], patch); 
        cv[m] = ice_matl->getSpecificHeat();
      }                             // A L L  M A T L S
      new_dw->get(mass_L[m],        lb->mass_L_CCLabel,    indx, patch,gn, 0); 
      new_dw->get(sp_vol_CC[m],     lb->sp_vol_CCLabel,    indx, patch,gn, 0);
      new_dw->get(mom_L[m],         lb->mom_L_CCLabel,     indx, patch,gn, 0); 
      new_dw->get(int_eng_L[m],     lb->int_eng_L_CCLabel, indx, patch,gn, 0); 
      new_dw->get(vol_frac_CC[m],   lb->vol_frac_CCLabel,  indx, patch,gn, 0);
      new_dw->get(speedSound[m],    lb->speedSound_CCLabel,indx, patch,gn, 0);
      new_dw->get(f_theta[m],       lb->f_theta_CCLabel,   indx, patch,gn, 0);
      new_dw->get(rho_CC[m],        lb->rho_CCLabel,       indx, patch,gn, 0);
        
      new_dw->allocateAndPut(Tdot[m],        lb->Tdot_CCLabel,    indx, patch);          
      new_dw->allocateAndPut(mom_L_ME[m],    lb->mom_L_ME_CCLabel,indx, patch);         
      new_dw->allocateAndPut(tot_eng_L_ME[m],lb->eng_L_ME_CCLabel,indx,patch);
      e_prime_v[m] = Joule_coeff * cv[m];
      Tdot[m].initialize(0.0);
      tot_eng_L_ME[m].initialize(0.0);
      Temp_CC[m].initialize(0.0);
    }
    new_dw->get(press_CC,           lb->press_CCLabel,     0,  patch,gn, 0);       
    
    // Convert momenta to velocities and internal energy to total energy
    // using the old_vel
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        KE = 0.5 * mass_L[m][c] * old_vel_CC[m][c].length() * old_vel_CC[m][c].length();
        tot_eng_L_ME[m][c] = int_eng_L[m][c] + KE;
        vel_CC[m][c]       = mom_L[m][c]/mass_L[m][c]; 
      }
    }
    //---- P R I N T   D A T A ------ 
    if (switchDebugMomentumExchange_CC ) {
      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc<<"TOP_addExchangeToMomentumAndEnergy_RF"<<indx<<"_patch_"
            <<patch->getID();
        printData(   indx, patch,1, desc.str(),"Temp_CC",    old_Temp[m]);     
        printData(   indx, patch,1, desc.str(),"int_eng_L",  int_eng_L[m]);   
        printData(   indx, patch,1, desc.str(),"mass_L",     mass_L[m]);      
        printVector( indx, patch,1, desc.str(),"vel_CC", 0,  vel_CC[m]);            
      }
    }
    //---------- M O M E N T U M   E X C H A N G E                  
    //   Form BETA matrix (a), off diagonal terms                   
    //   beta and (a) matrix are common to all momentum exchanges   
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;

      for(int m = 0; m < numALLMatls; m++)  {
        tmp = sp_vol_CC[m][c];
        for(int n = 0; n < numALLMatls; n++) {
          beta(m,n) = delT * vol_frac_CC[n][c]  * K(n,m) * tmp;
          a(m,n)    = -beta(m,n);
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numALLMatls; m++) {
        a(m,m) = 1.0;
        for(int n = 0; n < numALLMatls; n++) {
          a(m,m) +=  beta(m,n);
        }
      }
      a_inverse.destructiveInvert(a);
      
      for (int dir = 0; dir <3; dir++) {  //loop over all three directons
        for(int m = 0; m < numALLMatls; m++) {
          b[m] = 0.0;
          for(int n = 0; n < numALLMatls; n++) {
           b[m] += beta(m,n) * (vel_CC[n][c][dir] - vel_CC[m][c][dir]);
          }
        }
        
        a_inverse.multiply(b,X);
        
        for(int m = 0; m < numALLMatls; m++) {
          vel_CC[m][c][dir] =  vel_CC[m][c][dir] + X[m];
        }
      }
    }   // cell iterator

    //----------  E N E R G Y   E X C H A N G E 
    //  For mpm matls ignore thermal expansion term alpha
    //  compute phi and alpha
    // Convert total energy to Temp.  Use the vel after exchange
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;                                                        
      for (int m = 0; m < numALLMatls; m++) {                                     
        KE = 0.5 * mass_L[m][c] * vel_CC[m][c].length() * vel_CC[m][c].length();         
        Temp_CC[m][c] = (tot_eng_L_ME[m][c] - KE) /(mass_L[m][c]*cv[m]);          
      }                                                                           
  
      for(int m = 0; m < numALLMatls; m++) {
        for(int n = 0; n < numALLMatls; n++)  {
          beta(m,n)  = delT * vol_frac_CC[m][c] * vol_frac_CC[n][c] * H(n,m)/
                        (rho_CC[m][c] * cell_vol * cv[m]);
          alpha      = if_mpm_matl_ignore[n] * 1.0/Temp_CC[n][c];  
          phi(m,n)   = (f_theta[m][c] * vol_frac_CC[n][c]* alpha)/ 
                       (rho_CC[m][c]  * cell_vol * cv[m]);         
        }
      } 

      //  off diagonal terms, matrix A    
      for(int m = 0; m < numALLMatls; m++) {
        for(int n = 0; n < numALLMatls; n++)  {
          a(m,n) = -press_CC[c] * phi(m,n) - beta(m,n);
        }
      }
      //  diagonal terms, matrix A wipe out the above
      for(int m = 0; m < numALLMatls; m++) {
        a(m,m) += 1.0 + (press_CC[c] * phi(m,m))/f_theta[m][c];
        for(int n = 0; n < numALLMatls; n++) {  // sum beta 
          a(m,m) += beta(m,n); 
        }
      } 

      // -  F O R M   R H S   (b)         
      for(int m = 0; m < numALLMatls; m++)  {   
        b[m] = Temp_CC[m][c] + 
               (press_CC[c] * phi(m,m)/f_theta[m][c]) * old_Temp[m][c];
        for(int n = 0; n < numALLMatls; n++) {
          b[m] -= press_CC[c] * phi(m,n) * old_Temp[n][c];
        }
      }
      //     S O L V E  and backout Temp_CC 
      a.destructiveSolve(b,X);

      for(int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] = X[m];
      }
    }  //CellIterator loop 

    //__________________________________
    //  Set the Boundary conditions 
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int dwindex = matl->getDWIndex();
      setBC(vel_CC[m], "Velocity",   patch,dwindex);
      setBC(Temp_CC[m],"Temperature",patch, d_sharedState, dwindex);
    }
    //__________________________________
    // Convert vars. primitive-> flux 
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        mom_L_ME[m][c]     = vel_CC[m][c]  * mass_L[m][c];
        Tdot[m][c]         = (Temp_CC[m][c] - old_Temp[m][c])/delT; 
      }
    }
    
    //__________________________________
    //  Add on the KE
    // For total energy conservation use old_vel_CC
    // for a single matl or vel_CC_ME for multiple matls. -bak. 
    if (numALLMatls == 1 ){
      for (int m = 0; m < numALLMatls; m++) { 
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          KE = 0.5 * mass_L[m][c] * old_vel_CC[m][c].length() * old_vel_CC[m][c].length();
          tot_eng_L_ME[m][c] = Temp_CC[m][c] * cv[m] * mass_L[m][c] + KE;
        }
      }    
    } else {
      for (int m = 0; m < numALLMatls; m++) {
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          KE = 0.5 * mass_L[m][c] * vel_CC[m][c].length() * vel_CC[m][c].length(); 
          tot_eng_L_ME[m][c] = Temp_CC[m][c] * cv[m] * mass_L[m][c] + KE;
        }
      }    
    }

    //---- P R I N T   D A T A ------ 
    if (switchDebugMomentumExchange_CC ) {
      for(int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc<<"addExchangeToMomentumAndEnergy_RF"<<indx<<"_patch_"
            <<patch->getID();
        printVector(indx, patch,1, desc.str(),"mom_L_ME", 0, mom_L_ME[m]);
        printData(  indx, patch,1, desc.str(),"tot_eng_L_ME",tot_eng_L_ME[m]);
        printData(  indx, patch,1, desc.str(),"Tdot",        Tdot[m]);
        printData(  indx, patch,1, desc.str(),"Temp_CC",     Temp_CC[m]); 
      }
    }
  } //patches
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
    double areaX = dx.y() * dx.z();
    double areaY = dx.x() * dx.z();
    double areaZ = dx.x() * dx.y(); 
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    StaticArray<constSFCXVariable<double> > uvel_FC(numALLMatls);
    StaticArray<constSFCYVariable<double> > vvel_FC(numALLMatls);
    StaticArray<constSFCZVariable<double> > wvel_FC(numALLMatls); 
    
    StaticArray<constCCVariable<double> > Tdot(numALLMatls);
    StaticArray<constCCVariable<double> > vol_frac(numALLMatls);
    StaticArray<constCCVariable<double> > Temp_CC(numALLMatls);
    constCCVariable<double> rho_CC, rho_micro, f_theta,sp_vol_CC;
    constCCVariable<double> sp_vol_comb;
    CCVariable<double> sum_therm_exp;
    vector<double> if_mpm_matl_ignore(numALLMatls);

    new_dw->allocateTemporary(sum_therm_exp,patch);
    sum_therm_exp.initialize(0.);
 
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex();
      new_dw->get(Tdot[m],    lb->Tdot_CCLabel,    indx,patch, gn,0);  
      new_dw->get(vol_frac[m],lb->vol_frac_CCLabel,indx,patch, gac, 1);
      new_dw->get(uvel_FC[m], lb->uvel_FCMELabel,  indx, patch,gac, 1);   
      new_dw->get(vvel_FC[m], lb->vvel_FCMELabel,  indx, patch,gac, 1);   
      new_dw->get(wvel_FC[m], lb->wvel_FCMELabel,  indx, patch,gac, 1);
      if (ice_matl) {  
        old_dw->get(Temp_CC[m], lb->temp_CCLabel,    indx,patch, gn,0); 
      }
      if (mpm_matl) {
        new_dw->get(Temp_CC[m],lb->temp_CCLabel,   indx,patch, gn,0);
      }
    }
    
    //__________________________________
    // Sum of thermal expansion
    // alpha is hardwired for ideal gases
    // ignore contributions from mpm_matls
    for(int mm = 0; mm < numALLMatls; mm++) {
      Material* matl = d_sharedState->getMaterial( mm );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
 
      if_mpm_matl_ignore[mm] = 1.0; 
      if ( mpm_matl) {       
        if_mpm_matl_ignore[mm] = 0.0; 
      } 
      for(CellIterator iter=patch->getExtraCellIterator();
                                                        !iter.done();iter++){
        IntVector c = *iter;
        double alpha =  if_mpm_matl_ignore[mm] * 1.0/Temp_CC[mm][c];
        sum_therm_exp[c] += vol_frac[mm][c]*alpha*Tdot[mm][c];
      }  
    }

    //__________________________________ 
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      CCVariable<double> spec_vol_L, spec_vol_source;
      new_dw->allocateAndPut(spec_vol_L,     lb->spec_vol_L_CCLabel,     
                                                            indx,patch);
      new_dw->allocateAndPut(spec_vol_source,lb->spec_vol_source_CCLabel,
                                                            indx,patch);
      spec_vol_source.initialize(0.);
      
      new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn, 0); 
      new_dw->get(rho_CC,     lb->rho_CCLabel,        indx,patch,gn, 0); 
      new_dw->get(f_theta,    lb->f_theta_CCLabel,    indx,patch,gn, 0); 
      new_dw->get(sp_vol_comb,lb->created_vol_CCLabel,indx,patch,gn, 0); 

      //__________________________________
      //  compute spec_vol_L * mass
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        spec_vol_L[c] = (rho_CC[c] * vol)*sp_vol_CC[c];
      }
      //  Set Neumann = 0 if symmetric Boundary conditions
      setBC(spec_vol_L, "set_if_sym_BC",patch, d_sharedState, indx);
      
      //__________________________________
      //  add the sources to spec_vol_L
      for(CellIterator iter=patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        
        IntVector right, left, top, bottom, front, back;
        right    = c + IntVector(1,0,0);    left     = c;
        top      = c + IntVector(0,1,0);    bottom   = c;
        front    = c + IntVector(0,0,1);    back     = c;
        //__________________________________
        //  term1
        double term1, term1_X, term1_Y, term1_Z;
        double tmp_R = 0.0, tmp_L   = 0.0;
        double tmp_T = 0.0, tmp_BOT = 0.0;
        double tmp_F = 0.0, tmp_BK  = 0.0;
        //   O H   T H I S   I S   G O I N G   T O   B E   S L O W  
        for(int mm = 0; mm < numALLMatls; mm++) {
          //  use the upwinded vol_frac
          IntVector upwc;     
          upwc     = upwindCell_X(c,     uvel_FC[mm][right],  1.0);
          tmp_R   += vol_frac[mm][upwc] * uvel_FC[mm][right];
          upwc     = upwindCell_X(c,     uvel_FC[mm][left],   0.0);
          tmp_L   += vol_frac[mm][upwc] * uvel_FC[mm][left];
          
          upwc     = upwindCell_Y(c,     vvel_FC[mm][top],    1.0);
          tmp_T   += vol_frac[mm][upwc] * vvel_FC[mm][top];
          upwc     = upwindCell_Y(c,     vvel_FC[mm][bottom], 0.0);
          tmp_BOT += vol_frac[mm][upwc] * vvel_FC[mm][bottom];
          
          upwc     = upwindCell_Z(c,     wvel_FC[mm][front],  1.0);
          tmp_F   += vol_frac[mm][upwc] * wvel_FC[mm][front];
          upwc     = upwindCell_Z(c,     wvel_FC[mm][back],   0.0);
          tmp_BK  += vol_frac[mm][upwc] * wvel_FC[mm][back];
        } 
        term1_X = (tmp_R - tmp_L)   * areaX;
        term1_Y = (tmp_T - tmp_BOT) * areaY;
        term1_Z = (tmp_F - tmp_BK)  * areaZ;

        term1 = f_theta[c] * delT * (term1_X + term1_Y + term1_Z);
       

        double alpha = 1.0/Temp_CC[m][c];  // HARDWRIED FOR IDEAL GAS
        double term2 = delT * vol * (vol_frac[m][c] * alpha *  Tdot[m][c] -
                                   f_theta[c] * sum_therm_exp[c]);
                                   
        // This is actually mass * sp_vol
        double src = term1 + if_mpm_matl_ignore[m] * term2;
        spec_vol_L[c]     += src + sp_vol_comb[c]; 
        spec_vol_source[c] = src/(rho_CC[c] * vol);
        
/*`==========TESTING==========*/
//    do we really want this?  -Todd        
        spec_vol_L[c] = max(spec_vol_L[c], d_TINY_RHO * vol * sp_vol_CC[c]);
/*==========TESTING==========`*/ 
     }

      //  Set Neumann = 0 if symmetric Boundary conditions
      setBC(spec_vol_L, "set_if_sym_BC",patch, d_sharedState, indx); 

      //---- P R I N T   D A T A ------ 
      if (switchDebugLagrangianSpecificVol ) {
        ostringstream desc;
        desc <<"BOT_Lagrangian_spVolRF_Mat_"<<indx<< "_patch_"<<patch->getID();
        printData(indx, patch,1, desc.str(), "Temp",          Temp_CC[m]); 
        printData(indx, patch,1, desc.str(), "vol_frac",      vol_frac[m]);
        printData(indx, patch,1, desc.str(), "rho_CC",        rho_CC);
        printData(indx, patch,1, desc.str(), "sp_vol_CC",     sp_vol_CC);
        printData(indx, patch,1, desc.str(), "Tdot",          Tdot[m]);
        printData(indx, patch,1, desc.str(), "f_theta",       f_theta);
        printData(indx, patch,1, desc.str(), "sum_therm_exp", sum_therm_exp);
        printData(indx, patch,1, desc.str(), "spec_vol_source",spec_vol_source);
        printData(indx, patch,1, desc.str(), "spec_vol_L",     spec_vol_L);
      }
      //____ B U L L E T   P R O O F I N G----
      IntVector neg_cell;
      if (!areAllValuesPositive(spec_vol_L, neg_cell)) {
        cout << "matl            "<< indx << endl;
        cout << "sum_thermal_exp "<< sum_therm_exp[neg_cell] << endl;
        cout << "spec_vol_source "<< spec_vol_source[neg_cell] << endl;
        cout << "sp_vol_comb     "<< sp_vol_comb[neg_cell] << endl;
        cout << "mass sp_vol_L    "<< spec_vol_L[neg_cell] << endl;
        cout << "mass sp_vol_L_old"
             << (rho_CC[neg_cell]*vol*sp_vol_CC[neg_cell]) << endl;
        ostringstream warn;
        warn<<"ERROR ICE::computeLagrangianSpecificVolumeRF, mat "<<indx
            << " cell " <<neg_cell << " spec_vol_L is negative\n";
        throw InvalidValue(warn.str());
     } 
    }  // end numALLMatl loop
  }  // patch loop
}
