//
// $Id$
//
#include <Uintah/Components/MPMICE/MPMICE.h>
#include <Uintah/Components/MPMICE/MPMICELabel.h>

#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Components/MPM/Burn/HEBurn.h>
#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Uintah/Components/MPM/MPMPhysicalModules.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Components/ICE/ICEMaterial.h>

#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/Scheduler.h>
#include <SCICore/Datatypes/DenseMatrix.h>

using namespace Uintah;
using namespace Uintah::MPM;
using namespace Uintah::ICESpace;
using namespace Uintah::MPMICESpace;

using SCICore::Datatypes::DenseMatrix;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Dot;
using SCICore::Math::Min;
using SCICore::Math::Max;
using namespace std;

MPMICE::MPMICE(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  d_fracture = false;
  d_mpm      = scinew SerialMPM(myworld);
  d_ice      = scinew ICE(myworld);
}

MPMICE::~MPMICE()
{
  delete Mlb;
  delete Ilb;
  delete MIlb;
  delete d_mpm;
  delete d_ice;
}

void MPMICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
			  SimulationStateP& sharedState)
{
   d_sharedState = sharedState;

   d_mpm->setMPMLabel(Mlb);
   d_mpm->problemSetup(prob_spec, grid, d_sharedState);

   d_ice->setICELabel(Ilb);
   d_ice->problemSetup(prob_spec, grid, d_sharedState);

   cerr << "MPMICE::problemSetup passed.\n";
}

void MPMICE::scheduleInitialize(const LevelP& level,
				SchedulerP& sched,
				DataWarehouseP& dw)
{
  d_mpm->scheduleInitialize(level, sched, dw);
  d_ice->scheduleInitialize(level, sched, dw);
}

void MPMICE::scheduleComputeStableTimestep(const LevelP& level,
					   SchedulerP& sched,
					   DataWarehouseP& dw)
{
  // Schedule computing the ICE stable timestep
  d_ice->scheduleComputeStableTimestep(level, sched, dw);
  // MPM stable timestep is a by product of the CM
}

void MPMICE::scheduleTimeAdvance(double t, double dt,
				 const LevelP&         level,
				 SchedulerP&     sched,
				 DataWarehouseP& old_dw, 
				 DataWarehouseP& new_dw)
{
   int numMPMMatls = d_sharedState->getNumMPMMatls();
   int numICEMatls = d_sharedState->getNumICEMatls();

   for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

    const Patch* patch=*iter;
    if(d_fracture) {
       d_mpm->scheduleComputeNodeVisibility(patch,sched,old_dw,new_dw);
    }
    d_mpm->scheduleInterpolateParticlesToGrid(patch,sched,old_dw,new_dw);

    if (MPMPhysicalModules::thermalContactModel) {
       d_mpm->scheduleComputeHeatExchange(patch,sched,old_dw,new_dw);
    }

    d_mpm->scheduleExMomInterpolated(patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeStressTensor(patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeInternalForce(patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeInternalHeatRate(patch,sched,old_dw,new_dw);
    d_mpm->scheduleSolveEquationsMotion(patch,sched,old_dw,new_dw);
    d_mpm->scheduleSolveHeatEquations(patch,sched,old_dw,new_dw);
    d_mpm->scheduleIntegrateAcceleration(patch,sched,old_dw,new_dw);
    d_mpm->scheduleIntegrateTemperatureRate(patch,sched,old_dw,new_dw);

    d_ice->scheduleComputeEquilibrationPressure(    patch,sched,old_dw,new_dw);
    d_ice->scheduleComputeFaceCenteredVelocities(   patch,sched,old_dw,new_dw);
    d_ice->scheduleAddExchangeContributionToFCVel(  patch,sched,old_dw,new_dw);
    d_ice->scheduleComputeDelPressAndUpdatePressCC( patch,sched,old_dw,new_dw);
    d_ice->scheduleComputePressFC(                  patch,sched,old_dw,new_dw);
    d_ice->scheduleAccumulateMomentumSourceSinks(   patch,sched,old_dw,new_dw);
    d_ice->scheduleAccumulateEnergySourceSinks(     patch,sched,old_dw,new_dw);
    d_ice->scheduleComputeLagrangianValues(         patch,sched,old_dw,new_dw);

    scheduleInterpolateNCToCC(patch,sched,old_dw,new_dw);

    // Coupled Momentum Exchange
    scheduleCCMomExchange(patch,sched,old_dw,new_dw);

    // Separate Momentum Exchange
//    d_mpm->scheduleExMomIntegrated(patch,sched,old_dw,new_dw);
//    d_ice->scheduleAddExchangeToMomentumAndEnergy(patch,sched,old_dw,new_dw);

    d_mpm->scheduleInterpolateToParticlesAndUpdate(patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeMassRate(patch,sched,old_dw,new_dw);
    if(d_fracture) {
      d_mpm->scheduleCrackGrow(patch,sched,old_dw,new_dw);
      d_mpm->scheduleStressRelease(patch,sched,old_dw,new_dw);
      d_mpm->scheduleComputeCrackSurfaceContactForce(patch,sched,old_dw,new_dw);
    }

    d_mpm->scheduleCarryForwardVariables(patch,sched,old_dw,new_dw);

    // Step 6and7 advect and advance in time
    d_ice->scheduleAdvectAndAdvanceInTime(
            patch,sched,old_dw,new_dw);

#if 0
      {
	/* interpolateCCToNC */

	 Task* t=scinew Task("MPMICE::interpolateCCToNC",
		    patch, old_dw, new_dw,
		    this, &MPMICE::interpolateCCToNC);

	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    t->requires(new_dw, Mlb->cVelocityLabel,   idx, patch,
			Ghost::None, 0);
	    t->requires(new_dw, Mlb->cVelocityMELabel, idx, patch,
			Ghost::None, 0);
	    t->computes(new_dw, Mlb->gVelAfterIceLabel, idx, patch);
	    t->computes(new_dw, Mlb->gAccAfterIceLabel, idx, patch);
	 }
	 t->requires(old_dw, d_sharedState->get_delt_label() );

	sched->addTask(t);
      }
#endif
  }

    
   sched->scheduleParticleRelocation(level, old_dw, new_dw,
				     Mlb->pXLabel_preReloc, 
				     Mlb->d_particleState_preReloc,
				     Mlb->pXLabel, Mlb->d_particleState,
				     numMPMMatls);
}

void MPMICE::scheduleInterpolateNCToCC(const Patch* patch,
                                       SchedulerP& sched,
                                       DataWarehouseP& old_dw,
                                       DataWarehouseP& new_dw)
{
   /* interpolateNCToCC */

   int numMPMMatls = d_sharedState->getNumMPMMatls();
   Task* t=scinew Task("MPMICE::interpolateNCToCC",
		        patch, old_dw, new_dw,
		        this, &MPMICE::interpolateNCToCC);

   for(int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     int idx = mpm_matl->getDWIndex();
     t->requires(new_dw, Mlb->gVelocityStarLabel, idx, patch,
		Ghost::AroundCells, 1);
     t->requires(new_dw, Mlb->gMassLabel,                idx, patch,
		Ghost::AroundCells, 1);

     t->computes(new_dw, MIlb->cMomentumLabel, idx, patch);
     t->computes(new_dw, MIlb->cMassLabel,     idx, patch);
   }

   sched->addTask(t);

}

void MPMICE::scheduleCCMomExchange(const Patch* patch,
                                   SchedulerP& sched,
                                   DataWarehouseP& old_dw,
                                   DataWarehouseP& new_dw)
{
   Task* t=scinew Task("MPMICE::doCCMomExchange",
		        patch, old_dw, new_dw,
		        this, &MPMICE::doCCMomExchange);

   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     int mpmidx = mpm_matl->getDWIndex();
     t->requires(new_dw, MIlb->cMomentumLabel, mpmidx, patch, Ghost::None, 0);
     t->requires(new_dw, MIlb->cMassLabel,     mpmidx, patch, Ghost::None, 0);
     t->requires(new_dw, Mlb->gVelocityStarLabel,
					       mpmidx, patch, Ghost::None, 0);
     t->requires(new_dw, Mlb->gAccelerationLabel,
					       mpmidx, patch, Ghost::None, 0);

     t->computes(new_dw, Mlb->gMomExedVelocityStarLabel, mpmidx, patch);
     t->computes(new_dw, Mlb->gMomExedAccelerationLabel, mpmidx, patch);
   }

  for (int m = 0; m < d_sharedState->getNumICEMatls(); m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int iceidx = matl->getDWIndex();
    t->requires(old_dw,Ilb->rho_CCLabel,             iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->mom_L_CCLabel,           iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->int_eng_L_CCLabel,       iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->vol_frac_CCLabel,        iceidx,patch,Ghost::None);
    t->requires(old_dw,Ilb->cv_CCLabel,              iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->rho_micro_CCLabel, iceidx,patch,Ghost::None);
    t->computes(new_dw,Ilb->mom_L_ME_CCLabel,    iceidx, patch);
    t->computes(new_dw,Ilb->int_eng_L_ME_CCLabel, iceidx, patch);
  }

   sched->addTask(t);

}

void MPMICE::doCCMomExchange(const ProcessorGroup*,
                             const Patch* patch,
                             DataWarehouseP& old_dw,
                             DataWarehouseP& new_dw)
{
  int numMPMMatls = d_sharedState->getNumMPMMatls();
  int numICEMatls = d_sharedState->getNumICEMatls();
  int numALLMatls = numMPMMatls + numICEMatls;

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();

  // Create arrays for the grid data
  NCVariable<Vector> gacceleration, gvelocity;  //temporary to keep running
  NCVariable<Vector> gMEacceleration, gMEvelocity;

  // Create variables for the required values
  vector<CCVariable<double> > rho_CC(numALLMatls);  // See note below
  // rho_CC will be filled with rho_CCLabel for ICEMatls
  // but will contain cMassLabel for MPMMatls

  vector<CCVariable<double> > vol_frac_CC(numALLMatls);
  vector<CCVariable<double> > rho_micro_CC(numALLMatls);
  vector<CCVariable<double> > cv_CC(numALLMatls);

  vector<CCVariable<Vector> > mom_L(numALLMatls);
  vector<CCVariable<double> > int_eng_L(numALLMatls);

  // Create variables for the results
  vector<CCVariable<Vector> > mom_L_ME(numALLMatls);
  vector<CCVariable<double> > int_eng_L_ME(numALLMatls);

  vector<double> b(numALLMatls);
  vector<double> mass(numALLMatls);
  DenseMatrix beta(numALLMatls,numALLMatls),acopy(numALLMatls,numALLMatls);
  DenseMatrix K(numALLMatls,numALLMatls),H(numALLMatls,numALLMatls);
  DenseMatrix a(numALLMatls,numALLMatls);

//  for (int i = 0; i < numALLMatls; i++ ) {
//      K[numICEMatls-1-i][i] = d_K_mom[i];
//      H[numICEMatls-1-i][i] = d_K_heat[i];
//  }

  // Hardwiring the values for the momentum exchange for now
  K[0][0] = 0.;
//  K[0][1] = 1.e12;
//  K[1][0] = 1.e12;
  K[0][1] = 0.;
  K[1][0] = 0.;
  K[1][1] = 0.;

  for(int m = 0; m < numALLMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = mpm_matl->getDWIndex();

      // This is just the temporary stuff to carry forward data until this works
      {
      new_dw->get(gvelocity,     Mlb->gVelocityStarLabel, matlindex, patch,
							Ghost::None, 0);
      new_dw->get(gacceleration, Mlb->gAccelerationLabel, matlindex, patch,
							Ghost::None, 0);
      new_dw->allocate(gMEvelocity,     Mlb->gMomExedVelocityStarLabel,
							 matlindex, patch);
      new_dw->allocate(gMEacceleration, Mlb->gMomExedAccelerationLabel,
							 matlindex, patch);
      // Just carrying forward for now
      new_dw->put(gvelocity,    Mlb->gMomExedVelocityStarLabel,matlindex,patch);
      new_dw->put(gacceleration,Mlb->gMomExedAccelerationLabel,matlindex,patch);
      }


      new_dw->get(mom_L[m],      MIlb->cMomentumLabel, matlindex, patch,
							Ghost::None, 0);
      new_dw->get(rho_CC[m],     MIlb->cMassLabel,     matlindex, patch,
							Ghost::None, 0);
    }
  }

  for(int m = 0; m < numALLMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      int dwindex = ice_matl->getDWIndex();
      old_dw->get(rho_CC[m],      Ilb->rho_CCLabel,
                                dwindex, patch, Ghost::None, 0);
      new_dw->get(mom_L[m],       Ilb->mom_L_CCLabel,
                                dwindex, patch, Ghost::None, 0);
      new_dw->get(int_eng_L[m],    Ilb->int_eng_L_CCLabel,
                                dwindex, patch, Ghost::None, 0);
      new_dw->get(vol_frac_CC[m],  Ilb->vol_frac_CCLabel,
                                dwindex, patch, Ghost::None, 0);
      new_dw->get(rho_micro_CC[m], Ilb->rho_micro_CCLabel,
                                dwindex, patch, Ghost::None, 0);
      old_dw->get(cv_CC[m],        Ilb->cv_CCLabel,
                                dwindex, patch, Ghost::None, 0);

      new_dw->allocate(mom_L_ME[m],    Ilb->mom_L_ME_CCLabel,    dwindex,patch);
      new_dw->allocate(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex,patch);
    }
  }

  for (int m = 0; m < numALLMatls; m++) {
	mom_L_ME[m] = mom_L[m];
	int_eng_L_ME[m] = int_eng_L[m];
  }

//#if 0
  double vol = dx.x()*dx.y()*dx.z();
  double temp;
  int itworked;
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    //   Form BETA matrix (a), off diagonal terms
    for(int m = 0; m < numALLMatls; m++)  {
      Material* matlm = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matlm = dynamic_cast<ICEMaterial*>(matlm);
      MPMMaterial* mpm_matlm = dynamic_cast<MPMMaterial*>(matlm);
      if(ice_matlm){
        mass[m] = rho_CC[m][*iter] * vol;
        temp    = rho_micro_CC[m][*iter];
      }
      if(mpm_matlm){
        mass[m] = rho_CC[m][*iter];
        temp    = rho_CC[m][*iter];
	temp    = 1000.0;
      }
      
      for(int n = 0; n < numALLMatls; n++) {
//	beta[m][n] = delT * vol_frac_CC[n][*iter] * K[n][m]/temp;
	beta[m][n] = delT * K[n][m]/temp;
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
    //     X - M O M E N T U M
    // -  F O R M   R H S   (b)
    //   convert flux to primiative variable
    for(int m = 0; m < numALLMatls; m++) {
      b[m] = 0.0;
      for(int n = 0; n < numALLMatls; n++) {
	b[m] += beta[m][n] *
	  (mom_L[n][*iter].x()/mass[n] - mom_L[m][*iter].x()/mass[m]);
      }
    }
    //     S O L V E
    //  - push a copy of (a) into the solver
    //  - Add exchange contribution to orig value
    acopy = a;
    itworked = acopy.solve(b);
    for(int m = 0; m < numALLMatls; m++) {
        mom_L_ME[m][*iter].x( mom_L[m][*iter].x() + b[m]*mass[m] );
    }

    //     Y - M O M E N T U M
    // -  F O R M   R H S   (b)
    //   convert flux to primiative variable
//    cout << "Y - M O M E N T U M   F O R M   R H S   (b)" << endl;
    for(int m = 0; m < numALLMatls; m++) {
      b[m] = 0.0;
      for(int n = 0; n < numALLMatls; n++) {
	b[m] += beta[m][n] *
	  (mom_L[n][*iter].y()/mass[n] - mom_L[m][*iter].y()/mass[m]);
      }
    }

    //     S O L V E
    //  - push a copy of (a) into the solver
    //  - Add exchange contribution to orig value
//    cout << "Y - M O M E N T U M   Solve " << endl;
    acopy    = a;
    itworked = acopy.solve(b);
    for(int m = 0; m < numALLMatls; m++)   {
        mom_L_ME[m][*iter].y( mom_L[m][*iter].y() + b[m]*mass[m] );
    }

    //     Z - M O M E N T U M
    // -  F O R M   R H S   (b)
    //   convert flux to primiative variable
    for(int m = 0; m < numALLMatls; m++)  {
      b[m] = 0.0;
      for(int n = 0; n < numALLMatls; n++) {
	b[m] += beta[m][n] *
	  (mom_L[n][*iter].z()/mass[n] - mom_L[m][*iter].z()/mass[m]);
      }
    }    

    //     S O L V E
    //  - push a copy of (a) into the solver
    //  - Add exchange contribution to orig value
    acopy    = a;
    itworked = acopy.solve(b);
    for(int m = 0; m < numALLMatls; m++)  {
      mom_L_ME[m][*iter].z( mom_L[m][*iter].z() + b[m]*mass[m] );
    }
  }
//#endif
  

  // put ONLY the ICE Materials' data in the new_dw
  for(int m = 0; m < numALLMatls; m++){
     Material* matl = d_sharedState->getMaterial( m );
     ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
     if(ice_matl){
       int dwindex = ice_matl->getDWIndex();
       new_dw->put(mom_L_ME[m],    Ilb->mom_L_ME_CCLabel,    dwindex, patch);
       new_dw->put(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex, patch);
     }
  }

  // This is where I'll interpolate the CC changes to NCs for the MPMMatls

}

void MPMICE::interpolateNCToCC(const ProcessorGroup*,
                               const Patch* patch,
                               DataWarehouseP&,
                               DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Vector zero(0.,0.,0.);

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();

     // Create arrays for the grid data
     NCVariable<double> gmass;
     NCVariable<Vector> gvelocity;
     CCVariable<double> cmass;
     CCVariable<Vector> cmomentum;

     new_dw->get(gmass,     Mlb->gMassLabel,           matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gvelocity, Mlb->gVelocityStarLabel,   matlindex, patch,
							Ghost::AroundCells, 1);

     new_dw->allocate(cmass,     MIlb->cMassLabel,     matlindex, patch);
     new_dw->allocate(cmomentum, MIlb->cMomentumLabel, matlindex, patch);
 
     IntVector nodeIdx[8];

     for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
       patch->findNodesFromCell(*iter,nodeIdx);
       cmomentum[*iter] = zero;
       cmass[*iter]     = 0.;
       for (int in=0;in<8;in++){
 	 cmomentum[*iter] += gvelocity[nodeIdx[in]]*gmass[nodeIdx[in]];
	 cmass[*iter]     += gmass[nodeIdx[in]];
       }
     }

     new_dw->put(cmomentum, MIlb->cMomentumLabel, matlindex, patch);
     new_dw->put(cmass,     MIlb->cMassLabel,     matlindex, patch);
  }
}

// $Log$
// Revision 1.13  2001/01/13 01:43:08  harman
// -eliminated step1a
// -changed rho_micro_equil_CCLabel -> rho_micro_CCLabel
//
// Revision 1.12  2001/01/11 20:11:16  guilkey
// Working on getting momentum exchange to work.  It doesnt' yet.
//
// Revision 1.11  2001/01/11 14:13:57  harman
// -changed step names:
//     step1b  ComputeEquilibrationPressure
//     step1c  ComputeFaceCenteredVelocities
//     step1d  AddExchangeContributionToFCVel
//     step2   ComputeDelPressAndUpdatePressCC
//     step3   ComputePressFC
//     step4a  AccumulateMomentumSourceSinks
//     step4b  AccumulateEnergySourceSinks
//     step5b  ComputeLagrangianValues
//     step6&7 AdvectAndAdvanceInTime
//
// Revision 1.9  2001/01/08 20:38:49  jas
// Replace {x,y,z}mom_L_ME with a single CCVariable<Vector> mom_L_ME.
//
// Revision 1.8  2001/01/08 18:29:22  jas
// Replace {x,y,z}mom_L with a single CCVariable<Vector> mom_L.
//
// Revision 1.7  2000/12/29 00:32:03  guilkey
// More.
//
// Revision 1.6  2000/12/28 21:20:10  guilkey
// Adding beginnings of functions for coupling.
//
// Revision 1.5  2000/12/28 20:26:36  guilkey
// More work on coupling MPM and ICE
//
// Revision 1.4  2000/12/27 23:31:13  guilkey
// Fixed some minor problems in MPMICE.
//
// Revision 1.3  2000/12/07 01:25:01  witzel
// Commented out pleaseSave stuff (now done via the problem specification).
//
// Revision 1.2  2000/12/01 23:56:54  guilkey
// Cleaned up the scheduleTimeAdvance.
//
// Revision 1.1  2000/12/01 23:05:02  guilkey
// Adding stuff for coupled MPM and ICE.
//
