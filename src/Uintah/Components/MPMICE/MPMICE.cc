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

void MPMICE::scheduleComputeStableTimestep(const LevelP&,
					   SchedulerP&,
					   DataWarehouseP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
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

    scheduleInterpolateNCToCC(patch,sched,old_dw,new_dw);

    // Step 1a  computeSoundSpeed
    d_ice->scheduleStep1a(patch,sched,old_dw,new_dw);
    // Step 1b calculate equlibration pressure
    d_ice->scheduleStep1b(patch,sched,old_dw,new_dw);
    // Step 1c compute face centered velocities
    d_ice->scheduleStep1c(patch,sched,old_dw,new_dw);
    // Step 1d computes momentum exchange on FC velocities
    d_ice->scheduleStep1d(patch,sched,old_dw,new_dw);
    // Step 2 computes delPress and the new pressure
    d_ice->scheduleStep2(patch,sched,old_dw,new_dw);
    // Step 3 compute face centered pressure
    d_ice->scheduleStep3(patch,sched,old_dw,new_dw);
    // Step 4a compute sources of momentum
    d_ice->scheduleStep4a(patch,sched,old_dw,new_dw);
    // Step 4b compute sources of energy
    d_ice->scheduleStep4b(patch,sched,old_dw,new_dw);
    // Step 5a compute lagrangian quantities
    d_ice->scheduleStep5a(patch,sched,old_dw,new_dw);

    scheduleCCMomExchange(patch,sched,old_dw,new_dw);

//    d_mpm->scheduleExMomIntegrated(patch,sched,old_dw,new_dw);
    d_mpm->scheduleInterpolateToParticlesAndUpdate(patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeMassRate(patch,sched,old_dw,new_dw);
    if(d_fracture) {
      d_mpm->scheduleCrackGrow(patch,sched,old_dw,new_dw);
      d_mpm->scheduleStressRelease(patch,sched,old_dw,new_dw);
      d_mpm->scheduleComputeCrackSurfaceContactForce(patch,sched,old_dw,new_dw);
    }
    d_mpm->scheduleCarryForwardVariables(patch,sched,old_dw,new_dw);

    // Step 5b cell centered momentum exchange
//    d_ice->scheduleStep5b(patch,sched,old_dw,new_dw);
    // Step 6and7 advect and advance in time
    d_ice->scheduleStep6and7(patch,sched,old_dw,new_dw);

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
     t->computes(new_dw, MIlb->cVelocityLabel, idx, patch);
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
     t->requires(new_dw, MIlb->cVelocityLabel, mpmidx, patch, Ghost::None, 0);
     t->requires(new_dw, MIlb->cMassLabel,     mpmidx, patch, Ghost::None, 0);

     t->computes(new_dw, Mlb->gMomExedVelocityStarLabel, mpmidx, patch);
     t->computes(new_dw, Mlb->gMomExedAccelerationLabel, mpmidx, patch);
   }

  for (int m = 0; m < d_sharedState->getNumICEMatls(); m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int iceidx = matl->getDWIndex();
    t->requires(old_dw,Ilb->rho_CCLabel,       iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->mom_L_CCLabel,    iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->int_eng_L_CCLabel, iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->vol_frac_CCLabel,  iceidx,patch,Ghost::None);
    t->requires(old_dw,Ilb->cv_CCLabel,        iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->rho_micro_equil_CCLabel, iceidx,patch,Ghost::None);
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
  int numMatls = d_sharedState->getNumMPMMatls();
  // Create arrays for the grid data
  NCVariable<Vector> gacceleration, gvelocity;
  NCVariable<Vector> gMEacceleration, gMEvelocity;

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();

    new_dw->get(gvelocity,     Mlb->gVelocityStarLabel, matlindex, patch,
							Ghost::None, 0);
    new_dw->get(gacceleration, Mlb->gAccelerationLabel, matlindex, patch,
							Ghost::None, 0);

    new_dw->allocate(gMEvelocity,     Mlb->gMomExedVelocityStarLabel,
							 matlindex, patch);
    new_dw->allocate(gMEacceleration, Mlb->gMomExedAccelerationLabel,
							 matlindex, patch);

    // Just carrying forward for now, real code needs to go here

    new_dw->put(gvelocity,     Mlb->gMomExedVelocityStarLabel,matlindex, patch);
    new_dw->put(gacceleration, Mlb->gMomExedAccelerationLabel,matlindex, patch);
  }

  int numICEMatls = d_sharedState->getNumICEMatls();
  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();

  // Create variables for the required values
  vector<CCVariable<double> > rho_CC(numICEMatls);
  vector<CCVariable<Vector> > mom_L(numICEMatls);
  vector<CCVariable<double> > int_eng_L(numICEMatls);
  vector<CCVariable<double> > vol_frac_CC(numICEMatls);
  vector<CCVariable<double> > rho_micro_CC(numICEMatls);
  vector<CCVariable<double> > cv_CC(numICEMatls);

  // Create variables for the results
  vector<CCVariable<Vector> > mom_L_ME(numICEMatls);
  vector<CCVariable<double> > int_eng_L_ME(numICEMatls);

  vector<double> b(numICEMatls);
  vector<double> mass(numICEMatls);
//  DenseMatrix beta(numICEMatls,numICEMatls),acopy(numICEMatls,numICEMatls);
//  DenseMatrix K(numICEMatls,numICEMatls),H(numICEMatls,numICEMatls);
//  DenseMatrix a(numICEMatls,numICEMatls);

//  for (int i = 0; i < numICEMatls; i++ ) {
//      K[numICEMatls-1-i][i] = d_K_mom[i];
//      H[numICEMatls-1-i][i] = d_K_heat[i];
//  }

  for(int m = 0; m < numICEMatls; m++){
    ICEMaterial* matl = d_sharedState->getICEMaterial( m );
    int dwindex = matl->getDWIndex();
    old_dw->get(rho_CC[m],       Ilb->rho_CCLabel,
                                dwindex, patch, Ghost::None, 0);
    new_dw->get(mom_L[m],       Ilb->mom_L_CCLabel,
                                dwindex, patch, Ghost::None, 0);
    new_dw->get(int_eng_L[m],    Ilb->int_eng_L_CCLabel,
                                dwindex, patch, Ghost::None, 0);
    new_dw->get(vol_frac_CC[m],  Ilb->vol_frac_CCLabel,
                                dwindex, patch, Ghost::None, 0);
    new_dw->get(rho_micro_CC[m], Ilb->rho_micro_equil_CCLabel,
                                dwindex, patch, Ghost::None, 0);
    old_dw->get(cv_CC[m],        Ilb->cv_CCLabel,
                                dwindex, patch, Ghost::None, 0);

    new_dw->allocate(mom_L_ME[m],   Ilb->mom_L_ME_CCLabel,    dwindex, patch);
    new_dw->allocate(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex, patch);
  }

   
  for (int m = 0; m < numICEMatls; m++) {
    mom_L_ME[m] = mom_L[m];
    int_eng_L_ME[m] = int_eng_L[m];
  }

  for(int m = 0; m < numICEMatls; m++){
     ICEMaterial* matl = d_sharedState->getICEMaterial( m );
     int dwindex = matl->getDWIndex();
     new_dw->put(mom_L_ME[m],   Ilb->mom_L_ME_CCLabel,   dwindex, patch);
     new_dw->put(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex, patch);
  }

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
     CCVariable<Vector> cvelocity;

     new_dw->get(gmass,     Mlb->gMassLabel,           matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gvelocity, Mlb->gVelocityStarLabel,   matlindex, patch,
							Ghost::AroundCells, 1);

     new_dw->allocate(cmass,     MIlb->cMassLabel,     matlindex, patch);
     new_dw->allocate(cvelocity, MIlb->cVelocityLabel, matlindex, patch);
 
     IntVector nodeIdx[8];

     for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      patch->findNodesFromCell(*iter,nodeIdx);
      cvelocity[*iter] = zero;
      cmass[*iter]     = 0.;
      for (int in=0;in<8;in++){
	cvelocity[*iter] += gvelocity[nodeIdx[in]]*gmass[nodeIdx[in]];
	cmass[*iter]     += gmass[nodeIdx[in]];
      }
     }

     for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	cvelocity[*iter] = cvelocity[*iter]/cmass[*iter];
     }

     new_dw->put(cvelocity, MIlb->cVelocityLabel, matlindex, patch);
     new_dw->put(cmass,     MIlb->cMassLabel,     matlindex, patch);
  }
}

// $Log$
// Revision 1.10  2001/01/08 22:00:50  jas
// Removed #if 0  #endif pairs surrounding unused code related to momentum
// variables that are now combined into CCVariables<Vector>.  This includes
// mom_source, mom_L and mom_L_ME.
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
