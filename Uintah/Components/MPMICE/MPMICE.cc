//
// $Id$
//
#include <Uintah/Components/MPMICE/MPMICE.h>
#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Components/ICE/ICEMaterial.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/Scheduler.h>

#include <Uintah/Components/MPM/Burn/HEBurn.h>
#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Uintah/Components/MPM/MPMPhysicalModules.h>

using namespace Uintah;
using namespace Uintah::MPM;
using namespace Uintah::ICESpace;
using namespace Uintah::MPMICESpace;

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Dot;
using SCICore::Math::Min;
using SCICore::Math::Max;
using namespace std;

MPMICE::MPMICE(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  Mlb = scinew MPMLabel();
  Ilb = scinew ICELabel();
  d_fracture = false;
  d_mpm      = scinew SerialMPM(myworld);
  d_ice      = scinew ICE(myworld);
}

MPMICE::~MPMICE()
{
  delete Mlb;
  delete Ilb;
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
	 /*
	  * computeNodeVisibility
	  *   in(P.X, P.VOLUME, P.ISBROKEN, P.CRACKSURFACENORMAL)
	  *   operation(compute the visibility information of particles to the
	  *             related nodes)
	  * out(P.VISIBILITY)
	  */
	 Task* t = scinew Task(
	    "SerialMPM::computeNodeVisibility",
	    patch, old_dw, new_dw,
	    d_mpm,&SerialMPM::computeNodeVisibility);
	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
	    int idx = matl->getDWIndex();
  	    t->requires(old_dw, Mlb->pXLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, Mlb->pVolumeLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, Mlb->pIsBrokenLabel, idx, patch,
			Ghost::AroundNodes, 1 );
   	    t->requires(old_dw, Mlb->pCrackSurfaceNormalLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->computes(new_dw, Mlb->pVisibilityLabel, idx, patch );
	 }
	 sched->addTask(t);
      }
      
      {
	 /*
	  * interpolateParticlesToGrid
	  *   in(P.MASS, P.VELOCITY, P.NAT_X)
	  *   operation(interpolate the P.MASS and P.VEL to the grid
	  *             using P.NAT_X and some shape function evaluations)
	  * out(G.MASS, G.VELOCITY)
	  */
	 Task* t = scinew Task("SerialMPM::interpolateParticlesToGrid",
			    patch, old_dw, new_dw,
			    d_mpm,&SerialMPM::interpolateParticlesToGrid);
	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    t->requires(old_dw, Mlb->pMassLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, Mlb->pVelocityLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, Mlb->pXLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, Mlb->pExternalForceLabel, idx, patch,
			Ghost::AroundNodes, 1 );

            t->requires(old_dw, Mlb->pTemperatureLabel, idx, patch,
			Ghost::AroundNodes, 1 );
            /*
            t->requires(old_dw, Mlb->pExternalHeatRateLabel, idx, patch,
			Ghost::AroundNodes, 1 );
             */

	    if(mpm_matl->getFractureModel()) {
	      t->requires(old_dw,Mlb->pCrackSurfaceContactForceLabel,idx, patch,
			Ghost::AroundNodes, 1 );
	      t->requires(new_dw, Mlb->pVisibilityLabel, idx, patch,
			Ghost::AroundNodes, 1 );
   	    }

	    t->computes(new_dw, Mlb->gMassLabel, idx, patch );
	    t->computes(new_dw, Mlb->gVelocityLabel, idx, patch );
	    t->computes(new_dw, Mlb->gExternalForceLabel, idx, patch );

            t->computes(new_dw, Mlb->gTemperatureLabel, idx, patch );
            //t->computes(new_dw, Mlb->gExternalHeatRateLabel, idx, patch );
	 }
		     
	 t->computes(new_dw, Mlb->TotalMassLabel);
	 sched->addTask(t);
      }

      if (MPMPhysicalModules::thermalContactModel) {
	 /* computeHeatExchange
	  *   in(G.MASS, G.TEMPERATURE, G.EXTERNAL_HEAT_RATE)
	  *   operation(peform heat exchange which will cause each of
	  *		velocity fields to exchange heat according to 
	  *             the temperature differences)
	  *   out(G.EXTERNAL_HEAT_RATE)
	  */

	 Task* t = scinew Task("ThermalContact::computeHeatExchange",
			    patch, old_dw, new_dw,
			    MPMPhysicalModules::thermalContactModel,
			    &ThermalContact::computeHeatExchange);

	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
            MPMPhysicalModules::thermalContactModel->addComputesAndRequires(
		 t, mpm_matl, patch, old_dw, new_dw);
	 }

	 sched->addTask(t);
      }
      
      {
	 /* exMomInterpolated
	  *   in(G.MASS, G.VELOCITY)
	  *   operation(peform operations which will cause each of
	  *		  velocity fields to feel the influence of the
	  *		  the others according to specific rules)
	  *   out(G.VELOCITY)
	  */

	 Task* t = scinew Task("Contact::exMomInterpolated",
			    patch, old_dw, new_dw,
			    MPMPhysicalModules::contactModel,
			    &Contact::exMomInterpolated);

	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
            MPMPhysicalModules::contactModel->
				addComputesAndRequiresInterpolated(
					t, mpm_matl, patch, old_dw, new_dw);
	 }

	 sched->addTask(t);
      }
      
      {
	 /*
	  * computeStressTensor
	  *   in(G.VELOCITY, P.X, P.DEFORMATIONMEASURE)
	  *   operation(evaluate the gradient of G.VELOCITY at P.X, feed
	  *             this into a constitutive model, which will
	  *	           evaluate the stress and store it in the
	  *             DataWarehouse.  Each CM also computes the maximum
	  *             elastic wave speed for that material in the
	  *             patch.  This is used in calculating delt.)
	  * out(P.DEFORMATIONMEASURE,P.STRESS)
	  */
	 Task* t = scinew Task("SerialMPM::computeStressTensor",
			    patch, old_dw, new_dw,
			    d_mpm, &SerialMPM::computeStressTensor);
	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	       ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
	       cm->addComputesAndRequires(t, mpm_matl, patch, old_dw, new_dw);
	 }
	 
         t->computes(new_dw, Mlb->delTAfterConstitutiveModelLabel);
	   
         t->computes(new_dw, Mlb->StrainEnergyLabel);

	 sched->addTask(t);
      }

      {
	 /*
	  * computeInternalForce
	  *   in(P.CONMOD, P.NAT_X, P.VOLUME)
	  *   operation(evaluate the divergence of the stress (stored in
	  *	       P.CONMOD) using P.NAT_X and the gradients of the
	  *             shape functions)
	  * out(G.F_INTERNAL)
	  */
	 Task* t = scinew Task("SerialMPM::computeInternalForce",
			    patch, old_dw, new_dw,
			    d_mpm, &SerialMPM::computeInternalForce);
	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    
	    t->requires( new_dw, Mlb->pStressAfterStrainRateLabel, idx, patch,
			 Ghost::AroundNodes, 1);
	    
	    t->requires( new_dw, Mlb->pVolumeDeformedLabel, idx, patch,
			 Ghost::AroundNodes, 1);

	    t->requires( old_dw, Mlb->pMassLabel, idx, patch,
			 Ghost::AroundNodes, 1);

            t->requires( new_dw, Mlb->gMassLabel, idx, patch,
                         Ghost::None);

	    if(mpm_matl->getFractureModel()) {
	       t->requires(new_dw, Mlb->pVisibilityLabel, idx, patch,
			Ghost::AroundNodes, 1 );
   	    }

	    t->computes( new_dw, Mlb->gInternalForceLabel, idx, patch );
            t->computes(new_dw, Mlb->gStressForSavingLabel, idx, patch );
	 }

	 sched->addTask( t );
      }
      
      {
	 /*
	  * computeInternalHeatRate
	  *   in(P.X, P.VOLUME, P.TEMPERATUREGRADIENT)
	  *   operation(evaluate the grid internal heat rate using 
	  *   P.TEMPERATUREGRADIENT and the gradients of the
	  *   shape functions)
	  * out(G.INTERNALHEATRATE)
	  */

	 Task* t = scinew Task("SerialMPM::computeInternalHeatRate",
			    patch, old_dw, new_dw,
			    d_mpm, &SerialMPM::computeInternalHeatRate);

	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();

	    t->requires(old_dw, Mlb->pXLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(new_dw, Mlb->pVolumeDeformedLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, Mlb->pTemperatureGradientLabel, idx, patch,
			Ghost::AroundNodes, 1);

	    if(mpm_matl->getFractureModel()) {
	       t->requires(new_dw, Mlb->pVisibilityLabel, idx, patch,
			Ghost::AroundNodes, 1 );
   	    }

	    t->computes( new_dw, Mlb->gInternalHeatRateLabel, idx, patch );
	 }

	 sched->addTask( t );
      }
      
      {
	 /*
	  * solveEquationsMotion
	  *   in(G.MASS, G.F_INTERNAL)
	  *   operation(acceleration = f/m)
	  *   out(G.ACCELERATION)
	  * 
	  */
	 Task* t = scinew Task("SerialMPM::solveEquationsMotion",
			    patch, old_dw, new_dw,
			    d_mpm, &SerialMPM::solveEquationsMotion);
	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, Mlb->gMassLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, Mlb->gInternalForceLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, Mlb->gExternalForceLabel, idx, patch,
			 Ghost::None);

	    t->computes( new_dw, Mlb->gAccelerationLabel, idx, patch);
	 }

	 sched->addTask(t);
      }

      {
	 /*
	  * solveHeatEquations
	  *   in(G.MASS, G.INTERNALHEATRATE, G.EXTERNALHEATRATE)
	  *   out(G.TEMPERATURERATE)
	  * 
	  */
	 Task* t = scinew Task("SerialMPM::solveHeatEquations",
			    patch, old_dw, new_dw,
			    d_mpm, &SerialMPM::solveHeatEquations);
	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, Mlb->gMassLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, Mlb->gInternalHeatRateLabel, idx, patch,
			 Ghost::None);
	    /*
	    t->requires( new_dw, Mlb->gExternalHeatRateLabel, idx, patch,
			 Ghost::None);
		*/

	    if(MPMPhysicalModules::thermalContactModel) {
              t->requires( new_dw,
			Mlb->gThermalContactHeatExchangeRateLabel, idx, 
	                 patch, Ghost::None);
	    }
		
	    t->computes( new_dw, Mlb->gTemperatureRateLabel, idx, patch);
	 }

	 sched->addTask(t);
      }

      {
	 /*
	  * integrateAcceleration
	  *   in(G.ACCELERATION, G.VELOCITY)
	  *   operation(v* = v + a*dt)
	  *   out(G.VELOCITY_STAR)
	  * 
	  */
	 Task* t = scinew Task("SerialMPM::integrateAcceleration",
			    patch, old_dw, new_dw,
			    d_mpm, &SerialMPM::integrateAcceleration);
	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires(new_dw, Mlb->gAccelerationLabel, idx, patch,
			Ghost::None);
	    t->requires(new_dw, Mlb->gMomExedVelocityLabel, idx, patch,
			Ghost::None);
	    t->requires(old_dw, d_sharedState->get_delt_label() );

	    t->computes(new_dw, Mlb->gVelocityStarLabel, idx, patch );
	 }
		     
	 sched->addTask(t);
      }

      {
	 /*
	  * integrateTemperatureRate
	  *   in(G.TEMPERATURE, G.TEMPERATURERATE)
	  *   operation(t* = t + t_rate * dt)
	  *   out(G.TEMPERATURE_STAR)
	  * 
	  */
	 Task* t = scinew Task("SerialMPM::integrateTemperatureRate",
			    patch, old_dw, new_dw,
			    d_mpm, &SerialMPM::integrateTemperatureRate);
	 for(int m = 0; m < numMPMMatls; m++) {
	    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
	    int idx = matl->getDWIndex();

            t->requires( new_dw, Mlb->gTemperatureLabel, idx, 
	                 patch, Ghost::None);
            t->requires( new_dw, Mlb->gTemperatureRateLabel, idx, 
	                 patch, Ghost::None);
	    t->requires( old_dw, d_sharedState->get_delt_label() );
		     
            t->computes( new_dw, Mlb->gTemperatureStarLabel, idx, patch );
	 }
		     
	 sched->addTask(t);
      }
      
      {
	 /* exMomIntegrated
	  *   in(G.MASS, G.VELOCITY_STAR, G.ACCELERATION)
	  *   operation(peform operations which will cause each of
	  *		  velocity fields to feel the influence of the
	  *		  the others according to specific rules)
	  *   out(G.VELOCITY_STAR, G.ACCELERATION)
	  */

	Task* t = scinew Task("Contact::exMomIntegrated",
			   patch, old_dw, new_dw,
			   MPMPhysicalModules::contactModel,
			   &Contact::exMomIntegrated);
	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
            MPMPhysicalModules::contactModel->
				addComputesAndRequiresIntegrated(
				t, mpm_matl, patch, old_dw, new_dw);
	}

	sched->addTask(t);
      }

      {
	/* interpolateNCToCC */

	 Task* t=scinew Task("MPMICE::interpolateNCToCC",
		    patch, old_dw, new_dw,
		    this, &MPMICE::interpolateNCToCC);

	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    t->requires(new_dw, Mlb->gMomExedVelocityStarLabel, idx, patch,
			Ghost::AroundCells, 1);
	    t->requires(new_dw, Mlb->gMassLabel,                idx, patch,
			Ghost::AroundCells, 1);
	    t->computes(new_dw, Mlb->cVelocityLabel, idx, patch);
	 }

	sched->addTask(t);
      }

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

      {
	 /*
	  * interpolateToParticlesAndUpdate
	  *   in(G.ACCELERATION, G.VELOCITY_STAR, P.NAT_X)
	  *   operation(interpolate acceleration and v* to particles and
	  *             integrate these to get new particle velocity and
	  *             position)
	  * out(P.VELOCITY, P.X, P.NAT_X)
	  */
	 Task* t=scinew Task("SerialMPM::interpolateToParticlesAndUpdate",
		    patch, old_dw, new_dw,
		    d_mpm, &SerialMPM::interpolateToParticlesAndUpdate);

	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    t->requires(new_dw, Mlb->gMomExedAccelerationLabel, idx, patch,
			Ghost::AroundCells, 1);
	    t->requires(new_dw, Mlb->gMomExedVelocityStarLabel, idx, patch,
			Ghost::AroundCells, 1);
	    t->requires(old_dw, Mlb->pXLabel, idx, patch,
			Ghost::None);
	    t->requires(old_dw, Mlb->pExternalForceLabel, idx, patch,
			Ghost::None);

	    if(mpm_matl->getFractureModel()) {
	       t->requires(new_dw, Mlb->pVisibilityLabel, idx, patch,
			Ghost::None);
	       t->requires(old_dw, Mlb->pCrackSurfaceContactForceLabel,
						idx, patch, Ghost::None);
   	    }

	    t->requires(old_dw, d_sharedState->get_delt_label() );
						
	    t->requires(old_dw, Mlb->pMassLabel, idx, patch, Ghost::None);
	    t->computes(new_dw, Mlb->pVelocityLabel_preReloc, idx, patch );
	    t->computes(new_dw, Mlb->pXLabel_preReloc, idx, patch );
	    t->computes(new_dw, Mlb->pExternalForceLabel_preReloc,
							idx, patch);

	    t->requires(old_dw, Mlb->pParticleIDLabel, idx, patch,
							Ghost::None);
	    t->computes(new_dw, Mlb->pParticleIDLabel_preReloc,idx, patch);

            t->requires(old_dw, Mlb->pTemperatureLabel, idx, patch,
			Ghost::None);
            t->requires(new_dw, Mlb->gTemperatureRateLabel, idx, patch,
			Ghost::AroundCells, 1);
            t->requires(new_dw, Mlb->gTemperatureLabel, idx, patch,
			Ghost::AroundCells, 1);

            t->requires(new_dw, Mlb->gTemperatureStarLabel, idx, patch,
			Ghost::AroundCells, 1);
            t->computes(new_dw, Mlb->pTemperatureRateLabel_preReloc,
							idx, patch);
            t->computes(new_dw, Mlb->pTemperatureLabel_preReloc,
							idx, patch);
            t->computes(new_dw, Mlb->pTemperatureGradientLabel_preReloc,
							idx, patch);
	 }

	 t->computes(new_dw, Mlb->KineticEnergyLabel);
	 t->computes(new_dw, Mlb->CenterOfMassPositionLabel);
	 t->computes(new_dw, Mlb->CenterOfMassVelocityLabel);
	 sched->addTask(t);
      }

      {
	 /*
	  * computeMassRate
	  * in(P.TEMPERATURE_RATE)
	  * operation(based on the heat flux history, determine if
	  * each of the particles has ignited, adjust the mass of those
	  * particles which are burning)
	  * out(P.IGNITED)
	  *
	  */
	 Task *t = scinew Task("SerialMPM::computeMassRate",
                            patch, old_dw, new_dw,
                            d_mpm, &SerialMPM::computeMassRate);
         for(int m = 0; m < numMPMMatls; m++){
            MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
            HEBurn* heb = mpm_matl->getBurnModel();
            heb->addComputesAndRequires
				(t, mpm_matl, patch, old_dw, new_dw);
	    d_burns=heb->getBurns();
         }
         sched->addTask(t);

      }

      if(d_fracture) {
	 /*
	  * crackGrow
	  *   in(p.stress,p.isBroken,p.crackSurfaceNormal)
	  *   operation(check the stress on each particle to see
	  *   if the microcrack will initiate and/or grow)
	  * out(p.isBroken,p.crackSurfaceNormal)
	  */
	 Task* t = scinew Task("SerialMPM::crackGrow",
			    patch, old_dw, new_dw,
			    d_mpm,&SerialMPM::crackGrow);

	 for(int m = 0; m < numMPMMatls; m++) {
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    if(mpm_matl->getFractureModel()) {
	      t->requires( new_dw, Mlb->pStressAfterStrainRateLabel,
			idx, patch, Ghost::None);
	      t->requires( old_dw, Mlb->pIsBrokenLabel, idx, patch,
			 Ghost::None);
	      t->requires( old_dw, Mlb->pCrackSurfaceNormalLabel,
			idx, patch, Ghost::None);
	      t->requires( old_dw, Mlb->pTensileStrengthLabel, idx, patch,
			 Ghost::None);
	      t->requires( old_dw, Mlb->pVolumeLabel, idx, patch,
			 Ghost::None);
	      t->requires( new_dw, Mlb->pRotationRateLabel, idx, patch,
			 Ghost::None);
	      t->requires( new_dw, Mlb->pStressAfterStrainRateLabel,
			idx, patch, Ghost::None);
			 
 	      t->requires(old_dw, Mlb->delTLabel );
 	      t->requires(new_dw, Mlb->delTAfterConstitutiveModelLabel );

	      t->computes( new_dw, Mlb->pIsBrokenLabel_preReloc,
			 idx, patch);
	      t->computes( new_dw, Mlb->pCrackSurfaceNormalLabel_preReloc,
			idx, patch );
	      t->computes( new_dw, Mlb->pTensileStrengthLabel_preReloc,
			idx, patch );
	      t->computes( new_dw, Mlb->pIsNewlyBrokenLabel, idx, patch );
	    }
	 }

         t->computes(new_dw, Mlb->delTAfterFractureLabel);
	 sched->addTask(t);
      }

      if(d_fracture) {
	 /*
	  * stressRelease
	  *   in(p.x,p.stressBeforeFractureRelease,p.isNewlyBroken,
	  *   p.crackSurfaceNormal)
	  *   operation(check the stress on each particle to see
	  *   if the microcrack will initiate and/or grow)
	  * out(p.stress)
	  */
	 Task* t = scinew Task("SerialMPM::stressRelease",
			    patch, old_dw, new_dw,
			    d_mpm,&SerialMPM::stressRelease);

	 for(int m = 0; m < numMPMMatls; m++) {
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    if(mpm_matl->getFractureModel()) {
	      t->requires( old_dw, Mlb->pXLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	      t->requires( new_dw, Mlb->pIsNewlyBrokenLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	      t->requires( new_dw, Mlb->pCrackSurfaceNormalLabel_preReloc,
			idx, patch, Ghost::AroundNodes, 1 );
	      t->requires( new_dw, Mlb->pStressAfterStrainRateLabel,
			idx, patch, Ghost::None);			 
	      t->computes( new_dw, Mlb->pStressAfterFractureReleaseLabel,
			idx, patch );
	    }
	 }
	 sched->addTask(t);
      }
      
      if(d_fracture) {
	 /*
	  * computeCrackSurfaceContactForce
	  *   in(P.X, P.VOLUME, P.ISBROKEN, P.CRACKSURFACENORMAL)
	  *   operation(compute the surface contact force)
	  * out(P.SURFACECONTACTFORCE)
	  */
	 Task* t = scinew Task(
	    "SerialMPM::computeCrackSurfaceContactForce",
	    patch, old_dw, new_dw,
	    d_mpm,&SerialMPM::computeCrackSurfaceContactForce);
	 for(int m = 0; m < numMPMMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    if(mpm_matl->getFractureModel()) {
  	       mpm_matl->getConstitutiveModel()->
	         addComputesAndRequiresForCrackSurfaceContact(t,mpm_matl,
		    patch,old_dw,new_dw);
   	    }
	 }
         t->computes(new_dw, Mlb->delTAfterCrackSurfaceContactLabel);
	 sched->addTask(t);
      }      
      
      {
	 /*
	  * carryForwardVariables
	  *   in(p.x,p.stressBeforeFractureRelease,p.isNewlyBroken,
	  *   p.crackSurfaceNormal)
	  *   operation(check the stress on each particle to see
	  *   if the microcrack will initiate and/or grow)
	  * out(p.stress)
	  */
	 Task* t = scinew Task("SerialMPM::carryForwardVariables",
			    patch, old_dw, new_dw,
			    d_mpm,&SerialMPM::carryForwardVariables);

	 for(int m = 0; m < numMPMMatls; m++) {
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    
	    if(mpm_matl->getFractureModel()) {
	      t->requires( new_dw, Mlb->pStressAfterFractureReleaseLabel,
			idx, patch, Ghost::None);
	      t->requires( new_dw, Mlb->delTAfterCrackSurfaceContactLabel);
	    }
	    else {
	      t->requires( new_dw, Mlb->pStressAfterStrainRateLabel,
			idx, patch, Ghost::None);			 
	      t->requires( new_dw, Mlb->delTAfterConstitutiveModelLabel);
            }
	    t->computes(new_dw, Mlb->pStressLabel_preReloc, idx, patch);
	 }
	 t->computes(new_dw, Mlb->delTLabel);
	 sched->addTask(t);
      }

#if 0
    }

  for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;
#endif

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
    // Step 5b cell centered momentum exchange
    d_ice->scheduleStep5b(patch,sched,old_dw,new_dw);
    // Step 6and7 advect and advance in time
    d_ice->scheduleStep6and7(patch,sched,old_dw,new_dw);
#if 0
      // Step 1a  computeSoundSpeed
      {
	Task* t = scinew Task("ICE::step1a",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep1a);
	for (int m = 0; m < numICEMatls; m++) {
          ICEMaterial* matl = d_sharedState->getICEMaterial(m);
          EquationOfState* eos = matl->getEOS();
	  // Compute the speed of sound
          eos->addComputesAndRequiresSS(t,matl,patch,old_dw,new_dw);
	}
	sched->addTask(t);
      }

      // Step 1b calculate equlibration pressure
      {
	Task* t = scinew Task("ICE::step1b",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep1b);

	t->requires(old_dw,Ilb->press_CCLabel, 0,patch,Ghost::None);

	for (int m = 0; m < numICEMatls; m++) {
	  ICEMaterial*  matl = d_sharedState->getICEMaterial(m);
	  int dwindex = matl->getDWIndex();
	  EquationOfState* eos = matl->getEOS();
	  // Compute the rho micro
          eos->addComputesAndRequiresRM(t,matl,patch,old_dw,new_dw);
	  t->requires(old_dw,Ilb->vol_frac_CCLabel,  dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->rho_CCLabel,       dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->rho_micro_CCLabel, dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->temp_CCLabel,      dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->cv_CCLabel,        dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->speedSound_CCLabel,dwindex,patch,Ghost::None);
	  t->computes(new_dw,Ilb->vol_frac_CCLabel,          dwindex, patch);
	  t->computes(new_dw,Ilb->speedSound_equiv_CCLabel,  dwindex, patch);
	  t->computes(new_dw,Ilb->rho_micro_equil_CCLabel,   dwindex, patch);
	}

        t->computes(new_dw,Ilb->press_CCLabel,0, patch);

	sched->addTask(t);
      }

      // Step 1c compute face centered velocities
      {
	Task* t = scinew Task("ICE::step1c",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep1c);

	t->requires(new_dw,Ilb->press_CCLabel,0,patch,Ghost::None);

	for (int m = 0; m < numICEMatls; m++) {
	  ICEMaterial* matl = d_sharedState->getICEMaterial(m);
	  int dwindex = matl->getDWIndex();
	  t->requires(old_dw,Ilb->rho_CCLabel,   dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->uvel_CCLabel,  dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->vvel_CCLabel,  dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->wvel_CCLabel,  dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->rho_micro_equil_CCLabel,
						dwindex,patch,Ghost::None);


	  t->computes(new_dw,Ilb->uvel_FCLabel,  dwindex, patch);
	  t->computes(new_dw,Ilb->vvel_FCLabel,  dwindex, patch);
	  t->computes(new_dw,Ilb->wvel_FCLabel,  dwindex, patch);
	}
	sched->addTask(t);
      }

      // Step 1d computes momentum exchange on FC velocities
      {
	Task* t = scinew Task("ICE::step1d",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep1d);

	for (int m = 0; m < numICEMatls; m++) {
	  ICEMaterial* matl = d_sharedState->getICEMaterial(m);
	  int dwindex = matl->getDWIndex();
	  t->requires(new_dw,Ilb->rho_micro_equil_CCLabel,
			dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->vol_frac_CCLabel, dwindex, patch,Ghost::None);
	  t->requires(old_dw,Ilb->uvel_FCLabel,     dwindex, patch,Ghost::None);
	  t->requires(old_dw,Ilb->vvel_FCLabel,     dwindex, patch,Ghost::None);
	  t->requires(old_dw,Ilb->wvel_FCLabel,     dwindex, patch,Ghost::None);

	  t->computes(new_dw,Ilb->uvel_FCMELabel,   dwindex, patch);
	  t->computes(new_dw,Ilb->vvel_FCMELabel,   dwindex, patch);
	  t->computes(new_dw,Ilb->wvel_FCMELabel,   dwindex, patch);
	}
	sched->addTask(t);
      }

      // Step 2 computes delPress and the new pressure
      {
	Task* t = scinew Task("ICE::step2",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep2);

	t->requires(new_dw,Ilb->press_CCLabel, 0,patch,Ghost::None);

	for (int m = 0; m < numICEMatls; m++) {
	  ICEMaterial* matl = d_sharedState->getICEMaterial(m);
	  int dwindex = matl->getDWIndex();
	  t->requires(new_dw,Ilb->vol_frac_CCLabel, dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->uvel_FCMELabel, dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->vvel_FCMELabel, dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->wvel_FCMELabel, dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->speedSound_equiv_CCLabel,
						 dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->rho_micro_equil_CCLabel,
						 dwindex,patch,Ghost::None);

	  t->computes(new_dw,Ilb->div_velfc_CCLabel,dwindex,patch);
	}

	t->computes(new_dw,Ilb->pressdP_CCLabel,  0, patch);
	t->computes(new_dw,Ilb->delPress_CCLabel, 0, patch);

	sched->addTask(t);
      }

      // Step 3 compute face centered pressure
      {
	Task* t = scinew Task("ICE::step3",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep3);

	t->requires(new_dw,Ilb->pressdP_CCLabel,0,patch,Ghost::None);

	for (int m = 0; m < numICEMatls; m++) {
	  ICEMaterial* matl = d_sharedState->getICEMaterial(m);
	  int dwindex = matl->getDWIndex();
	  t->requires(old_dw,Ilb->rho_CCLabel, dwindex,patch,Ghost::None);
	}

	t->computes(new_dw,Ilb->press_FCLabel, 0, patch);

	sched->addTask(t);
      }

      // Step 4a compute sources of momentum
      {
	Task* t = scinew Task("ICE::step4a",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep4a);

	t->requires(new_dw,Ilb->press_FCLabel, 0,patch,Ghost::None);

	for (int m = 0; m < numICEMatls; m++) {
	  ICEMaterial* matl = d_sharedState->getICEMaterial(m);
	  int dwindex = matl->getDWIndex();
	  t->requires(old_dw,Ilb->rho_CCLabel,       dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->uvel_CCLabel,      dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->vvel_CCLabel,      dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->wvel_CCLabel,      dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->viscosity_CCLabel, dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->vol_frac_CCLabel,  dwindex,patch,Ghost::None);

	  t->computes(new_dw,Ilb->xmom_source_CCLabel, dwindex,patch);
	  t->computes(new_dw,Ilb->ymom_source_CCLabel, dwindex,patch);
	  t->computes(new_dw,Ilb->zmom_source_CCLabel, dwindex,patch);
	  t->computes(new_dw,Ilb->tau_X_FCLabel,       dwindex,patch);
	  t->computes(new_dw,Ilb->tau_Y_FCLabel,       dwindex,patch);
	  t->computes(new_dw,Ilb->tau_Z_FCLabel,       dwindex,patch);
	}
	sched->addTask(t);
      }

      // Step 4b compute sources of energy
      {
	Task* t = scinew Task("ICE::step4b",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep4b);

	t->requires(new_dw,Ilb->press_CCLabel,    0,patch,Ghost::None);
	t->requires(new_dw,Ilb->delPress_CCLabel, 0,patch,Ghost::None);

	for (int m = 0; m < numICEMatls; m++) {
	  ICEMaterial* matl = d_sharedState->getICEMaterial(m);
	  int dwindex = matl->getDWIndex();
	  t->requires(new_dw,Ilb->rho_micro_equil_CCLabel,
			dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->speedSound_equiv_CCLabel,
			dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->vol_frac_CCLabel, dwindex,patch,Ghost::None);

	  t->computes(new_dw,Ilb->int_eng_source_CCLabel, dwindex,patch);
	}
	sched->addTask(t);
      }

      // Step 5a compute lagrangian quantities
      {
	Task* t = scinew Task("ICE::step5a",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep5a);
	for (int m = 0; m < numICEMatls; m++) {
	 ICEMaterial* matl = d_sharedState->getICEMaterial(m);
	 int dwindex = matl->getDWIndex();
	 t->requires(old_dw,Ilb->rho_CCLabel,        dwindex,patch,Ghost::None);
	 t->requires(old_dw,Ilb->uvel_CCLabel,       dwindex,patch,Ghost::None);
	 t->requires(old_dw,Ilb->vvel_CCLabel,       dwindex,patch,Ghost::None);
	 t->requires(old_dw,Ilb->wvel_CCLabel,       dwindex,patch,Ghost::None);
	 t->requires(old_dw,Ilb->cv_CCLabel,         dwindex,patch,Ghost::None);
	 t->requires(old_dw,Ilb->temp_CCLabel,       dwindex,patch,Ghost::None);
	 t->requires(new_dw,Ilb->xmom_source_CCLabel,dwindex,patch,Ghost::None);
	 t->requires(new_dw,Ilb->ymom_source_CCLabel,dwindex,patch,Ghost::None);
	 t->requires(new_dw,Ilb->zmom_source_CCLabel,dwindex,patch,Ghost::None);
	 t->requires(new_dw,Ilb->int_eng_source_CCLabel,
						     dwindex,patch,Ghost::None);

	 t->computes(new_dw,Ilb->xmom_L_CCLabel,     dwindex, patch);
	 t->computes(new_dw,Ilb->ymom_L_CCLabel,     dwindex, patch);
	 t->computes(new_dw,Ilb->zmom_L_CCLabel,     dwindex, patch);
	 t->computes(new_dw,Ilb->int_eng_L_CCLabel,  dwindex, patch);
	 t->computes(new_dw,Ilb->mass_L_CCLabel,     dwindex, patch);
	 t->computes(new_dw,Ilb->rho_L_CCLabel,      dwindex, patch);
	}
	sched->addTask(t);
      }

      // Step 5b cell centered momentum exchange
      {
	Task* t = scinew Task("ICE::step5b",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep5b);
	for (int m = 0; m < numICEMatls; m++) {
	  ICEMaterial* matl = d_sharedState->getICEMaterial(m);
	  int dwindex = matl->getDWIndex();
	  t->requires(old_dw,Ilb->rho_CCLabel,       dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->xmom_L_CCLabel,    dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->ymom_L_CCLabel,    dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->zmom_L_CCLabel,    dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->int_eng_L_CCLabel, dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->vol_frac_CCLabel,  dwindex,patch,Ghost::None);
	  t->requires(old_dw,Ilb->cv_CCLabel,        dwindex,patch,Ghost::None);
	  t->requires(new_dw,Ilb->rho_micro_equil_CCLabel,
						    dwindex,patch,Ghost::None);

	  t->computes(new_dw,Ilb->xmom_L_ME_CCLabel,    dwindex, patch);
	  t->computes(new_dw,Ilb->ymom_L_ME_CCLabel,    dwindex, patch);
	  t->computes(new_dw,Ilb->zmom_L_ME_CCLabel,    dwindex, patch);
	  t->computes(new_dw,Ilb->int_eng_L_ME_CCLabel, dwindex, patch);
	}
	sched->addTask(t);
      }

      // Step 6and7 advect and advance in time
      {
	Task* t = scinew Task("ICE::step6and7",patch, old_dw, new_dw,d_ice,
			       &ICE::actuallyStep6and7);
	for (int m = 0; m < numICEMatls; m++ ) {
	  ICEMaterial* matl = d_sharedState->getICEMaterial(m);
	  int dwindex = matl->getDWIndex();
	  t->requires(new_dw, Ilb->xmom_L_ME_CCLabel,
				dwindex,patch,Ghost::None,0);
	  t->requires(new_dw, Ilb->ymom_L_ME_CCLabel,
				dwindex,patch,Ghost::None,0);
	  t->requires(new_dw, Ilb->zmom_L_ME_CCLabel,
				dwindex,patch,Ghost::None,0);
	  t->requires(new_dw, Ilb->int_eng_L_ME_CCLabel,
				dwindex,patch,Ghost::None,0);

	  t->computes(new_dw, Ilb->temp_CCLabel,dwindex, patch);
	  t->computes(new_dw, Ilb->rho_CCLabel, dwindex, patch);
	  t->computes(new_dw, Ilb->cv_CCLabel,  dwindex, patch);
	  t->computes(new_dw, Ilb->uvel_CCLabel,dwindex, patch);
	  t->computes(new_dw, Ilb->vvel_CCLabel,dwindex, patch);
	  t->computes(new_dw, Ilb->wvel_CCLabel,dwindex, patch);
	}
//	t->computes(new_dw, d_sharedState->get_delt_label());
	sched->addTask(t);
      }
#endif

  }

    
   sched->scheduleParticleRelocation(level, old_dw, new_dw,
				     Mlb->pXLabel_preReloc, 
				     Mlb->d_particleState_preReloc,
				     Mlb->pXLabel, Mlb->d_particleState,
				     numMPMMatls);

   new_dw->pleaseSave(Mlb->pXLabel, numMPMMatls);
   new_dw->pleaseSave(Mlb->pVolumeLabel, numMPMMatls);
   new_dw->pleaseSave(Mlb->pStressLabel, numMPMMatls);

   new_dw->pleaseSave(Mlb->gMassLabel, numMPMMatls);

   // Add pleaseSaves here for each of the grid variables
   // created by interpolateParticlesForSaving
   new_dw->pleaseSave(Mlb->gStressForSavingLabel, numMPMMatls);

   if(d_fracture) {
     new_dw->pleaseSave(Mlb->pCrackSurfaceNormalLabel, numMPMMatls);
     new_dw->pleaseSave(Mlb->pIsBrokenLabel, numMPMMatls);
   }

   new_dw->pleaseSaveIntegrated(Mlb->StrainEnergyLabel);
   new_dw->pleaseSaveIntegrated(Mlb->KineticEnergyLabel);
   new_dw->pleaseSaveIntegrated(Mlb->TotalMassLabel);
   new_dw->pleaseSaveIntegrated(Mlb->CenterOfMassPositionLabel);
   new_dw->pleaseSaveIntegrated(Mlb->CenterOfMassVelocityLabel);
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

     new_dw->get(gmass,     Mlb->gMassLabel,                matlindex, patch,
					   Ghost::AroundCells, 1);
     new_dw->get(gvelocity, Mlb->gMomExedVelocityStarLabel, matlindex, patch,
					   Ghost::AroundCells, 1);
     new_dw->allocate(cmass,     Mlb->cMassLabel,     matlindex, patch);
     new_dw->allocate(cvelocity, Mlb->cVelocityLabel, matlindex, patch);
 
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
     new_dw->put(cvelocity, Mlb->cVelocityLabel, matlindex, patch);
  }
}

// $Log$
// Revision 1.1  2000/12/01 23:05:02  guilkey
// Adding stuff for coupled MPM and ICE.
//
