//
// $Id$
//

#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <Uintah/Parallel/ProcessorGroup.h>

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>

#include <Uintah/Components/MPM/Burn/HEBurn.h>
#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>

#include <Uintah/Components/MPM/Fracture/Visibility.h>

#include <iostream>
#include <fstream>

#include <Uintah/Components/MPM/MPMLabel.h>


#include <Uintah/Grid/BoundCond.h>
#include <Uintah/Grid/VelocityBoundCond.h>
#include <Uintah/Grid/SymmetryBoundCond.h>
#include <Uintah/Grid/TemperatureBoundCond.h>

#include <Uintah/Components/MPM/MPMPhysicalModules.h>
#include <Uintah/Components/MPM/Contact/Contact.h>
#include <Uintah/Components/MPM/HeatConduction/HeatConduction.h>
#include <Uintah/Components/MPM/Fracture/Fracture.h>
#include <Uintah/Components/MPM/ThermalContact/ThermalContact.h>

#include <Uintah/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Uintah/Components/MPM/PhysicalBC/ForceBC.h>

using namespace Uintah;
using namespace Uintah::MPM;

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Dot;
using SCICore::Math::Min;
using SCICore::Math::Max;
using namespace std;

SerialMPM::SerialMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  d_nextOutputTime=0.;
  d_fracture = false;
  d_analyze = NULL;
}

SerialMPM::~SerialMPM()
{
}

void SerialMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
			     SimulationStateP& sharedState)
{
   //The next line is used for data analyze, please do not move.  --tan
   if(d_analyze) d_analyze->problemSetup(prob_spec, grid, sharedState);

   d_sharedState = sharedState;

   MPMPhysicalModules::build(prob_spec,d_sharedState);
   MPMPhysicalBCFactory::create(prob_spec);

   ProblemSpecP p = prob_spec->findBlock("DataArchiver");
   if(!p->get("outputInterval", d_outputInterval))
      d_outputInterval = 1.0;

  cerr << "In SerialMPM::problemSetup . . ." << endl;

  // Search for the MaterialProperties block and then get the MPM section

  ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");

  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");

  for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
     MPMMaterial *mat = scinew MPMMaterial(ps);
     //register as a generic material
     sharedState->registerMaterial(mat);
     //register as an MPM material
     sharedState->registerMPMMaterial(mat);
  }

  cout << "Number of materials: " << d_sharedState->getNumMatls() << endl;


   // Load up all the VarLabels that will be used in each of the
   // physical models
   lb->d_particleState.resize(d_sharedState->getNumMPMMatls());
   lb->d_particleState_preReloc.resize(d_sharedState->getNumMPMMatls());

   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     lb->registerPermanentParticleState(m,lb->pVelocityLabel,
					lb->pVelocityLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pExternalForceLabel,
					lb->pExternalForceLabel_preReloc);

     if(mpm_matl->getBurnModel()->getBurns()){
       //     lb->registerPermanentParticleState(m,lb->pSurfLabel,
       //lb->pSurfLabel_preReloc);
       lb->registerPermanentParticleState(m,lb->pIsIgnitedLabel,
					  lb->pIsIgnitedLabel_preReloc);
     }
     if(mpm_matl->getFractureModel()){
       lb->registerPermanentParticleState(m,lb->pCrackSurfaceNormalLabel,
					 lb->pCrackSurfaceNormalLabel_preReloc); 
       lb->registerPermanentParticleState(m,lb->pTensileStrengthLabel,
					  lb->pTensileStrengthLabel_preReloc); 
       lb->registerPermanentParticleState(m,lb->pIsBrokenLabel,
					  lb->pIsBrokenLabel_preReloc); 
       lb->registerPermanentParticleState(m,lb->pCrackSurfaceContactForceLabel,
				  lb->pCrackSurfaceContactForceLabel_preReloc); 
       lb->registerPermanentParticleState(m,lb->pImageVelocityLabel,
					  lb->pImageVelocityLabel_preReloc); 
     }
     
     lb->registerPermanentParticleState(m,lb->pTemperatureLabel,
					  lb->pTemperatureLabel_preReloc); 
     lb->registerPermanentParticleState(m,lb->pTemperatureGradientLabel,
				  lb->pTemperatureGradientLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pTemperatureRateLabel,
				  lb->pTemperatureRateLabel_preReloc);
       //lb->registerPermanentParticleState(m,lb->pExternalHeatRateLabel,
       //lb->pExternalHeatRateLabel_preReloc); 

     lb->registerPermanentParticleState(m,lb->pParticleIDLabel,
					lb->pParticleIDLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pMassLabel,
					lb->pMassLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pVolumeLabel,
					lb->pVolumeLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pDeformationMeasureLabel,
					lb->pDeformationMeasureLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pStressLabel,
					lb->pStressLabel_preReloc);
     
     mpm_matl->getConstitutiveModel()->addParticleState(lb->d_particleState[m],
					lb->d_particleState_preReloc[m]);
   }
   cerr << "SerialMPM::problemSetup passed.\n";
}

void SerialMPM::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& dw)
{
   Level::const_patchIterator iter;

   for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter; {
	 Task* t = scinew Task("SerialMPM::actuallyInitialize", patch, dw, dw,
			       this, &SerialMPM::actuallyInitialize);
	 t->computes(dw, d_sharedState->get_delt_label());
	 sched->addTask(t);
      }
   }
}

void SerialMPM::scheduleComputeStableTimestep(const LevelP&,
					      SchedulerP&,
					      DataWarehouseP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
}

void SerialMPM::scheduleTimeAdvance(double t, double dt,
				    const LevelP&         level,
				    SchedulerP&     sched,
				    DataWarehouseP& old_dw, 
				    DataWarehouseP& new_dw)
{
   int numMatls = d_sharedState->getNumMPMMatls();

   for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;
    
      //The next line is used for data analyze, please do not move.  --tan
      if(d_analyze) d_analyze->performAnalyze(patch, sched, old_dw, new_dw);

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
	    this,&SerialMPM::computeNodeVisibility);
	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
	    int idx = matl->getDWIndex();
  	    t->requires(old_dw, lb->pXLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, lb->pVolumeLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, lb->pIsBrokenLabel, idx, patch,
			Ghost::AroundNodes, 1 );
   	    t->requires(old_dw, lb->pCrackSurfaceNormalLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->computes(new_dw, lb->pVisibilityLabel, idx, patch );
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
			    this,&SerialMPM::interpolateParticlesToGrid);
	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    t->requires(old_dw, lb->pMassLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, lb->pVelocityLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, lb->pXLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, lb->pExternalForceLabel, idx, patch,
			Ghost::AroundNodes, 1 );

            t->requires(old_dw, lb->pTemperatureLabel, idx, patch,
			Ghost::AroundNodes, 1 );
            /*
            t->requires(old_dw, lb->pExternalHeatRateLabel, idx, patch,
			Ghost::AroundNodes, 1 );
             */

	    if(mpm_matl->getFractureModel()) {
	       t->requires(old_dw,lb->pCrackSurfaceContactForceLabel,idx, patch,
			Ghost::AroundNodes, 1 );
	       t->requires(new_dw, lb->pVisibilityLabel, idx, patch,
			Ghost::AroundNodes, 1 );
   	    }

	    t->computes(new_dw, lb->gMassLabel, idx, patch );
	    t->computes(new_dw, lb->gVelocityLabel, idx, patch );
	    t->computes(new_dw, lb->gExternalForceLabel, idx, patch );

            t->computes(new_dw, lb->gTemperatureLabel, idx, patch );
            //t->computes(new_dw, lb->gExternalHeatRateLabel, idx, patch );
	 }
		     
	 t->computes(new_dw, lb->TotalMassLabel);
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

	 for(int m = 0; m < numMatls; m++){
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

	 for(int m = 0; m < numMatls; m++){
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
			    this, &SerialMPM::computeStressTensor);
	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	       ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
	       cm->addComputesAndRequires(t, mpm_matl, patch, old_dw, new_dw);
	 }
	 
         t->computes(new_dw, lb->delTAfterConstitutiveModelLabel);
	   
         t->computes(new_dw, lb->StrainEnergyLabel);

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
			    this, &SerialMPM::computeInternalForce);
	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    
	    t->requires( new_dw, lb->pStressAfterStrainRateLabel, idx, patch,
			 Ghost::AroundNodes, 1);
	    
	    t->requires( new_dw, lb->pVolumeDeformedLabel, idx, patch,
			 Ghost::AroundNodes, 1);

	    t->requires( old_dw, lb->pMassLabel, idx, patch,
			 Ghost::AroundNodes, 1);

            t->requires( new_dw, lb->gMassLabel, idx, patch,
                         Ghost::None);

	    if(mpm_matl->getFractureModel()) {
	       t->requires(new_dw, lb->pVisibilityLabel, idx, patch,
			Ghost::AroundNodes, 1 );
   	    }

	    t->computes( new_dw, lb->gInternalForceLabel, idx, patch );
            t->computes(new_dw, lb->gStressForSavingLabel, idx, patch );
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
			    this, &SerialMPM::computeInternalHeatRate);

	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();

	    t->requires(old_dw, lb->pXLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(new_dw, lb->pVolumeDeformedLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, lb->pTemperatureGradientLabel, idx, patch,
			Ghost::AroundNodes, 1);

	    if(mpm_matl->getFractureModel()) {
	       t->requires(new_dw, lb->pVisibilityLabel, idx, patch,
			Ghost::AroundNodes, 1 );
   	    }

	    t->computes( new_dw, lb->gInternalHeatRateLabel, idx, patch );
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
			    this, &SerialMPM::solveEquationsMotion);
	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, lb->gMassLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, lb->gInternalForceLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, lb->gExternalForceLabel, idx, patch,
			 Ghost::None);

	    t->computes( new_dw, lb->gAccelerationLabel, idx, patch);
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
			    this, &SerialMPM::solveHeatEquations);
	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, lb->gMassLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, lb->gInternalHeatRateLabel, idx, patch,
			 Ghost::None);
	    /*
	    t->requires( new_dw, lb->gExternalHeatRateLabel, idx, patch,
			 Ghost::None);
		*/

	    if(MPMPhysicalModules::thermalContactModel) {
              t->requires( new_dw,
			lb->gThermalContactHeatExchangeRateLabel, idx, 
	                 patch, Ghost::None);
	    }
		
	    t->computes( new_dw, lb->gTemperatureRateLabel, idx, patch);
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
			    this, &SerialMPM::integrateAcceleration);
	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires(new_dw, lb->gAccelerationLabel, idx, patch,
			Ghost::None);
	    t->requires(new_dw, lb->gMomExedVelocityLabel, idx, patch,
			Ghost::None);
	    t->requires(old_dw, d_sharedState->get_delt_label() );

	    t->computes(new_dw, lb->gVelocityStarLabel, idx, patch );
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
			    this, &SerialMPM::integrateTemperatureRate);
	 for(int m = 0; m < numMatls; m++) {
	    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
	    int idx = matl->getDWIndex();

            t->requires( new_dw, lb->gTemperatureLabel, idx, 
	                 patch, Ghost::None);
            t->requires( new_dw, lb->gTemperatureRateLabel, idx, 
	                 patch, Ghost::None);
	    t->requires( old_dw, d_sharedState->get_delt_label() );
		     
            t->computes( new_dw, lb->gTemperatureStarLabel, idx, patch );
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
	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
            MPMPhysicalModules::contactModel->
				addComputesAndRequiresIntegrated(
				t, mpm_matl, patch, old_dw, new_dw);
	}

	sched->addTask(t);
      }

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
		    this, &SerialMPM::interpolateToParticlesAndUpdate);

	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    t->requires(new_dw, lb->gMomExedAccelerationLabel, idx, patch,
			Ghost::AroundCells, 1);
	    t->requires(new_dw, lb->gMomExedVelocityStarLabel, idx, patch,
			Ghost::AroundCells, 1);
	    t->requires(old_dw, lb->pXLabel, idx, patch,
			Ghost::None);
	    t->requires(old_dw, lb->pExternalForceLabel, idx, patch,
			Ghost::None);

	    if(mpm_matl->getFractureModel()) {
	       t->requires(new_dw, lb->pVisibilityLabel, idx, patch,
			Ghost::None);
	       t->requires(old_dw, lb->pCrackSurfaceContactForceLabel,
						idx, patch, Ghost::None);
	       t->requires(old_dw, lb->pImageVelocityLabel, idx, patch,
			Ghost::None);
   	    }

	    t->requires(old_dw, d_sharedState->get_delt_label() );
						
	    t->requires(old_dw, lb->pMassLabel, idx, patch, Ghost::None);
	    t->computes(new_dw, lb->pVelocityAfterUpdateLabel, idx, patch );
	    t->computes(new_dw, lb->pXLabel_preReloc, idx, patch );
	    t->computes(new_dw, lb->pExternalForceLabel_preReloc,
							idx, patch);

	    t->requires(old_dw, lb->pParticleIDLabel, idx, patch,
							Ghost::None);
	    t->computes(new_dw, lb->pParticleIDLabel_preReloc,idx, patch);

            t->requires(old_dw, lb->pTemperatureLabel, idx, patch,
			Ghost::None);
            t->requires(new_dw, lb->gTemperatureRateLabel, idx, patch,
			Ghost::AroundCells, 1);
            t->requires(new_dw, lb->gTemperatureLabel, idx, patch,
			Ghost::AroundCells, 1);

            t->requires(new_dw, lb->gTemperatureStarLabel, idx, patch,
			Ghost::AroundCells, 1);
            t->computes(new_dw, lb->pTemperatureRateLabel_preReloc,
							idx, patch);
            t->computes(new_dw, lb->pTemperatureLabel_preReloc,
							idx, patch);
            t->computes(new_dw, lb->pTemperatureGradientLabel_preReloc,
							idx, patch);
	 }

	 t->computes(new_dw, lb->KineticEnergyLabel);
	 t->computes(new_dw, lb->CenterOfMassPositionLabel);
	 t->computes(new_dw, lb->CenterOfMassVelocityLabel);
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
                            this, &SerialMPM::computeMassRate);
         for(int m = 0; m < numMatls; m++){
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
			    this,&SerialMPM::crackGrow);

	 for(int m = 0; m < numMatls; m++) {
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    if(mpm_matl->getFractureModel()) {
	      t->requires( new_dw, lb->pStressAfterStrainRateLabel,
			idx, patch, Ghost::None);
	      t->requires( old_dw, lb->pIsBrokenLabel, idx, patch,
			 Ghost::None);
	      t->requires( old_dw, lb->pTensileStrengthLabel, idx, patch,
			 Ghost::None);
			 
	      t->computes( new_dw, lb->pIsNewlyBrokenLabel, idx, patch );
	      t->computes( new_dw, lb->pNewlyBrokenSurfaceNormalLabel,
			idx, patch );
	      t->computes( new_dw, lb->pTensileStrengthLabel_preReloc,
			idx, patch );
	    }
	 }

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
			    this,&SerialMPM::stressRelease);

	 for(int m = 0; m < numMatls; m++) {
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    if(mpm_matl->getFractureModel()) {
	      t->requires( old_dw, lb->pXLabel, 
			idx, patch, Ghost::AroundNodes, 1 );
	      t->requires( new_dw, lb->pIsNewlyBrokenLabel,
			idx, patch, Ghost::AroundNodes, 1 );
	      t->requires( new_dw, lb->pNewlyBrokenSurfaceNormalLabel,
			idx, patch, Ghost::AroundNodes, 1 );
	      t->requires( new_dw, lb->pStressAfterStrainRateLabel,
			idx, patch, Ghost::AroundNodes, 1 );

	      t->requires(old_dw, d_sharedState->get_delt_label() );

	      //explosive fracture type
	      t->requires( old_dw, lb->pImageVelocityLabel,
			idx, patch, Ghost::None);

	      //normal fracture type
	      t->requires( old_dw, lb->pIsBrokenLabel, idx, patch,
			 Ghost::None);
	      t->requires( old_dw, lb->pCrackSurfaceNormalLabel,
			idx, patch, Ghost::None);
	      t->requires( new_dw, lb->pRotationRateLabel, idx, patch,
			 Ghost::None);
			 
	      t->requires( old_dw, lb->pMassLabel,
			idx, patch, Ghost::None);
	      t->requires( new_dw, lb->pStrainEnergyLabel,
			idx, patch, Ghost::None);
	      t->requires( new_dw, lb->pVelocityAfterUpdateLabel,
			idx, patch, Ghost::None);
			 

 	      t->requires(new_dw, lb->delTAfterConstitutiveModelLabel );

	      t->computes( new_dw, lb->pStressReleasedLabel,
			idx, patch );
	      t->computes( new_dw, lb->pStressAfterFractureReleaseLabel,
			idx, patch );
	      t->computes( new_dw, lb->pImageVelocityLabel_preReloc,
			idx, patch );
	      t->computes( new_dw, lb->pVelocityAfterFractureLabel,
			idx, patch );
	      t->computes( new_dw, lb->pIsBrokenLabel_preReloc,
			 idx, patch);
	      t->computes( new_dw, lb->pCrackSurfaceNormalLabel_preReloc,
			 idx, patch);
	    }
	 }
         t->computes(new_dw, lb->delTAfterFractureLabel);
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
	    this,&SerialMPM::computeCrackSurfaceContactForce);
	 for(int m = 0; m < numMatls; m++){
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    if(mpm_matl->getFractureModel()) {
  	       mpm_matl->getConstitutiveModel()->
	         addComputesAndRequiresForCrackSurfaceContact(t,mpm_matl,
		    patch,old_dw,new_dw);
   	    }
	 }
         t->computes(new_dw, lb->delTAfterCrackSurfaceContactLabel);
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
			    this,&SerialMPM::carryForwardVariables);

	 for(int m = 0; m < numMatls; m++) {
	    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	    int idx = mpm_matl->getDWIndex();
	    
	    //stress
            if(d_fracture) {
	    //if(mpm_matl->getFractureModel()) {
	      t->requires( new_dw, lb->pStressAfterFractureReleaseLabel,
			idx, patch, Ghost::None);
	      t->requires( new_dw, lb->delTAfterCrackSurfaceContactLabel);
	    }
	    else {
	      t->requires( new_dw, lb->pStressAfterStrainRateLabel,
			idx, patch, Ghost::None);			 
	      t->requires( new_dw, lb->delTAfterConstitutiveModelLabel);
            }

	    //velocity
            if(d_fracture) {
	    //if(mpm_matl->getFractureModel()) {
	      t->requires( new_dw, lb->pVelocityAfterFractureLabel,
			idx, patch, Ghost::None);
	    }
	    else {
	      t->requires( new_dw, lb->pVelocityAfterUpdateLabel,
			idx, patch, Ghost::None);			 
            }
	    t->computes(new_dw, lb->pStressLabel_preReloc, idx, patch);
	    t->computes(new_dw, lb->pVelocityLabel_preReloc, idx, patch);
	 }
	 //t->computes(new_dw, d_sharedState->get_delt_label() );
	 t->computes(new_dw, lb->delTLabel );
	 sched->addTask(t);
      }

#if 0
      if(t + dt >= d_nextOutputTime)
      {
	 Task *t = scinew Task("SerialMPM::interpolateParticlesForSaving",
                         patch, old_dw, new_dw,
                         this, &SerialMPM::interpolateParticlesForSaving);
         for(int m = 0; m < numMatls; m++){
            MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
//            MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
            if(mpm_matl){
	      Material* matl = d_sharedState->getMaterial(m);
	      int idx = matl->getDWIndex();
              // Add "requires" for each of the particle variables
              // to be interpolated, as well as the weighting variables.
	      // Add "computes" for the resulting grid variables.
	      t->requires(new_dw, lb->pStressLabel_preReloc, idx, patch,
			Ghost::AroundNodes, 1 );
	      t->requires(new_dw, lb->pMassLabel_preReloc, idx, patch,
			Ghost::AroundNodes, 1 );
	      t->requires(old_dw, lb->pXLabel, idx, patch,
			Ghost::AroundNodes, 1 );

	      t->computes(new_dw, lb->gStressForSavingLabel, idx, patch );
            }
         }
         sched->addTask(t);

      }
#endif
    }
    
    if(t + dt >= d_nextOutputTime)
    {
	d_nextOutputTime += d_outputInterval;
    }

   sched->scheduleParticleRelocation(level, old_dw, new_dw,
				     lb->pXLabel_preReloc, 
				     lb->d_particleState_preReloc,
				     lb->pXLabel, lb->d_particleState,
				     numMatls);

   new_dw->pleaseSave(lb->pXLabel, numMatls);
//   new_dw->pleaseSave(lb->pVelocityLabel, numMatls);
   new_dw->pleaseSave(lb->pVolumeLabel, numMatls);
//   new_dw->pleaseSave(lb->pMassLabel, numMatls);
   new_dw->pleaseSave(lb->pStressLabel, numMatls);

//   new_dw->pleaseSave(lb->gAccelerationLabel, numMatls);
//   new_dw->pleaseSave(lb->gInternalForceLabel, numMatls);
   new_dw->pleaseSave(lb->gMassLabel, numMatls);
//   new_dw->pleaseSave(lb->gVelocityLabel, numMatls);

//   new_dw->pleaseSave(lb->pTemperatureLabel, numMatls);
//   new_dw->pleaseSave(lb->pTemperatureGradientLabel, numMatls);

//   new_dw->pleaseSave(lb->gTemperatureLabel, numMatls);

   // Add pleaseSaves here for each of the grid variables
   // created by interpolateParticlesForSaving
   new_dw->pleaseSave(lb->gStressForSavingLabel, numMatls);

   if(d_fracture) {
     new_dw->pleaseSave(lb->pCrackSurfaceNormalLabel, numMatls);
     new_dw->pleaseSave(lb->pIsBrokenLabel, numMatls);
   }

   new_dw->pleaseSaveIntegrated(lb->StrainEnergyLabel);
   new_dw->pleaseSaveIntegrated(lb->KineticEnergyLabel);
   new_dw->pleaseSaveIntegrated(lb->TotalMassLabel);
   new_dw->pleaseSaveIntegrated(lb->CenterOfMassPositionLabel);
   new_dw->pleaseSaveIntegrated(lb->CenterOfMassVelocityLabel);
}

void SerialMPM::interpolateParticlesForSaving(const ProcessorGroup*,
				    	      const Patch* patch,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{
   int numMatls = d_sharedState->getNumMatls();

   vector<const VarLabel* > vars;
   vector<const VarLabel* > varweights;
   vector<const VarLabel* > gvars;

   // Add items to each of the three vectors here.
   // vars is the particle data to be interpolated
   // varweights is the particle data by which to weight
   // the interpolation, and gvars is the resulting
   // interpolated version of that variable.
   vars.push_back(lb->pStressLabel_preReloc);
   varweights.push_back(lb->pMassLabel_preReloc);
   gvars.push_back(lb->gStressForSavingLabel);

   for(int i=0;i<(int)vars.size();i++){
     for(int m = 0; m < numMatls; m++){
        Material* matl = d_sharedState->getMaterial( m );
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
        if(mpm_matl){
          int matlindex = matl->getDWIndex();
          int vfindex = matl->getVFIndex();
          ParticleSubset* pset = old_dw->getParticleSubset(matlindex,
			patch, Ghost::AroundNodes, 1, lb->pXLabel);

          // Allocate storage & retrieve particle weighting and position
          ParticleVariable<double> weighting;
          new_dw->get(weighting, varweights[i], pset);
          NCVariable<double> gweight;
          new_dw->allocate(gweight, lb->gWeightLabel, vfindex, patch);
          ParticleVariable<Point> px;
          old_dw->get(px, lb->pXLabel, pset);


          if (vars[i]->typeDescription()->getSubType()->getType()
			 == TypeDescription::Vector) {
              NCVariable<Vector> gdata;
              ParticleVariable<Vector> pdata;
              new_dw->allocate(gdata, gvars[i], vfindex, patch);
              new_dw->get(pdata, vars[i], pset);
              gdata.initialize(Vector(0,0,0));
              // Do interpolation
	      for(ParticleSubset::iterator iter = pset->begin();
		  iter != pset->end(); iter++){
		 particleIndex idx = *iter;

		 // Get the node indices that surround the cell
		 IntVector ni[8];  double S[8];
	 
		 if(!patch->findCellAndWeights(px[idx], ni, S))
		    throw InternalError("Particle not in patch");

		 // Add each particles contribution
		 for(int k = 0; k < 8; k++) {
		    if(patch->containsNode(ni[k])){
		       gdata[ni[k]]  += pdata[idx] * weighting[idx] *S[k];
		       gweight[ni[k]]+= weighting[idx] *S[k];
		    }
		 }
	      }
	      for(NodeIterator iter = patch->getNodeIterator();
					  !iter.done(); iter++){
		 if(gweight[*iter] >= 1.e-10){
		    gdata[*iter] *= 1./gweight[*iter];
		 }
	      }
              new_dw->put(gdata, gvars[i], vfindex, patch);
          }
          else if (vars[i]->typeDescription()->getSubType()->getType()
			== TypeDescription::Matrix3) {
              NCVariable<Matrix3> gdata;
              ParticleVariable<Matrix3> pdata;
	      new_dw->allocate(gdata, gvars[i], vfindex, patch);
	      new_dw->get(pdata, vars[i], pset);
              gdata.initialize(Matrix3(0.));
              // Do interpolation
	      for(ParticleSubset::iterator iter = pset->begin();
		  iter != pset->end(); iter++){
		 particleIndex idx = *iter;

		 // Get the node indices that surround the cell
		 IntVector ni[8];  double S[8];
	 
		 if(!patch->findCellAndWeights(px[idx], ni, S))
		    throw InternalError("Particle not in patch");

		 // Add each particles contribution
		 for(int k = 0; k < 8; k++) {
		    if(patch->containsNode(ni[k])){
		       gdata[ni[k]]   += pdata[idx] * weighting[idx]*S[k];
		       gweight[ni[k]] += weighting[idx] * S[k];
		    }
		 }
	      }
	      for(NodeIterator iter = patch->getNodeIterator();
					  !iter.done(); iter++){
		 if(gweight[*iter] >= 1.e-10){
		    gdata[*iter] *= 1./gweight[*iter];
		 }
	      }

              new_dw->put(gdata, gvars[i], vfindex, patch);
          }
          else if (vars[i]->typeDescription()->getSubType()->getType()
			== TypeDescription::double_type) {
              NCVariable<double> gdata;
              ParticleVariable<double> pdata;
	      new_dw->allocate(gdata, gvars[i], vfindex, patch);
	      new_dw->get(pdata, vars[i], pset);
              gdata.initialize(0.);
              // Do interpolation
	      for(ParticleSubset::iterator iter = pset->begin();
		  iter != pset->end(); iter++){
		 particleIndex idx = *iter;

		 // Get the node indices that surround the cell
		 IntVector ni[8];  double S[8];
	 
		 if(!patch->findCellAndWeights(px[idx], ni, S))
		    throw InternalError("Particle not in patch");

		 // Add each particles contribution
		 for(int k = 0; k < 8; k++) {
		    if(patch->containsNode(ni[k])){
		       gdata[ni[k]]   += pdata[idx] * weighting[idx]*S[k];
		       gweight[ni[k]] += weighting[idx] * S[k];
		    }
		 }
	      }
	      for(NodeIterator iter = patch->getNodeIterator();
					  !iter.done(); iter++){
		 if(gweight[*iter] >= 1.e-10){
		    gdata[*iter] *= 1./gweight[*iter];
		 }
	      }

              new_dw->put(gdata, gvars[i], vfindex, patch);
          }
        } // if mpm_matl
     }  // for matl's
   }
}

void SerialMPM::actuallyInitialize(const ProcessorGroup*,
				   const Patch* patch,
				   DataWarehouseP& /* old_dw */,
				   DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();

  PerPatch<long> NAPID(0);
  if(new_dw->exists(lb->ppNAPIDLabel, 0, patch))
      new_dw->get(NAPID,lb->ppNAPIDLabel, 0, patch);

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
       particleIndex numParticles = mpm_matl->countParticles(patch);

       mpm_matl->createParticles(numParticles, NAPID, patch, new_dw);

       NAPID=NAPID + numParticles;

       mpm_matl->getConstitutiveModel()->initializeCMData(patch,
						mpm_matl, new_dw);
       mpm_matl->getBurnModel()->initializeBurnModelData(patch,
						mpm_matl, new_dw);
       if(mpm_matl->getFractureModel()) {
         d_fracture = true;
	 mpm_matl->getFractureModel()->initializeFractureModelData( patch,
						mpm_matl, new_dw);
       }       

       int dwindex = mpm_matl->getDWIndex();

       MPMPhysicalModules::contactModel->
			initializeContact(patch,dwindex,new_dw);
  }

  new_dw->put(NAPID, lb->ppNAPIDLabel, 0, patch);
}


void SerialMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
					      const Patch*,
					      DataWarehouseP&,
					      DataWarehouseP&)
{
}

void SerialMPM::computeNodeVisibility(
                   const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    mpm_matl->getFractureModel()->computeNodeVisibility(
	  patch, mpm_matl, old_dw, new_dw);
  }
}

void SerialMPM::computeCrackSurfaceContactForce(
                   const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    if(mpm_matl->getFractureModel()) {
        mpm_matl->getConstitutiveModel()->computeCrackSurfaceContactForce(
	  patch, mpm_matl, old_dw, new_dw);
    }
  }
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorGroup*,
					   const Patch* patch,
					   DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle data
      ParticleVariable<Point> px;
      ParticleVariable<double> pmass;
      ParticleVariable<Vector> pvelocity;
      ParticleVariable<Vector> pexternalforce;
      ParticleVariable<double> pTemperature;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);
      old_dw->get(px,             lb->pXLabel, pset);
      old_dw->get(pmass,          lb->pMassLabel, pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel, pset);
      old_dw->get(pexternalforce, lb->pExternalForceLabel, pset);
      old_dw->get(pTemperature,   lb->pTemperatureLabel, pset);
      
      ParticleVariable<int> pVisibility;
      ParticleVariable<Vector> pCrackSurfaceContactForce;
      if(mpm_matl->getFractureModel()) {
        new_dw->get(pVisibility, lb->pVisibilityLabel, pset);
	old_dw->get(pCrackSurfaceContactForce,
			lb->pCrackSurfaceContactForceLabel, pset);
      }

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> gexternalforce;
      NCVariable<double> gTemperature;

      new_dw->allocate(gmass,         lb->gMassLabel, matlindex, patch);
      new_dw->allocate(gvelocity,     lb->gVelocityLabel, matlindex, patch);
      new_dw->allocate(gexternalforce, lb->gExternalForceLabel,
						matlindex, patch);
      new_dw->allocate(gTemperature, lb->gTemperatureLabel,
						matlindex, patch);

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity

      gmass.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);

      double totalmass = 0;
      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	 particleIndex idx = *iter;

	 // Get the node indices that surround the cell
	 IntVector ni[8];
	 double S[8];

  	 if(!patch->findCellAndWeights(px[idx], ni, S))
	      throw InternalError("Particle not in patch");

         Visibility vis;
         if(mpm_matl->getFractureModel()) {
  	   vis = pVisibility[idx];
      	   vis.modifyWeights(S);
   	 }
	 
	 // Add each particles contribution to the local mass & velocity 
	 // Must use the node indices
	 for(int k = 0; k < 8; k++) {
	    if(patch->containsNode(ni[k]) && vis.visible(k) ) {
	       gmass[ni[k]] += pmass[idx] * S[k];
	       gvelocity[ni[k]] += pvelocity[idx] * pmass[idx] * S[k];
	       
	       gexternalforce[ni[k]] += pexternalforce[idx] * S[k];
	       if(mpm_matl->getFractureModel()) {
  	         gexternalforce[ni[k]] += pCrackSurfaceContactForce[idx]
							 * S[k];
	       }
	       
	       totalmass += pmass[idx] * S[k];
	       
               gTemperature[ni[k]] += pTemperature[idx]*pmass[idx] * S[k];
	    }
	 }
      }
      for(NodeIterator iter = patch->getNodeIterator();
				!iter.done(); iter++) {
	 if(gmass[*iter] >= 1.e-10){
	    gvelocity[*iter] *= 1./gmass[*iter];
            gTemperature[*iter] /= gmass[*iter];
	 }
      }

      // Apply grid boundary conditions to the velocity
      // before storing the data

      //      cout << "Patch id = " << patch->getID() << endl;
      for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
	vector<BoundCondBase* > bcs;
	bcs = patch->getBCValues(face);
	//cout << "number of bcs on face " << face << " = " 
	//	     << bcs.size() << endl;

	for (int i = 0; i<(int)bcs.size(); i++ ) {
	  string bcs_type = bcs[i]->getType();
	  if (bcs_type == "Velocity") {
	    VelocityBoundCond* bc = 
	      dynamic_cast<VelocityBoundCond*>(bcs[i]);
	    //	    cout << "bc value = " << bc->getValue() << endl;
	    gvelocity.fillFace(face,bc->getValue());
	  }
	  if (bcs_type == "Symmetric") {
	     gvelocity.fillFaceNormal(face);
	  }
	  if (bcs_type == "Temperature") {
	    // TemperatureBoundCond* bc = 
	    //  dynamic_cast<TemperatureBoundCond*>(bcs[i]);
	    // if (bc->getKind() == "Dirichlet")
	    //cout << "bc value = " << bc->getValue() << endl;
	    // if (bc->getKind() == "Neumann")
	  }
	}
      }

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);

      new_dw->put(gmass,         lb->gMassLabel,          matlindex, patch);
      new_dw->put(gvelocity,     lb->gVelocityLabel,      matlindex, patch);
      new_dw->put(gexternalforce,lb->gExternalForceLabel, matlindex, patch);
      new_dw->put(gTemperature,  lb->gTemperatureLabel,   matlindex, patch);
  }
}

void SerialMPM::computeStressTensor(const ProcessorGroup*,
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw)
{
   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->computeStressTensor(patch, mpm_matl, old_dw, new_dw);
   }
}

void SerialMPM::computeMassRate(const ProcessorGroup*,
			 	const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw)
{
   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      HEBurn* heb = mpm_matl->getBurnModel();
      heb->computeMassRate(patch, mpm_matl, old_dw, new_dw);
   }
}

void SerialMPM::crackGrow(const ProcessorGroup*,
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw)
{
   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      if(mpm_matl->getFractureModel()) {
        mpm_matl->getFractureModel()->
			crackGrow(patch, mpm_matl, old_dw, new_dw);
      }
   }
}

void SerialMPM::carryForwardVariables( const ProcessorGroup*,
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw)
{
   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    int matlindex = mpm_matl->getDWIndex();
        
    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);

    //stress
    ParticleVariable<Matrix3> pStress;
    if(mpm_matl->getFractureModel()) {
      new_dw->get(pStress, lb->pStressAfterFractureReleaseLabel, pset);
    }
    else {
      new_dw->get(pStress, lb->pStressAfterStrainRateLabel, pset);
    }

    //stress
    ParticleVariable<Vector> pVelocity;
    if(mpm_matl->getFractureModel()) {
      new_dw->get(pVelocity, lb->pVelocityAfterFractureLabel, pset);
    }
    else {
      new_dw->get(pVelocity, lb->pVelocityAfterUpdateLabel, pset);
    }

    //timestep
    delt_vartype delT;
    if(d_fracture) {
      new_dw->get(delT, lb->delTAfterCrackSurfaceContactLabel);
    }
    else {
      new_dw->get(delT, lb->delTAfterConstitutiveModelLabel);
    }

    new_dw->put(pStress, lb->pStressLabel_preReloc);
    new_dw->put(pVelocity, lb->pVelocityLabel_preReloc);
    new_dw->put(delT, lb->delTLabel);
  }
}
	    
void SerialMPM::stressRelease(const ProcessorGroup*,
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw)
{
   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      if(mpm_matl->getFractureModel()) {
        mpm_matl->getFractureModel()->stressRelease(patch, mpm_matl,
							old_dw, new_dw);
      }
   }
}

void SerialMPM::computeInternalForce(const ProcessorGroup*,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{

  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();

  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle position, volume
      // and the constitutive model
      ParticleVariable<Point>  px;
      ParticleVariable<double>  pvol;
      ParticleVariable<double>  pmass;
      ParticleVariable<Matrix3> pstress;
      NCVariable<Vector>        internalforce;
      NCVariable<Matrix3>       gstress;
      NCVariable<double>        gmass;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);
      old_dw->get(px,      lb->pXLabel, pset);
      old_dw->get(pmass,   lb->pMassLabel, pset);

      new_dw->get(pvol,   lb->pVolumeDeformedLabel, pset);
      new_dw->get(gmass,  lb->gMassLabel, matlindex, patch, Ghost::None, 0);
      new_dw->get(pstress,lb->pStressAfterStrainRateLabel, pset);

      new_dw->allocate(gstress,lb->gStressForSavingLabel, matlindex, patch);
      new_dw->allocate(internalforce, lb->gInternalForceLabel,
						matlindex, patch);
  
      internalforce.initialize(Vector(0,0,0));

      ParticleVariable<int> pVisibility;
      if(mpm_matl->getFractureModel()) {
        new_dw->get(pVisibility, lb->pVisibilityLabel, pset);
      }

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;
  
         // Get the node indices that surround the cell
         IntVector ni[8];
         Vector d_S[8];
         double S[8];

         if(!patch->findCellAndShapeDerivatives(px[idx], ni, d_S))
  	   continue;
         if(!patch->findCellAndWeights(px[idx], ni, S))
           continue;

         Visibility vis;
         if(mpm_matl->getFractureModel()) {
           vis = pVisibility[idx];
           vis.modifyShapeDerivatives(d_S);
         }

         for (int k = 0; k < 8; k++){
	   if(patch->containsNode(ni[k]) && vis.visible(k)){
	     Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
						d_S[k].z()*oodx[2]);
	     internalforce[ni[k]] -= (div * pstress[idx] * pvol[idx]);
             gstress[ni[k]] += pstress[idx] * pmass[idx] * S[k];
	   }
         }
      }
      for(NodeIterator iter = patch->getNodeIterator();
				!iter.done(); iter++) {
         if(gmass[*iter] >= 1.e-10){
            gstress[*iter] /= gmass[*iter];
         }
      }
      new_dw->put(internalforce, lb->gInternalForceLabel, matlindex, patch);
      new_dw->put(gstress, lb->gStressForSavingLabel,     matlindex, patch);
  }
}

void SerialMPM::computeInternalHeatRate(
                                     const ProcessorGroup*,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{
  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      double thermalConductivity = mpm_matl->getThermalConductivity();
      
      ParticleVariable<Point>  px;
      ParticleVariable<double> pvol;
      ParticleVariable<Vector> pTemperatureGradient;
      
      NCVariable<double>       internalHeatRate;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);
      old_dw->get(px,      lb->pXLabel, pset);
      new_dw->get(pvol,    lb->pVolumeDeformedLabel, pset);
      old_dw->get(pTemperatureGradient,
			 lb->pTemperatureGradientLabel, pset);

      new_dw->allocate(internalHeatRate, lb->gInternalHeatRateLabel,
			matlindex, patch);
  
      internalHeatRate.initialize(0.);

      ParticleVariable<int> pVisibility;
      if(mpm_matl->getFractureModel()) {
        new_dw->get(pVisibility, lb->pVisibilityLabel, pset);
      }

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;
  
         // Get the node indices that surround the cell
         IntVector ni[8];
         Vector d_S[8];
         if(!patch->findCellAndShapeDerivatives(px[idx], ni, d_S))
  	   continue;

         Visibility vis;
         if(mpm_matl->getFractureModel()) {
           vis = pVisibility[idx];
           vis.modifyShapeDerivatives(d_S);
         }

         for (int k = 0; k < 8; k++){
	   if(patch->containsNode(ni[k]) && vis.visible(k)){
             Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
						d_S[k].z()*oodx[2]);
	     internalHeatRate[ni[k]] -= Dot( div,
				 pTemperatureGradient[idx] ) * 
	                                pvol[idx] * thermalConductivity;
 	   }
         }
      }
      new_dw->put(internalHeatRate, lb->gInternalHeatRateLabel,
							matlindex, patch);
  }
}


void SerialMPM::solveEquationsMotion(const ProcessorGroup*,
				     const Patch* patch,
				     DataWarehouseP& /*old_dw*/,
				     DataWarehouseP& new_dw)
{
  Vector zero(0.,0.,0.);
  Vector gravity = d_sharedState->getGravity();

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Get required variables for this patch
      NCVariable<double> mass;
      NCVariable<Vector> internalforce;
      NCVariable<Vector> externalforce;

      new_dw->get(mass,  lb->gMassLabel, matlindex, patch, Ghost::None, 0);
      new_dw->get(internalforce, lb->gInternalForceLabel, matlindex, patch,
		  Ghost::None, 0);
      new_dw->get(externalforce, lb->gExternalForceLabel, matlindex, patch,
		  Ghost::None, 0);

      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocate(acceleration, lb->gAccelerationLabel,
						matlindex, patch);

      // Do the computation of a = F/m for nodes where m!=0.0
      for(NodeIterator iter = patch->getNodeIterator();
				!iter.done(); iter++){
	if(mass[*iter]>0.0){
	  acceleration[*iter] =
		(internalforce[*iter] + externalforce[*iter])/ mass[*iter]
		 + gravity;
	}
	else{
	  acceleration[*iter] = zero;
	}
      }

      // Put the result in the datawarehouse
      new_dw->put(acceleration, lb->gAccelerationLabel, matlindex, patch);
  }
}

void SerialMPM::solveHeatEquations(const ProcessorGroup*,
				     const Patch* patch,
				     DataWarehouseP& /*old_dw*/,
				     DataWarehouseP& new_dw)
{
  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      double specificHeat = mpm_matl->getSpecificHeat();
     
      // Get required variables for this patch
      NCVariable<double> mass,internalHeatRate,externalHeatRate;
      NCVariable<double> thermalContactHeatExchangeRate;

      new_dw->get(mass, lb->gMassLabel, dwindex, patch, Ghost::None, 0);
      new_dw->get(internalHeatRate, lb->gInternalHeatRateLabel, 
                  dwindex, patch, Ghost::None, 0);

      if(MPMPhysicalModules::thermalContactModel) {
        new_dw->get(thermalContactHeatExchangeRate,
                  lb->gThermalContactHeatExchangeRateLabel, 
                  dwindex, patch, Ghost::None, 0);
      }
                  
/*
      new_dw->get(externalHeatRate, lb->gExternalHeatRateLabel, 
                  dwindex, patch, Ghost::None, 0);
*/

      // Create variables for the results
      NCVariable<double> temperatureRate;
      new_dw->allocate(temperatureRate, lb->gTemperatureRateLabel,
         dwindex, patch);

      for(NodeIterator iter = patch->getNodeIterator();
					!iter.done(); iter++){
	if(mass[*iter]>0.0){
	  temperatureRate[*iter] = internalHeatRate[*iter]
		   /*+ externalHeatRate[*iter]*/;
	  temperatureRate[*iter] /= (mass[*iter] * specificHeat);
          if(MPMPhysicalModules::thermalContactModel) {
            temperatureRate[*iter]+=thermalContactHeatExchangeRate[*iter];
          }
	}
	else{
	  temperatureRate[*iter] = 0;
	}
      }

      // Put the result in the datawarehouse
      new_dw->put(temperatureRate, lb->gTemperatureRateLabel, dwindex, patch);

  }
}


void SerialMPM::integrateAcceleration(const ProcessorGroup*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{
  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      // Get required variables for this patch
      NCVariable<Vector>        acceleration;
      NCVariable<Vector>        velocity;
      delt_vartype delT;

      new_dw->get(acceleration, lb->gAccelerationLabel, dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(velocity, lb->gMomExedVelocityLabel, dwindex, patch,
		  Ghost::None, 0);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      // Create variables for the results
      NCVariable<Vector> velocity_star;
      new_dw->allocate(velocity_star, lb->gVelocityStarLabel,
						dwindex, patch);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	velocity_star[*iter] = velocity[*iter] + acceleration[*iter]*delT;
      }

      // Put the result in the datawarehouse
      new_dw->put( velocity_star, lb->gVelocityStarLabel, dwindex, patch);
  }
}

void SerialMPM::integrateTemperatureRate(const ProcessorGroup*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{
  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();

      NCVariable<double> temperature;
      NCVariable<double> temperatureRate;
      delt_vartype delT;
 
      new_dw->get(temperature, lb->gTemperatureLabel, dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(temperatureRate, lb->gTemperatureRateLabel,
			dwindex, patch, Ghost::None, 0);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      NCVariable<double> temperatureStar;
      new_dw->allocate(temperatureStar,
		lb->gTemperatureStarLabel, dwindex, patch);

      for(NodeIterator iter = patch->getNodeIterator();
				!iter.done(); iter++){
        temperatureStar[*iter] = temperature[*iter] +
				 temperatureRate[*iter] * delT;
      }

      new_dw->put( temperatureStar, lb->gTemperatureStarLabel,
						dwindex, patch );
  }
}


void SerialMPM::setAnalyze(PatchDataAnalyze* analyze)
{
  d_analyze = analyze;
}

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
						const Patch* patch,
						DataWarehouseP& old_dw,
						DataWarehouseP& new_dw)
{
  // Performs the interpolation from the cell vertices of the grid
  // acceleration and velocity to the particles to update their
  // velocity and position respectively

  Vector dx = patch->dCell();
  double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

  Vector vel(0.0,0.0,0.0);
  Vector acc(0.0,0.0,0.0);
  
  double tempRate;
  
  //  double thermal_energy = 0.0;
  // DON'T MOVE THESE!!!
  Vector CMX(0.0,0.0,0.0);
  Vector CMV(0.0,0.0,0.0);
  double ke=0;
  int numPTotal = 0;

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      ParticleVariable<Point> px;
      ParticleVariable<Point> pxnew;
      ParticleVariable<Vector> pvelocity;
      ParticleVariable<Vector> pvelocitynew;
      ParticleVariable<double> pmass;
      ParticleVariable<Vector> pexternalForce;
      
      ParticleVariable<double> pTemperature;    //for heat conduction
      ParticleVariable<double> pTemperatureNew; //for heat conduction
      ParticleVariable<Vector> pTemperatureGradient; //for heat conduction
      ParticleVariable<double> pTemperatureRate; //for heat conduction
      NCVariable<double> gTemperatureRate; //for heat conduction
      NCVariable<double> gTemperatureStar; //for heat conduction
      NCVariable<double> gTemperature; //for heat conduction

      ParticleSubset* pset = old_dw->getParticleSubset(dwindex, patch);
      old_dw->get(px,        lb->pXLabel, pset);
      new_dw->allocate(pxnew,lb->pXLabel_preReloc, pset);
      old_dw->get(pvelocity, lb->pVelocityLabel, pset);
      new_dw->allocate(pvelocitynew, lb->pVelocityAfterUpdateLabel, pset);
      old_dw->get(pmass,     lb->pMassLabel, pset);
      old_dw->get(pexternalForce, lb->pExternalForceLabel, pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel, pset);
      new_dw->allocate(pTemperatureNew, lb->pTemperatureLabel_preReloc, pset);
      new_dw->allocate(pTemperatureRate,lb->pTemperatureRateLabel, pset);
      new_dw->allocate(pTemperatureGradient,lb->pTemperatureGradientLabel,
			 pset);

      ParticleVariable<int> pVisibility;
      ParticleVariable<Vector> pCrackSurfaceContactForce;
      if(mpm_matl->getFractureModel()) {
        new_dw->get(pVisibility, lb->pVisibilityLabel, pset);
	old_dw->get(pCrackSurfaceContactForce,
		 lb->pCrackSurfaceContactForceLabel, pset);
      }

      // Get the arrays of grid data on which the new part. values depend
      NCVariable<Vector> gvelocity_star;
      NCVariable<Vector> gacceleration;
      delt_vartype delT;

      new_dw->get(gvelocity_star,lb->gMomExedVelocityStarLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gacceleration, lb->gMomExedAccelerationLabel,
			dwindex, patch, Ghost::AroundCells, 1);
		  
      new_dw->get(gTemperatureRate, lb->gTemperatureRateLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gTemperatureStar, lb->gTemperatureStarLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gTemperature, lb->gTemperatureLabel, dwindex, patch,
           Ghost::AroundCells, 1);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles
      for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
	vector<BoundCondBase* > bcs;
	bcs = patch->getBCValues(face);
	//cout << "number of bcs on face " << face << " = " 
	//    << bcs.size() << endl;

	for (int i = 0; i<(int)bcs.size(); i++ ) {
	  string bcs_type = bcs[i]->getType();
	  if (bcs_type == "Velocity") {
	    VelocityBoundCond* bc = 
	      dynamic_cast<VelocityBoundCond*>(bcs[i]);
	    //  cout << "bc value = " << bc->getVelocity() << endl;
	     gvelocity_star.fillFace(face,bc->getValue());
	     gacceleration.fillFace(face,Vector(0.0,0.0,0.0));
	  }
	  if (bcs_type == "Symmetric") {
	     gvelocity_star.fillFaceNormal(face);
	     gacceleration.fillFaceNormal(face);
	  }
	  if (bcs_type == "Temperature") {
	    TemperatureBoundCond* bc = 
	      dynamic_cast<TemperatureBoundCond*>(bcs[i]);
	    if (bc->getKind() == "Dirichlet") {
	      //cout << "bc value = " << bc->getTemp() << endl;
	      IntVector low = gTemperature.getLowIndex();
	      IntVector hi = gTemperature.getHighIndex();
	      gTemperatureStar.fillFace(face,bc->getValue());
	      if(face==Patch::xplus || face==Patch::xminus){
		int I=-1234;
		if(face==Patch::xminus){ I=low.x(); }
		if(face==Patch::xplus){ I=hi.x()-1; }
		for (int j = low.y(); j<hi.y(); j++) { 
		  for (int k = low.z(); k<hi.z(); k++) {
		    gTemperatureRate[IntVector(I,j,k)] +=
		      (gTemperatureStar[IntVector(I,j,k)]-
		       gTemperature[IntVector(I,j,k)])/delT;
		  }
		}
	      }
	      if(face==Patch::yplus || face==Patch::yminus){
		int J=-1234;
		if(face==Patch::yminus){ J=low.y(); }
		if(face==Patch::yplus){ J=hi.y()-1; }
		for (int i = low.x(); i<hi.x(); i++) {
		  for (int k = low.z(); k<hi.z(); k++) {
		    gTemperatureRate[IntVector(i,J,k)] +=
		      (gTemperatureStar[IntVector(i,J,k)]-
		       gTemperature[IntVector(i,J,k)])/delT;
		  }
		}
	      }
	      if(face==Patch::zplus || face==Patch::zminus){
		int K=-1234;
		if(face==Patch::zminus){ K=low.z(); }
		if(face==Patch::zplus){ K=hi.z()-1; }
		for (int i = low.x(); i<hi.x(); i++) {
		  for (int j = low.y(); j<hi.y(); j++) {
		    gTemperatureRate[IntVector(i,j,K)] +=
		      (gTemperatureStar[IntVector(i,j,K)]-
		       gTemperature[IntVector(i,j,K)])/delT;
		  }
		}
	      }
	    }
	    if (bc->getKind() == "Neumann") {
	      //cout << "bc value = " << bc->getValue() << endl;
	    }
	  }
	}
      }
    
      numPTotal += pset->numParticles();

//      double Cp=mpm_matl->getSpecificHeat();

      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
	 particleIndex idx = *iter;

        // Get the node indices that surround the cell
	IntVector ni[8];
        double S[8];
        Vector d_S[8];

        if(!patch->findCellAndWeights(px[idx], ni, S))
	    continue;
        if(!patch->findCellAndShapeDerivatives(px[idx], ni, d_S))
            continue;

        Visibility vis;
	int numVisibleNodes = 0;
        if(mpm_matl->getFractureModel()) {
  	   vis = pVisibility[idx];
	   vis.modifyWeights(S);
	   vis.modifyShapeDerivatives(d_S);
        }

        vel = Vector(0.0,0.0,0.0);
        acc = Vector(0.0,0.0,0.0);

        pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
        tempRate = 0;

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < 8; k++) {
 	   if(vis.visible(k) ) {
  	      ++numVisibleNodes;
	      vel += gvelocity_star[ni[k]]  * S[k];
   	      acc += gacceleration[ni[k]]   * S[k];
	   
              tempRate += gTemperatureRate[ni[k]] * S[k];
              for (int j = 0; j<3; j++) {
                pTemperatureGradient[idx](j) += 
                   gTemperatureStar[ni[k]] * d_S[k](j) * oodx[j];
              }
	   }
        }

        // Update the particle's position and velocity
	if(numVisibleNodes != 0) {
          pxnew[idx]      = px[idx] + vel * delT;
          pvelocitynew[idx] = pvelocity[idx] + acc * delT;
          pTemperatureRate[idx] = tempRate;
          pTemperatureNew[idx] = pTemperature[idx] + tempRate * delT;
//          thermal_energy += pTemperature[idx] * Cp;
        }
	else {        
   	  //for isolated particles in fracture
	  /*
          pxnew[idx]      =  px[idx] + pvelocity[idx] * delT;
          pvelocity[idx] +=
		(pexternalForce[idx] + pCrackSurfaceContactForce[idx])
	     /pmass[idx] * delT;
	     */
        }
	
	
        ke += .5*pmass[idx]*pvelocitynew[idx].length2();
	CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
	CMV += pvelocitynew[idx]*pmass[idx];
      }

      // Store the new result
      new_dw->put(pxnew,        lb->pXLabel_preReloc);
      new_dw->put(pvelocitynew, lb->pVelocityAfterUpdateLabel);
      new_dw->put(pexternalForce, lb->pExternalForceLabel_preReloc);

      ParticleVariable<long> pids;
      old_dw->get(pids, lb->pParticleIDLabel, pset);
      new_dw->put(pids, lb->pParticleIDLabel_preReloc);

      new_dw->put(pTemperatureRate, lb->pTemperatureRateLabel_preReloc);
      new_dw->put(pTemperatureNew, lb->pTemperatureLabel_preReloc);
      new_dw->put(pTemperatureGradient, lb->pTemperatureGradientLabel_preReloc);
  }
  // DON'T MOVE THESE!!!
  new_dw->put(sum_vartype(ke), lb->KineticEnergyLabel);
  new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
  new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);

//   cout << "THERMAL ENERGY " << thermal_energy << endl;

}


// $Log$
// Revision 1.164  2000/11/28 00:17:28  tan
// Fixed a bug caused by unique-variable_lable_name-problem.
//
// Revision 1.163  2000/11/21 20:51:02  tan
// Implemented different models for fracture simulations.  SimpleFracture model
// is for the simulation where the resolution focus only on macroscopic major
// cracks. NormalFracture and ExplosionFracture models are more sophiscated
// and specific fracture models that are currently underconstruction.
//
// Revision 1.162  2000/11/15 18:51:50  guilkey
// Reduced warnings.
//
// Revision 1.161  2000/11/15 01:39:42  guilkey
// Made the way in which materials were looped over more consistent.
// Got rid of references to VFIndex, use only DWIndex now.
//
// Revision 1.160  2000/11/07 22:52:15  guilkey
// Changed the way that materials are looped over.  Instead of each
// function iterating over all materials, and then figuring out which ones
// are MPMMaterials on the fly, SimulationState now stores specific information
// about MPMMaterials, so that for doing MPM, only those materials are returned
// and then looped over.  This will make coupling with a cfd code easier I hope.
//
// Revision 1.159  2000/11/07 04:28:41  jas
// Now doing Symmetry BC in interpolateParticlesToGrid.  Had them commented
// out.
//
// Revision 1.158  2000/11/06 20:50:11  guilkey
// Moved functionality of Problem into SerialMPM.
//
// Revision 1.157  2000/11/02 21:29:22  jas
// Changed the way grid bcs are implemented.
//
// Revision 1.156  2000/10/12 18:37:42  guilkey
// Rec. 21.5 *
//    Never use more than 80 columns per line.
//
// Revision 1.155  2000/09/29 05:32:50  sparker
// Quiet warning from g++
//
// Revision 1.154  2000/09/27 16:47:18  guilkey
// Moved the creation of gStressForSaving from my failed function into
// computeInternalForce.  Lame.
//
// Revision 1.153  2000/09/25 20:23:13  sparker
// Quiet g++ warnings
//
// Revision 1.152  2000/09/22 19:40:52  guilkey
// Added a pxnew to interpolateToParticles... so that the data in
// pXLabel doesn't get changed.
//
// Revision 1.151  2000/09/22 07:13:04  tan
// MPM code works with fracture in three point bending.
//
// Revision 1.150  2000/09/16 04:27:34  tan
// Modifications to make fracture works well.
//
// Revision 1.149  2000/09/15 00:12:40  guilkey
// Recommitting the (hopefully) fixed interpolateParticlesForSaving
// code.
//
// Revision 1.148  2000/09/12 16:52:04  tan
// Reorganized crack surface contact force algorithm.
//
// Revision 1.147  2000/09/11 20:23:19  tan
// Fixed a mistake in crack surface contact force algorithm.
//
// Revision 1.146  2000/09/11 19:45:15  tan
// Implemented crack surface contact force calculation algorithm.
//
// Revision 1.145  2000/09/11 19:01:47  tan
// Crack surface contact force is now considered in the simulation.
//
// Revision 1.144  2000/09/11 17:55:08  guilkey
// Commented out some code that is apparently breaking the 1000grains
// runs.
//
// Revision 1.143  2000/09/11 01:08:37  tan
// Modified time step calculation (in constitutive model computeStressTensor(...))
// when fracture cracking speed involved.
//
// Revision 1.142  2000/09/11 00:14:55  tan
// Added calculations on random distributed microcracks in broken particles.
//
// Revision 1.141  2000/09/10 23:09:57  tan
// Added calculations on the rotation of crack surafce during fracture.
//
// Revision 1.140  2000/09/10 22:51:09  tan
// Added particle rotationRate computation in computeStressTensor functions
// in each constitutive model classes.  The particle rotationRate will be used
// for fracture.
//
// Revision 1.139  2000/09/09 20:39:12  tan
// Modify interpolation weights and shape function derivatives when fracture
// involved.
//
// Revision 1.138  2000/09/09 20:23:19  tan
// Replace BrokenCellShapeFunction with particle visibility information in shape
// function computation.
//
// Revision 1.137  2000/09/09 19:34:11  tan
// Added MPMLabel::pVisibilityLabel and SerialMPM::computerNodesVisibility().
//
// Revision 1.136  2000/09/08 22:54:02  guilkey
// Fixed an error with interpolateParticlesForSaving.
//
// Revision 1.135  2000/09/08 20:27:59  tan
// Added visibility calculation to fracture broken cell shape function
// interpolation.
//
// Revision 1.134  2000/09/08 17:31:28  guilkey
// Added interpolateParticlesForSaving task which interpolates particle
// data, interpolates it to the grid using another particle scalar variable
// for weighting, and saves it to the grid data to the uda.  Note that these
// interpolations only get done when it's time to save data to the uda.
//
// Revision 1.133  2000/09/08 01:48:04  tan
// Added pDilatationalWaveSpeedLabel for fracture and is saved as a
// side-effect of computeStressTensor in each constitutive model class.
//
// Revision 1.132  2000/09/07 21:11:04  tan
// Added particle variable pMicrocrackSize for fracture.
//
// Revision 1.131  2000/09/06 20:41:46  tan
// Fixed bugs on pleasesave for fracture stuff.
//
// Revision 1.130  2000/09/06 20:15:45  tan
// Fixed bugs in crackGrow.
//
// Revision 1.129  2000/09/06 19:35:46  jas
// Changed new to scinew and fixed a memory leak introduced by the fracture
// stuff.
//
// Revision 1.128  2000/09/05 19:37:15  tan
// Fracture starts to run in Uintah/MPM!
//
// Revision 1.127  2000/09/05 08:00:19  tan
// Added crack grow for crack propagation.
//
// Revision 1.126  2000/09/05 07:43:51  tan
// Applied BrokenCellShapeFunction to constitutive models where fracture
// is involved.
//
// Revision 1.125  2000/09/05 07:00:16  tan
// Applied BrokenCellShapeFunction to SerialMPM::interpolateToParticlesAndUpdate
// where fracture is involved.
//
// Revision 1.124  2000/09/05 06:33:43  tan
// Introduced BrokenCellShapeFunction for SerialMPM::interpolateParticlesToGrid
// where farcture is involved.
//
// Revision 1.123  2000/09/05 05:11:31  tan
// Moved Fracture Model to MPMMaterial class.
//
// Revision 1.121  2000/08/30 00:12:42  guilkey
// Added some stuff for interpolating particle data to the grid solely
// for the purpose of saving to an uda.  This doesn't work yet.
//
// Revision 1.120  2000/08/29 21:46:25  guilkey
// Moved the divide by mass stuff in interpolateParticlesToGrid to before
// the BCs are applied.
//
// Revision 1.119  2000/08/24 00:17:09  guilkey
// Put saving and zeroing of integrated quantities in the correct places.
//
// Revision 1.118  2000/08/19 03:20:17  tan
// Fix a fault by my misdeleting the code on boundary conditions in function
// interpolateParticlesToGrid.
//
// Revision 1.117  2000/08/19 01:04:56  guilkey
// Fixed the calculation of kinetic energy for multiple materials.
//
// Revision 1.116  2000/08/18 20:28:59  tan
// Fixed some bugs in SerialMPM, mainly in applyPhysicalBC.
//
// Revision 1.115  2000/08/11 23:16:58  jas
// Added a missing requires for gTemperatureLabel in the interpolateToParticles
// AndUpdate method.  MPI now works for the latest SerialMPM.cc.
//
// Revision 1.114  2000/08/09 03:17:58  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.113  2000/08/07 20:22:32  tan
// Applied force boundary conditions on particles during each simulation step.
//
// Revision 1.112  2000/08/07 17:09:51  tan
// Added applyPhysicalBCToParticles to handle particle boundary conditions
// in each step.
//
// Revision 1.111  2000/08/07 00:37:48  tan
// Added MPMPhysicalBC class to handle all kinds of physical boundary conditions
// in MPM.  Current implemented force boundary conditions.
//
// Revision 1.110  2000/08/04 16:50:56  tan
// The "if(MPMPhysicalModules::heatConductionModel)" is removed.  Simulations
// are thermal-mechanical.
//
// Revision 1.109  2000/08/02 03:28:45  jas
// Fixed grid bc problems.
//
// Revision 1.108  2000/07/28 22:45:13  jas
// particle relocation now uses separate var labels for each material.
// Addd <iostream> for ReductionVariable.  Commented out protected: in
// Scheduler class that preceeded scheduleParticleRelocation.
//
// Revision 1.107  2000/07/27 22:39:43  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.106  2000/07/27 22:17:16  jas
// Consolidated the registerPermanentParticleState to take both the
// regular labels and the pre_Reloc labels.
//
// Revision 1.105  2000/07/27 20:29:50  jas
// In SerialMPM.cc, problemSetup, there are now labels for each material.
// So it is now possible for different materials to have different VarLabels
// depending on the simulation requirements.
//
// Revision 1.104  2000/07/25 22:52:44  guilkey
// Completed John's work with the grid boundary conditions.  Now in addition
// to fishing the boundary information from the problem spec, it actually
// gets used as well.  Currently, specified velocity, symmetry and specified
// temperature boundaries are implemented.
//
// Revision 1.103  2000/07/25 19:10:23  guilkey
// Changed code relating to particle combustion as well as the
// heat conduction.
//
// Revision 1.102  2000/07/20 19:42:25  guilkey
// Made minor changes to the heat conduction algorithm, mainly involving
// the thermal contact.
//
// Revision 1.100  2000/07/17 23:45:03  tan
// Fixed problems in MPM heat conduction.
//
// Revision 1.99  2000/07/12 18:45:07  jas
// Cleaned up the creation of the particle state and moved things into
// MPMLabel.
//
// Revision 1.98  2000/07/05 23:43:30  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.97  2000/06/28 01:08:20  tan
// Thermal contact model start to work!
//
// Revision 1.96  2000/06/27 23:10:53  jas
// Added in grid bcs.
//
// Revision 1.95  2000/06/27 21:50:57  guilkey
// Added saving of more "meta data."  Namely, center of mass position and
// velocity.  These are both actually mass weighted, so should be divided
// by total mass which is being saved in TotalMass.dat.
//
// Revision 1.94  2000/06/26 18:47:13  tan
// Different heat_conduction properties for different materials are allowed
// in the MPM simulation.
//
// Revision 1.93  2000/06/24 04:06:40  tan
// SerialMPM works for heat-conduction now!
//
// Revision 1.92  2000/06/22 22:37:05  tan
// Moved heat conduction physical parameters (thermalConductivity, specificHeat,
// and heatTransferCoefficient) from MPMMaterial class to HeatConduction class.
//
// Revision 1.91  2000/06/22 21:22:36  tan
// MPMPhysicalModules class is created to handle all the physical modules
// in MPM, currently those physical submodules include HeatConduction,
// Fracture, Contact, and ThermalContact.
//
// Revision 1.90  2000/06/20 18:24:06  tan
// Arranged the physical models implemented in MPM.
//
// Revision 1.89  2000/06/20 04:12:55  tan
// WHen d_thermalContactModel != NULL, heat conduction will be included in MPM
// algorithm.  The d_thermalContactModel is set by ThermalContactFactory according
// to the information in ProblemSpec from input file.
//
// Revision 1.88  2000/06/19 23:52:12  guilkey
// Added boolean d_burns so that certain stuff only gets done
// if a burn model is present.  Not to worry, the if's on this
// are not inside of inner loops.
//
// Revision 1.87  2000/06/19 21:22:28  bard
// Moved computes for reduction variables outside of loops over materials.
//
// Revision 1.86  2000/06/17 07:06:33  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.85  2000/06/16 23:23:35  guilkey
// Got rid of pVolumeDeformedLabel_preReloc to fix some confusion
// the scheduler was having.
//
// Revision 1.84  2000/06/15 21:57:00  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.83  2000/06/08 22:23:46  guilkey
// Removed a bunch of wrapped lines.
//
// Revision 1.82  2000/06/08 21:05:03  bigler
// Added support to ouput gMass values (where pleaseSave is used).
//
// Revision 1.81  2000/06/08 16:56:51  guilkey
// Added tasks and VarLabels for HE burn model stuff.
//
// Revision 1.80  2000/06/05 19:48:57  guilkey
// Added Particle IDs.  Also created NAPID (Next Available Particle ID)
// on a per patch basis so that any newly created particles will know where
// the indexing left off.
//
// Revision 1.79  2000/06/03 05:25:44  sparker
// Added a new for pSurfLabel (was uninitialized)
// Uncommented pleaseSaveIntegrated
// Minor cleanups of reduction variable use
// Removed a few warnings
//
// Revision 1.78  2000/06/01 23:12:04  guilkey
// Code to store integrated quantities in the DW and save them in
// an archive of sorts.  Also added the "computes" in the right tasks.
//
// Revision 1.77  2000/05/31 21:20:50  tan
// ThermalContact::computeHeatExchange() linked to scheduleTimeAdvance to handle
// thermal contact.
//
// Revision 1.76  2000/05/31 18:30:21  tan
// Create linkage to ThermalContact model.
//
// Revision 1.75  2000/05/31 17:40:57  tan
// Particle temperature gradient computations included in
// interpolateToParticlesAndUpdate().
//
// Revision 1.74  2000/05/31 16:10:17  tan
// Heat conduction computations included in interpolateParticlesToGrid().
//
// Revision 1.73  2000/05/31 00:34:43  tan
// temp to tempRate
//
// Revision 1.72  2000/05/30 21:03:22  dav
// delt to delT
//
// Revision 1.71  2000/05/30 20:18:58  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.70  2000/05/30 18:17:26  dav
// removed the include varLaberl
//
// Revision 1.69  2000/05/30 17:07:34  dav
// Removed commented out labels.  Other MPI fixes.  Changed delt to delT so I would stop thinking of it as just delta.
//
// Revision 1.68  2000/05/30 04:26:17  tan
// Heat conduction algorithm integrated into scheduleTimeAdvance().
//
// Revision 1.67  2000/05/26 23:07:03  tan
// Rewrite interpolateToParticlesAndUpdate to include heat conduction.
//
// Revision 1.66  2000/05/26 21:37:30  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
// Revision 1.65  2000/05/26 17:14:56  tan
// Added solveHeatEquations on grid.
//
// Revision 1.64  2000/05/26 02:27:48  tan
// Added computeHeatRateGeneratedByInternalHeatFlux() for thermal field
// computation.
//
// Revision 1.63  2000/05/25 23:03:11  guilkey
// Implemented calls to addComputesAndRequires for the Contact
// funtions (addComputesAndRequiresInterpolated and
// addComputesAndRequiresIntegrated)
//
// Revision 1.62  2000/05/25 22:06:34  tan
// A boolean variable d_heatConductionInvolved is set to true when
// heat conduction considered in the simulation.
//
// Revision 1.61  2000/05/23 02:25:45  tan
// Put all velocity-field independent variables on material
// index of 0.
//
// Revision 1.60  2000/05/18 18:50:25  jas
// Now using the gravity from the input file.
//
// Revision 1.59  2000/05/18 16:36:37  guilkey
// Numerous small changes including:
//   1.  Moved carry forward of particle volume to the cons. models.
//   2.  Added more data to the tecplot files (commented out)
//   3.  Computing strain energy on all particles at each time step,
//       in addition to the kinetic energy.
//   4.  Now printing out two files for diagnostics, one with energies
//       and one with center of mass.
//
// Revision 1.58  2000/05/16 00:40:51  guilkey
// Added code to do boundary conditions, print out tecplot files, and a
// few other things.  Most of this is now commented out.
//
// Revision 1.57  2000/05/15 20:03:22  dav
// couple of cleanups
//
// Revision 1.56  2000/05/15 19:39:37  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.55  2000/05/15 19:02:44  tan
// A little change on initializedFracture Interface.
//
// Revision 1.54  2000/05/15 18:59:46  tan
// Initialized NCVariables and CCVaribles for Fracture.
//
// Revision 1.53  2000/05/12 01:45:17  tan
// Added call to initializeFracture in SerialMPM's actuallyInitailize.
//
// Revision 1.52  2000/05/11 20:10:12  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.51  2000/05/10 20:02:42  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.50  2000/05/10 18:34:15  tan
// Added computations on self-contact cells for cracked surfaces.
//
// Revision 1.49  2000/05/10 05:01:33  tan
// linked to farcture model.
//
// Revision 1.48  2000/05/09 23:45:09  jas
// Fixed the application of grid boundary conditions.  It is probably slow
// as mud but hopefully the gist is right.
//
// Revision 1.47  2000/05/09 21:33:02  guilkey
// Added gravity to the acceleration.  Currently hardwired, I need a little
// help seeing how to get it out of the ProblemSpec.
//
// Revision 1.46  2000/05/09 03:27:55  jas
// Using the enums for boundary conditions hack.
//
// Revision 1.45  2000/05/08 18:46:16  guilkey
// Added call to initializeContact in SerialMPM's actuallyInitailize
//
// Revision 1.44  2000/05/08 17:16:51  tan
// Added grid VarLabel selfContactLabel for fracture simulation.
//
// Revision 1.43  2000/05/07 06:02:01  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.42  2000/05/04 23:40:00  tan
// Added fracture interface to general MPM.
//
// Revision 1.41  2000/05/04 19:10:52  guilkey
// Added code to apply boundary conditions.  This currently calls empty
// functions which will be implemented soon.
//
// Revision 1.40  2000/05/04 17:30:32  tan
//   Add surfaceNormal for boundary particle tracking.
//
// Revision 1.39  2000/05/03 23:52:44  guilkey
// Fixed some small errors in the MPM code to make it work
// and give what appear to be correct answers.
//
// Revision 1.38  2000/05/02 18:41:15  guilkey
// Added VarLabels to the MPM algorithm to comply with the
// immutable nature of the DataWarehouse. :)
//
// Revision 1.37  2000/05/02 17:54:21  sparker
// Implemented more of SerialMPM
//
// Revision 1.36  2000/05/02 06:07:08  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.35  2000/05/01 16:18:07  sparker
// Completed more of datawarehouse
// Initial more of MPM data
// Changed constitutive model for bar
//
// Revision 1.34  2000/04/28 21:08:15  jas
// Added exception to the creation of Contact factory if contact is not
// specified.
//
// Revision 1.33  2000/04/28 08:11:29  sparker
// ConstitutiveModelFactory should return null on failure
// MPMMaterial checks for failed constitutive model creation
// DWindex and VFindex are now initialized
// Fixed input file to match ConstitutiveModelFactory
//
// Revision 1.32  2000/04/28 07:35:26  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.31  2000/04/27 23:18:41  sparker
// Added problem initialization for MPM
//
// Revision 1.30  2000/04/27 21:39:27  jas
// Now creating contact via a factory.
//
// Revision 1.29  2000/04/26 06:48:12  sparker
// Streamlined namespaces
//
// Revision 1.28  2000/04/25 22:57:29  guilkey
// Fixed Contact stuff to include VarLabels, SimulationState, etc, and
// made more of it compile.
//
// Revision 1.27  2000/04/25 00:41:19  dav
// more changes to fix compilations
//
// Revision 1.26  2000/04/24 21:04:24  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.25  2000/04/24 15:16:58  sparker
// Fixed unresolved symbols
//
// Revision 1.24  2000/04/21 17:46:52  jas
// Inserted & for function pointers needed by Task constructor (required by
// gcc).  Moved around WONT_COMPILE_YET to get around allocate.
//
// Revision 1.23  2000/04/20 23:20:26  dav
// updates
//
// Revision 1.22  2000/04/20 22:13:36  dav
// making SerialMPM compile
//
// Revision 1.21  2000/04/20 18:56:16  sparker
// Updates to MPM
//
// Revision 1.20  2000/04/19 22:38:16  dav
// Make SerialMPM a UintahParallelComponent
//
// Revision 1.19  2000/04/19 22:30:00  dav
// somehow SerialMPM.cc had cvs diff stuff in it.  I have removed it.  I hope I removed the right stuff.
//
// Revision 1.18  2000/04/19 21:20:01  dav
// more MPI stuff
//
// Revision 1.17  2000/04/19 05:26:01  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.16  2000/04/14 02:16:33  jas
// Added variable to the problemSetup declaration.
//
// Revision 1.15  2000/04/14 02:13:57  jas
// problemSetup runs the preProcessor.
//
// Revision 1.14  2000/04/13 06:50:55  sparker
// More implementation to get this to work
//
// Revision 1.13  2000/04/12 22:59:03  sparker
// Working to make it compile
// Added xerces to link line
//
// Revision 1.12  2000/04/12 16:57:23  guilkey
// Converted the SerialMPM.cc to have multimaterial/multivelocity field
// capabilities.  Tried to guard all the functions against breaking the
// compilation, but then who really cares?  It's not like sus has compiled
// for more than 5 minutes in a row for two months.
//
// Revision 1.11  2000/03/23 20:42:16  sparker
// Added copy ctor to exception classes (for Linux/g++)
// Helped clean up move of ProblemSpec from Interface to Grid
//
// Revision 1.10  2000/03/21 02:13:56  dav
// commented out bad portions of code so I could check in a version that compiles
//
// Revision 1.9  2000/03/21 01:29:39  dav
// working to make MPM stuff compile successfully
//
// Revision 1.8  2000/03/20 17:17:05  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.7  2000/03/17 21:01:50  dav
// namespace mods
//
// Revision 1.6  2000/03/17 02:57:01  dav
// more namespace, cocoon, etc
//
// Revision 1.5  2000/03/16 01:11:23  guilkey
// To timeStep added tasks to do contact.
//
// Revision 1.4  2000/03/15 22:13:04  jas
// Added log and changed header file locations.
//
