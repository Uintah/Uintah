#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h> // 
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/Fracture/Fracture.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/Fracture/Connectivity.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/SymmetryBoundCond.h>
#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/NotFinished.h>

#include <Packages/Uintah/CCA/Components/MPM/Fracture/NormalFracture.h>

#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace SCIRun;

using namespace std;

/*`====================*/ 
//#define DOING    
#undef DOING
/*====================`*/

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

SerialMPM::SerialMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  d_nextOutputTime=0.;
  d_fracture = false;
  d_SMALL_NUM_MPM=1e-200;
  d_with_ice    = false;
  d_with_arches = false;
}

SerialMPM::~SerialMPM()
{
  delete lb;
}

void SerialMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& /*grid*/,
			     SimulationStateP& sharedState)
{
   d_sharedState = sharedState;

   MPMPhysicalBCFactory::create(prob_spec);

   contactModel = ContactFactory::create(prob_spec,sharedState, lb);
   thermalContactModel =
		 ThermalContactFactory::create(prob_spec, sharedState, lb);

   ProblemSpecP p = prob_spec->findBlock("DataArchiver");
   if(!p->get("outputInterval", d_outputInterval))
      d_outputInterval = 1.0;

   //Search for the MaterialProperties block and then get the MPM section

   ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");

   ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");

   for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
     MPMMaterial *mat = scinew MPMMaterial(ps, lb);
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

     if(mpm_matl->getFractureModel()){
       d_fracture = true;
       lb->registerPermanentParticleState(m,lb->pCrackNormalLabel,
			lb->pCrackNormalLabel_preReloc); 
       lb->registerPermanentParticleState(m,lb->pTipNormalLabel,
			lb->pTipNormalLabel_preReloc);
       lb->registerPermanentParticleState(m,lb->pExtensionDirectionLabel,
			lb->pExtensionDirectionLabel_preReloc);
       lb->registerPermanentParticleState(m,lb->pToughnessLabel,
			lb->pToughnessLabel_preReloc); 
       lb->registerPermanentParticleState(m,lb->pIsBrokenLabel,
			lb->pIsBrokenLabel_preReloc); 
       lb->registerPermanentParticleState(m,lb->pCrackSurfacePressureLabel,
			lb->pCrackSurfacePressureLabel_preReloc); 
       lb->registerPermanentParticleState(m,lb->pDisplacementLabel,
			lb->pDisplacementLabel_preReloc); 
     }
     
     lb->registerPermanentParticleState(m,lb->pTemperatureLabel,
					  lb->pTemperatureLabel_preReloc); 
       //lb->registerPermanentParticleState(m,lb->pExternalHeatRateLabel,
       //lb->pExternalHeatRateLabel_preReloc); 

     lb->registerPermanentParticleState(m,lb->pParticleIDLabel,
					lb->pParticleIDLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pMassLabel,
					lb->pMassLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pVolumeLabel,
					lb->pVolumeLabel_preReloc);
     
     mpm_matl->getConstitutiveModel()->addParticleState(lb->d_particleState[m],
					lb->d_particleState_preReloc[m]);
   }
}

void SerialMPM::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched)
{
  Task* t = scinew Task("SerialMPM::actuallyInitialize",
			this, &SerialMPM::actuallyInitialize);
  t->computes(lb->partCountLabel);
  t->computes(lb->pXLabel);
  t->computes(lb->pMassLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pTemperatureGradientLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->doMechLabel);
  if(d_fracture){
    t->computes(lb->pIsBrokenLabel);
    t->computes(lb->pCrackNormalLabel);
    t->computes(lb->pTipNormalLabel);
    t->computes(lb->pExtensionDirectionLabel);
    t->computes(lb->pToughnessLabel);
    t->computes(lb->pCrackSurfacePressureLabel);
    t->computes(lb->pDisplacementLabel);
  }
  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->pCellNAPIDLabel);

  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  t = scinew Task("SerialMPM::printParticleCount",
		  this, &SerialMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
}

void SerialMPM::scheduleComputeStableTimestep(const LevelP&,
					      SchedulerP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
}

void SerialMPM::scheduleTimeAdvance(const LevelP&         level,
				    SchedulerP&     sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  if(d_fracture) {
    scheduleSetPositions(                 sched, patches, matls);
    scheduleComputeBoundaryContact(       sched, patches, matls);
    scheduleComputeConnectivity(          sched, patches, matls);
  }

  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  scheduleComputeHeatExchange(            sched, patches, matls);
  scheduleExMomInterpolated(              sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);
  scheduleComputeInternalForce(           sched, patches, matls);
  scheduleComputeInternalHeatRate(        sched, patches, matls);
  scheduleSolveEquationsMotion(           sched, patches, matls);
  scheduleSolveHeatEquations(             sched, patches, matls);
  scheduleIntegrateAcceleration(          sched, patches, matls);
  // scheduleIntegrateTemperatureRate(    sched, patches, matls);
  scheduleExMomIntegrated(                sched, patches, matls);
  scheduleSetGridBoundaryConditions(      sched, patches, matls);
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);

  if(d_fracture) {
    scheduleComputeFracture(              sched, patches, matls);
    scheduleComputeCrackExtension(        sched, patches, matls);
  }
  scheduleCarryForwardVariables(          sched, patches, matls);

  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc, 
				    lb->d_particleState_preReloc,
				    lb->pXLabel, lb->d_particleState,
				    lb->pParticleIDLabel, matls);
}

void SerialMPM::scheduleSetPositions(SchedulerP& sched,
				     const PatchSet* patches,
				     const MaterialSet* matls)
{
  Task* t = scinew Task( "SerialMPM::setPositions",
			  this,&SerialMPM::setPositions);
  t->requires(Task::OldDW, lb->pXLabel, Ghost::None);
  t->computes(lb->pXXLabel);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeBoundaryContact(SchedulerP& sched,
					       const PatchSet* patches,
					       const MaterialSet* matls)
{
 /*
  * computeBoundaryContact
  *   in(P.X, P.VOLUME, P.NEWISBROKEN, P.NEWCRACKSURFACENORMAL)
  *   operation(computeBoundaryContact)
  * out(P.STRESS) */

  Task* t = scinew Task( "SerialMPM::computeBoundaryContact",
			  this,&SerialMPM::computeBoundaryContact);

  t->requires(Task::OldDW, lb->pXLabel,           Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pVolumeLabel,      Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pIsBrokenLabel,    Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormalLabel, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pVelocityLabel,    Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pMassLabel,        Ghost::AroundCells, 1);

  t->requires(Task::NewDW, lb->pXXLabel, Ghost::None);

  t->computes(lb->pContactForceLabel);
  t->computes(lb->pCrackEffectiveLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeConnectivity(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{
 /*
  * computeConnectivity
  *   in(P.X, P.VOLUME, P.ISBROKEN, P.CRACKSURFACENORMAL)
  *   operation(compute the visibility information of particles to the
  *   related nodes)
  * out(P.VISIBILITY) */

  Task* t = scinew Task( "SerialMPM::computeConnectivity",
			  this,&SerialMPM::computeConnectivity);

  t->requires(Task::OldDW, lb->pXLabel, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pVolumeLabel, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormalLabel, Ghost::AroundCells, 1);
  t->requires(Task::NewDW, lb->pCrackEffectiveLabel, Ghost::AroundCells, 1);

  t->requires(Task::NewDW, lb->pXXLabel, Ghost::None);

  t->computes(lb->pConnectivityLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
						   const PatchSet* patches,
						   const MaterialSet* matls)
{
  /* interpolateParticlesToGrid
   *   in(P.MASS, P.VELOCITY, P.NAT_X)
   *   operation(interpolate the P.MASS and P.VEL to the grid
   *             using P.NAT_X and some shape function evaluations)
   *   out(G.MASS, G.VELOCITY) */


  Task* t = scinew Task("SerialMPM::interpolateParticlesToGrid",
			this,&SerialMPM::interpolateParticlesToGrid);
  t->requires(Task::OldDW, lb->pMassLabel,           Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pVolumeLabel,       Ghost::AroundNodes,1);
  
  t->requires(Task::OldDW, lb->pVelocityLabel,     Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pXLabel,            Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pExternalForceLabel,Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pTemperatureLabel,  Ghost::AroundNodes,1);
//    t->requires(Task::OldDW, lb->pExternalHeatRateLabel,
//						Ghost::AroundNodes,1);

  t->computes(lb->gMassLabel);
  t->computes(lb->gMassLabel,        d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);
  t->computes(lb->gTemperatureLabel, d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureNoBCLabel);
  t->computes(lb->gExternalHeatRateLabel);

  if(d_fracture) {
    t->requires(Task::NewDW,lb->pContactForceLabel, Ghost::AroundNodes, 1 );
    t->requires(Task::NewDW, lb->pConnectivityLabel, Ghost::AroundNodes, 1 );

    //for explosive fracture
    t->requires(Task::OldDW, lb->pCrackNormalLabel,  Ghost::AroundNodes, 1 );
    t->requires(Task::OldDW, lb->pCrackSurfacePressureLabel,
      Ghost::AroundNodes, 1 );
  }
     
  t->computes(lb->TotalMassLabel);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeHeatExchange(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{
 /* computeHeatExchange
  *   in(G.MASS, G.TEMPERATURE, G.EXTERNAL_HEAT_RATE)
  *   operation(peform heat exchange which will cause each of
  *   velocity fields to exchange heat according to 
  *   the temperature differences)
  *   out(G.EXTERNAL_HEAT_RATE) */


  Task* t = scinew Task("ThermalContact::computeHeatExchange",
		        thermalContactModel,
		        &ThermalContact::computeHeatExchange);

  thermalContactModel->addComputesAndRequires(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleExMomInterpolated(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* t = scinew Task("Contact::exMomInterpolated",
		    contactModel,
		    &Contact::exMomInterpolated);

  contactModel->addComputesAndRequiresInterpolated(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeStressTensor(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("SerialMPM::computeStressTensor",
		    this, &SerialMPM::computeStressTensor);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches);
  }
	 
  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->StrainEnergyLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeInternalForce(SchedulerP& sched,
					     const PatchSet* patches,
					     const MaterialSet* matls)
{
 /*
  * computeInternalForce
  *   in(P.CONMOD, P.NAT_X, P.VOLUME)
  *   operation(evaluate the divergence of the stress (stored in
  *   P.CONMOD) using P.NAT_X and the gradients of the
  *   shape functions)
  * out(G.F_INTERNAL) */

  Task* t = scinew Task("SerialMPM::computeInternalForce",
		    this, &SerialMPM::computeInternalForce);

  t->requires(Task::NewDW,lb->gMassLabel, Ghost::None);
  t->requires(Task::NewDW,lb->gMassLabel, d_sharedState->getAllInOneMatl(),
						Task::OutOfDomain, Ghost::None);
  t->requires(Task::NewDW,lb->pStressLabel_afterStrainRate,Ghost::AroundNodes,1);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,       Ghost::AroundNodes,1);
  t->requires(Task::OldDW,lb->pXLabel,                 Ghost::AroundNodes,1);
  t->requires(Task::OldDW,lb->pMassLabel,                 Ghost::AroundNodes,1);

  if(d_with_ice){
    t->requires(Task::NewDW, lb->pPressureLabel,          Ghost::AroundNodes,1);
  }

  if(d_fracture) {
    t->requires(Task::NewDW, lb->pConnectivityLabel, Ghost::AroundNodes, 1 );
  }

  t->computes(lb->gInternalForceLabel);
  t->computes(lb->NTractionZMinusLabel);
  t->computes(lb->gStressForSavingLabel);
  t->computes(lb->gStressForSavingLabel, d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeInternalHeatRate(SchedulerP& sched,
						const PatchSet* patches,
						const MaterialSet* matls)
{  
  /*
   * computeInternalHeatRate
   * out(G.INTERNALHEATRATE) */

  Task* t = scinew Task("SerialMPM::computeInternalHeatRate",
			this, &SerialMPM::computeInternalHeatRate);

  t->requires(Task::OldDW, lb->pXLabel,              Ghost::AroundNodes, 1);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel, Ghost::AroundNodes, 1);
  t->requires(Task::NewDW, lb->gTemperatureLabel,    Ghost::AroundCells, 2);

  if(d_fracture) {
    t->requires(Task::NewDW,lb->pConnectivityLabel,Ghost::AroundNodes,1);
  }

  t->computes(lb->gInternalHeatRateLabel);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleSolveEquationsMotion(SchedulerP& sched,
					     const PatchSet* patches,
					     const MaterialSet* matls)
{
  /* solveEquationsMotion
   *   in(G.MASS, G.F_INTERNAL)
   *   operation(acceleration = f/m)
   *   out(G.ACCELERATION) */

  Task* t = scinew Task("SerialMPM::solveEquationsMotion",
		    this, &SerialMPM::solveEquationsMotion);

  t->requires(Task::OldDW, d_sharedState->get_delt_label());

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
      
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->requires(Task::OldDW, lb->doMechLabel);
  t->computes(lb->doMechLabel);

  if(d_with_ice){
    t->requires(Task::NewDW, lb->gradPAccNCLabel,   Ghost::None);
  }
  if(d_with_arches){
    t->requires(Task::NewDW, lb->AccArchesNCLabel,  Ghost::None);
  }

  t->computes(lb->gAccelerationLabel);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleSolveHeatEquations(SchedulerP& sched,
					   const PatchSet* patches,
					   const MaterialSet* matls)
{
  /* solveHeatEquations
   *   in(G.MASS, G.INTERNALHEATRATE, G.EXTERNALHEATRATE)
   *   out(G.TEMPERATURERATE) */

  Task* t = scinew Task("SerialMPM::solveHeatEquations",
			    this, &SerialMPM::solveHeatEquations);

  t->requires(Task::NewDW, lb->gMassLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,           Ghost::None);
  t->requires(Task::NewDW, lb->gExternalHeatRateLabel, Ghost::None);

  t->requires(Task::NewDW, lb->gThermalContactHeatExchangeRateLabel,
						       Ghost::None);
		
  // It isn't really modifying for anybody, but it is more efficient
  // to modify than to copy to a temporary variable.
  //t->modifies(lb->gInternalHeatRateLabel);
  t->requires(Task::NewDW, lb->gInternalHeatRateLabel, Ghost::None);

  t->computes(lb->gTemperatureRateLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleIntegrateAcceleration(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls)
{
  /* integrateAcceleration
   *   in(G.ACCELERATION, G.VELOCITY)
   *   operation(v* = v + a*dt)
   *   out(G.VELOCITY_STAR) */

  Task* t = scinew Task("SerialMPM::integrateAcceleration",
			    this, &SerialMPM::integrateAcceleration);

  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  //t->modifies(             lb->gAccelerationLabel, mss);
  t->requires(Task::NewDW, lb->gAccelerationLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,          Ghost::None);

  t->computes(lb->gVelocityStarLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleIntegrateTemperatureRate(SchedulerP& sched,
						 const PatchSet* patches,
						 const MaterialSet* matls)
{
  /* integrateTemperatureRate
   *   in(G.TEMPERATURE, G.TEMPERATURERATE)
   *   operation(t* = t + t_rate * dt)
   *   out(G.TEMPERATURE_STAR) */

  Task* t = scinew Task("SerialMPM::integrateTemperatureRate",
		    this, &SerialMPM::integrateTemperatureRate);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gTemperatureLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel, Ghost::None);
		     
  t->computes(lb->gTemperatureStarLabel);
		     
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleExMomIntegrated(SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* matls)
{
  /* exMomIntegrated
   *   in(G.MASS, G.VELOCITY_STAR, G.ACCELERATION)
   *   operation(peform operations which will cause each of
   *		  velocity fields to feel the influence of the
   *		  the others according to specific rules)
   *   out(G.VELOCITY_STAR, G.ACCELERATION) */

  Task* t = scinew Task("Contact::exMomIntegrated",
		   contactModel,
		   &Contact::exMomIntegrated);

  contactModel->addComputesAndRequiresIntegrated(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleSetGridBoundaryConditions(SchedulerP& sched,
						       const PatchSet* patches,
						       const MaterialSet* matls)

{
  Task* t=scinew Task("SerialMPM::setGridBoundaryConditions",
		    this, &SerialMPM::setGridBoundaryConditions);
                  
  const MaterialSubset* mss = matls->getUnion();
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->modifies(             lb->gTemperatureRateLabel,  mss);
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel,  Ghost::AroundCells,1);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
						       const PatchSet* patches,
						       const MaterialSet* matls)

{
 /*
  * interpolateToParticlesAndUpdate
  *   in(G.ACCELERATION, G.VELOCITY_STAR, P.NAT_X)
  *   operation(interpolate acceleration and v* to particles and
  *   integrate these to get new particle velocity and position)
  * out(P.VELOCITY, P.X, P.NAT_X) */

  Task* t=scinew Task("SerialMPM::interpolateToParticlesAndUpdate",
		    this, &SerialMPM::interpolateToParticlesAndUpdate);


  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gAccelerationLabel,     Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,     Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,  Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->gTemperatureLabel,      Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel,  Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->frictionalWorkLabel,    Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pXLabel,                Ghost::None);
  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      Ghost::None);
  t->requires(Task::OldDW, lb->pVelocityLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pVolumeLabel,           Ghost::None);
  //t->requires(Task::NewDW, lb->pVolumeDeformedLabel,   Ghost::None);

  if(d_with_ice){
    t->requires(Task::NewDW, lb->dTdt_NCLabel,            Ghost::AroundCells,1);
    t->requires(Task::NewDW, lb->massBurnFractionLabel,   Ghost::AroundCells,1);
  }

  if(d_fracture) {
    t->requires(Task::NewDW, lb->pConnectivityLabel,      Ghost::None);
    t->requires(Task::OldDW, lb->pDisplacementLabel,      Ghost::None);
    t->computes(lb->pDisplacementLabel_preReloc);
  }

  t->computes(lb->pVelocityLabel_afterUpdate);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pExternalForceLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);

  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->CenterOfMassVelocityLabel);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeFracture(SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* matls)
{
 /*
  * computeFracture
  *   in(P.X, P.VOLUME, P.ISBROKEN, P.CRACKSURFACENORMAL)
  *   operation(compute the visibility information of particles to the
  *   related nodes)
  * out(P.VISIBILITY) */

  Task* t = scinew Task( "SerialMPM::computeFracture",
			  this,&SerialMPM::computeFracture);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::OldDW, lb->pXLabel,                   Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pVolumeLabel,              Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pTipNormalLabel,           Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pExtensionDirectionLabel,  Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->pStressLabel_afterStrainRate,
    Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pToughnessLabel,           Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->pStrainEnergyLabel,        Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pMassLabel,                Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pIsBrokenLabel,            Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pCrackNormalLabel,         Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pDisplacementLabel,        Ghost::AroundCells,1);

  t->requires(Task::NewDW, lb->pXXLabel,                    Ghost::None);
  t->requires(Task::NewDW, lb->pRotationRateLabel,          Ghost::None);
  t->requires(Task::OldDW, lb->pCrackSurfacePressureLabel,  Ghost::None);
  t->requires(Task::NewDW, lb->pVelocityLabel_afterUpdate,  Ghost::None);

  t->computes(lb->pToughnessLabel_preReloc);
  t->computes(lb->pIsBrokenLabel_preReloc);
  t->computes(lb->pCrackNormalLabel_preReloc);
  t->computes(lb->pStressLabel_afterFracture);
  t->computes(lb->pCrackSurfacePressureLabel_preReloc);
  t->computes(lb->pVelocityLabel_afterFracture);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeCrackExtension(SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* matls)
{
 /*
  * scheduleComputeCrackExtension
  *   in(P.X, P.VOLUME, P.ISBROKEN, P.CRACKSURFACENORMAL)
  *   operation(compute the visibility information of particles to the
  *   related nodes)
  * out(P.VISIBILITY) */

  Task* t = scinew Task( "SerialMPM::computeCrackExtension",
			  this,&SerialMPM::computeCrackExtension);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::OldDW, lb->pXLabel,                   Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->pIsBrokenLabel_preReloc,   Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->pCrackNormalLabel_preReloc,Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pExtensionDirectionLabel,  Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pVolumeLabel,              Ghost::AroundCells,1);

  t->requires(Task::NewDW, lb->pXXLabel,                  Ghost::None);
  t->requires(Task::OldDW, lb->pTipNormalLabel,           Ghost::None);
  t->requires(Task::NewDW, lb->pRotationRateLabel,        Ghost::None);

  t->computes(lb->pTipNormalLabel_preReloc);
  t->computes(lb->pExtensionDirectionLabel_preReloc);
  
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleCarryForwardVariables(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls)
{
  /* carryForwardVariables
   * in(p.x,p.stressBeforeFractureRelease,p.isNewlyBroken,
   *   p.crackSurfaceNormal)
   *   operation(check the stress on each particle to see
   *   if the microcrack will initiate and/or grow)
   * out(p.stress) */

  Task* t = scinew Task("SerialMPM::carryForwardVariables",
		         this,&SerialMPM::carryForwardVariables);

  if(d_fracture) {
    t->requires(Task::NewDW, lb->pStressLabel_afterFracture,
		Ghost::None);
    t->requires(Task::NewDW, lb->pVelocityLabel_afterFracture,
		Ghost::None);
  }
  else {
    t->requires(Task::NewDW, lb->pStressLabel_afterStrainRate,
		Ghost::None);			 
    t->requires(Task::NewDW, lb->pVelocityLabel_afterUpdate,
		Ghost::None);
  }
  
  t->computes(lb->pStressLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  sched->addTask(t, patches, matls);
}

void SerialMPM::printParticleCount(const ProcessorGroup* pg,
				   const PatchSubset*,
				   const MaterialSubset*,
				   DataWarehouse*,
				   DataWarehouse* new_dw)
{
  if(pg->myrank() == 0){
    static bool printed=false;
    if(!printed){
      sumlong_vartype pcount;
      new_dw->get(pcount, lb->partCountLabel);
      cerr << "Created " << pcount << " total particles\n";
      printed=true;
    }
  }
}

void SerialMPM::actuallyInitialize(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse*,
				   DataWarehouse* new_dw)
{
  particleIndex totalParticles=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing actuallyInitialize on patch " << patch->getID()
         <<"\t\t\t MPM"<< endl;
  #endif
    CCVariable<short int> cellNAPID;
    new_dw->allocate(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
    cellNAPID.initialize(0);
    
    for(int m=0;m<matls->size();m++){
      //cerrLock.lock();
      //NOT_FINISHED("not quite right - mapping of matls, use matls->get()");
      //cerrLock.unlock();
       MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
       particleIndex numParticles = mpm_matl->countParticles(patch);
       totalParticles+=numParticles;

       mpm_matl->createParticles(numParticles, cellNAPID, patch, new_dw);

       mpm_matl->getConstitutiveModel()->initializeCMData(patch,
						mpm_matl, new_dw);
       if(mpm_matl->getFractureModel()) {
	 mpm_matl->getFractureModel()->initializeFractureModelData( patch,
						mpm_matl, new_dw);
       }       

       int dwindex = mpm_matl->getDWIndex();

       contactModel->initializeContact(patch,dwindex,new_dw);
    }
    new_dw->put(cellNAPID, lb->pCellNAPIDLabel, 0, patch);

    double doMech = -999.9;
    new_dw->put(delt_vartype(doMech), lb->doMechLabel);
  }
  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);

}


void SerialMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
					      const PatchSubset*,
					      const MaterialSubset*,
					      DataWarehouse*,
					      DataWarehouse*)
{
}

void SerialMPM::computeConnectivity(
                   const ProcessorGroup*,
		   const PatchSubset* patches,
		   const MaterialSubset*,
		   DataWarehouse* old_dw,
		   DataWarehouse* new_dw)
{
#ifdef DOING
  cout <<"computeConnectivity " <<"\t\t\t\t MPM"<< endl;
#endif
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    mpm_matl->getFractureModel()->computeConnectivity(
	  patches, mpm_matl, old_dw, new_dw);
  }
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorGroup*,
					   const PatchSubset* patches,
					   const MaterialSubset* ,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing interpolateParticlesToGrid on patch " << patch->getID()
         <<"\t\t MPM"<< endl;
  #endif
    int numMatls = d_sharedState->getNumMPMMatls();

    NCVariable<double> gmassglobal,gtempglobal;
    new_dw->allocate(gmassglobal,lb->gMassLabel,
		     d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocate(gtempglobal,lb->gTemperatureLabel,
		     d_sharedState->getAllInOneMatl()->get(0), patch);
    gmassglobal.initialize(d_SMALL_NUM_MPM);
    gtempglobal.initialize(0.0);

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      constParticleVariable<Vector> pvelocity, pexternalforce;
      ParticleVariable<double> pexternalheatrate;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      old_dw->get(pexternalforce, lb->pExternalForceLabel, pset);
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);

      new_dw->allocate(pexternalheatrate, lb->pExternalHeatRateLabel, pset);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> gexternalforce;
      NCVariable<double> gexternalheatrate;
      NCVariable<double> gTemperature;
      NCVariable<double> gTemperatureNoBC;

      new_dw->allocate(gmass,            lb->gMassLabel,      matlindex, patch);
      new_dw->allocate(gvolume,          lb->gVolumeLabel,    matlindex, patch);
      new_dw->allocate(gvelocity,        lb->gVelocityLabel,  matlindex, patch);
      new_dw->allocate(gTemperature,   lb->gTemperatureLabel, matlindex, patch);
      new_dw->allocate(gTemperatureNoBC, lb->gTemperatureNoBCLabel,
							      matlindex, patch);
      new_dw->allocate(gexternalforce,lb->gExternalForceLabel,matlindex, patch);
      new_dw->allocate(gexternalheatrate, lb->gExternalHeatRateLabel,
							      matlindex, patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);
      gTemperatureNoBC.initialize(0);
      gexternalheatrate.initialize(0);

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity
      
      double totalmass = 0;
      Vector total_mom(0.0,0.0,0.0);

#if 0
      for(ParticleSubset::iterator iter = pset->begin();
                iter != pset->end(); iter++){
        particleIndex idx = *iter;

        if(px[idx].z() < -0.0475 && fabs(px[idx].x()) < .03
                                 && fabs(px[idx].y()) < .03){
          pexternalheatrate[idx]=20.0;
        }
        else{
          pexternalheatrate[idx]=0.0;
        }
      }
#endif

      if(mpm_matl->getFractureModel()) {  // Do interpolation with fracture

        constParticleVariable<int> pConnectivity;
        constParticleVariable<Vector> pContactForce;
        constParticleVariable<Vector> pCrackNormal;
	constParticleVariable<double> pCrackSurfacePressure;

        new_dw->get(pConnectivity, lb->pConnectivityLabel, pset);
        new_dw->get(pContactForce, lb->pContactForceLabel, pset);
	
	//for explosive fracture
	old_dw->get(pCrackNormal, lb->pCrackNormalLabel, pset);
	old_dw->get(pCrackSurfacePressure, lb->pCrackSurfacePressureLabel,pset);


        for(ParticleSubset::iterator iter = pset->begin();
		iter != pset->end(); iter++){
	  particleIndex idx = *iter;

	  // Get the node indices that surround the cell
	  IntVector ni[8];
	  double S[8];

  	  patch->findCellAndWeights(px[idx], ni, S);

          Connectivity connectivity(pConnectivity[idx]);
  	  int conn[8];
	  connectivity.getInfo(conn);
      	  Connectivity::modifyWeights(conn,S,Connectivity::connect);

          for(int k = 0; k < 8; k++) {
	    if( patch->containsNode(ni[k]) ) {
	      if( conn[k] ) {
	        gmassglobal[ni[k]]    += pmass[idx]          * S[k];
	        gmass[ni[k]]          += pmass[idx]          * S[k];
	        gvolume[ni[k]]        += pvolume[idx]        * S[k];
	        gTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
	        gtempglobal[ni[k]]    += pTemperature[idx] * pmass[idx] * S[k];
                gexternalheatrate[ni[k]] += pexternalheatrate[idx]      * S[k];

	        totalmass += pmass[idx] * S[k];

	        double area = M_PI * pow(0.75/M_PI*pvolume[idx],0.6667);
	        gexternalforce[ni[k]] += 
		  ( pexternalforce[idx] + pContactForce[idx] - 
		    pCrackNormal[idx] * (pCrackSurfacePressure[idx] * area)
		  ) * S[k];
	        gvelocity[ni[k]]     += pvelocity[idx] *pmass[idx]*S[k];
	      }
	    }
	  }
        }
      }
      else {  // Do interpolation without fracture
        for(ParticleSubset::iterator iter = pset->begin();
						iter != pset->end(); iter++){
	  particleIndex idx = *iter;

	  // Get the node indices that surround the cell
	  IntVector ni[8];
	  double S[8];

  	  patch->findCellAndWeights(px[idx], ni, S);

          total_mom += pvelocity[idx]*pmass[idx];

	  // Add each particles contribution to the local mass & velocity 
	  // Must use the node indices
	  for(int k = 0; k < 8; k++) {
	    if(patch->containsNode(ni[k])) {
	       gmassglobal[ni[k]]    += pmass[idx]          * S[k];
	       gmass[ni[k]]          += pmass[idx]          * S[k];
	       gvolume[ni[k]]        += pvolume[idx]        * S[k];
	       gexternalforce[ni[k]] += pexternalforce[idx] * S[k];
	       gvelocity[ni[k]]      += pvelocity[idx]    * pmass[idx] * S[k];
	       gTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
	       gtempglobal[ni[k]]    += pTemperature[idx] * pmass[idx] * S[k];
               gexternalheatrate[ni[k]] += pexternalheatrate[idx]      * S[k];

	       totalmass += pmass[idx] * S[k];
	    }
	  }
        }
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
	gvelocity[*iter] /= gmass[*iter];
        gTemperatureNoBC[*iter] = gTemperature[*iter]/gmass[*iter];
        gTemperature[*iter] /= gmass[*iter];
      }

      // Apply grid boundary conditions to the velocity before storing the data
      IntVector offset =  IntVector(0,0,0);

      for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
        BoundCondBase *vel_bcs, *temp_bcs, *sym_bcs;
        if (patch->getBCType(face) == Patch::None) {
	   vel_bcs  = patch->getBCValues(matlindex,"Velocity",face);
	   temp_bcs = patch->getBCValues(matlindex,"Temperature",face);
	   sym_bcs  = patch->getBCValues(matlindex,"Symmetric",face);
        } else
          continue;

	  if (vel_bcs != 0) {
	    VelocityBoundCond* bc = dynamic_cast<VelocityBoundCond*>(vel_bcs);
	    if (bc->getKind() == "Dirichlet") {
	      //cout << "Velocity bc value = " << bc->getValue() << endl;
	      gvelocity.fillFace(patch, face,bc->getValue(),offset);
	    }
	  }
	  if (sym_bcs != 0) {
	     gvelocity.fillFaceNormal(patch, face,offset);
	  }
	  if (temp_bcs != 0) {
            TemperatureBoundCond* bc =
	      dynamic_cast<TemperatureBoundCond*>(temp_bcs);
            if (bc->getKind() == "Dirichlet") {
              gTemperature.fillFace(patch, face,bc->getValue(),offset);
	    }
	  }
      }

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);

      new_dw->put(gmass,         lb->gMassLabel,          matlindex, patch);
      new_dw->put(gvolume,       lb->gVolumeLabel,        matlindex, patch);
      new_dw->put(gvelocity,     lb->gVelocityLabel,      matlindex, patch);
      new_dw->put(gexternalforce,lb->gExternalForceLabel, matlindex, patch);
      new_dw->put(gTemperature,  lb->gTemperatureLabel,   matlindex, patch);
      new_dw->put(gTemperatureNoBC,  lb->gTemperatureNoBCLabel,
							  matlindex, patch);
      new_dw->put(gexternalheatrate,lb->gExternalHeatRateLabel,
							  matlindex, patch);

    }  // End loop over materials

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        gtempglobal[*iter] /= gmassglobal[*iter];
    }
    new_dw->put(gmassglobal, lb->gMassLabel,
			d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->put(gtempglobal, lb->gTemperatureLabel,
			d_sharedState->getAllInOneMatl()->get(0), patch);
  }  // End loop over patches
}

void SerialMPM::computeStressTensor(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* ,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw)
{
 #ifdef DOING
   cout <<"Doint computeStressTensor " <<"\t\t\t\t MPM"<< endl;
 #endif
   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);
   }
}

void SerialMPM::setPositions( const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset*,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing setPositions " << patch->getID()
         <<"\t\t\t\t MPM"<< endl;
  #endif
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      int matlindex = mpm_matl->getDWIndex();
        
      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);

      constParticleVariable<Point> pX;
      ParticleVariable<Point> pXX;
    
      old_dw->get(pX, lb->pXLabel, pset);
      new_dw->allocate(pXX, lb->pXXLabel, pset);
    
      for(ParticleSubset::iterator iter=pset->begin();iter!=pset->end();iter++){
        pXX[*iter] = pX[*iter];
      }

      new_dw->put(pXX, lb->pXXLabel);
    }
  }
}
	    
void SerialMPM::computeInternalForce(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* ,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing computeInternalForce on patch " << patch->getID()
         <<"\t\t\t MPM"<< endl;
  #endif
    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    Matrix3 Id;
    Id.Identity();

    int numMPMMatls = d_sharedState->getNumMPMMatls();

    double integralTraction = 0.;
    double integralArea = 0.;

    NCVariable<Matrix3>       gstressglobal;
    constNCVariable<double>        gmassglobal;
    new_dw->get(gmassglobal,  lb->gMassLabel,
		d_sharedState->getAllInOneMatl()->get(0), patch, Ghost::None,0);
    new_dw->allocate(gstressglobal,lb->gStressForSavingLabel, 
		     d_sharedState->getAllInOneMatl()->get(0), patch);

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle position, volume
      // and the constitutive model
      constParticleVariable<Point>   px;
      constParticleVariable<double>  pvol, pmass;
      constParticleVariable<Matrix3> pstress;
      NCVariable<Vector>        internalforce;
      NCVariable<Matrix3>       gstress;
      constNCVariable<double>        gmass;

      constParticleVariable<double>  p_pressure;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel, pset);
      old_dw->get(pmass,   lb->pMassLabel, pset);
      new_dw->get(pvol,    lb->pVolumeDeformedLabel, pset);
      new_dw->get(pstress, lb->pStressLabel_afterStrainRate, pset);
      new_dw->get(gmass,   lb->gMassLabel, matlindex, patch, Ghost::None, 0);

      new_dw->allocate(gstress,lb->gStressForSavingLabel, matlindex, patch);
      new_dw->allocate(internalforce, lb->gInternalForceLabel,
						matlindex, patch);

      if(d_with_ice){
        new_dw->get(p_pressure,lb->pPressureLabel, pset);
      }
      else {
	ParticleVariable<double>  p_pressure_create;
	new_dw->allocate(p_pressure_create,lb->pPressureLabel, pset);
	for(ParticleSubset::iterator iter = pset->begin();
                                     iter != pset->end(); iter++){
	   p_pressure_create[*iter]=0.0;
	}
	p_pressure = p_pressure_create; // reference created data
      }

      internalforce.initialize(Vector(0,0,0));

      if(mpm_matl->getFractureModel()) {
        constParticleVariable<int>    pConnectivity;
        new_dw->get(pConnectivity,  lb->pConnectivityLabel,  pset);

        for(ParticleSubset::iterator iter = pset->begin();
		iter != pset->end(); iter++){
           particleIndex idx = *iter;
  
           // Get the node indices that surround the cell
           IntVector ni[8];

           Vector d_S[8];
	   double S[8];

           patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S);

           Connectivity connectivity(pConnectivity[idx]);
  	   int conn[8];
	   connectivity.getInfo(conn);
	 
           Connectivity::modifyShapeDerivatives(conn,d_S,Connectivity::connect);
      	   Connectivity::modifyWeights(conn,S,Connectivity::connect);

           for(int k = 0; k < 8; k++) {
	     if( patch->containsNode(ni[k]) ) {
	       if( conn[k] ) {
                 gstress[ni[k]] += pstress[idx] * pmass[idx] * S[k];
                 gstressglobal[ni[k]] += pstress[idx] * pmass[idx]*S[k];
	         Vector div(d_S[k].x()*oodx[0],
	                    d_S[k].y()*oodx[1],
			    d_S[k].z()*oodx[2]);
	         internalforce[ni[k]] -=
		     (div * (pstress[idx] - Id*p_pressure[idx]) * pvol[idx]);
	       }
	     }
	   }
        }
      }
      else {
        for(ParticleSubset::iterator iter = pset->begin();
						iter != pset->end(); iter++){
          particleIndex idx = *iter;
  
          // Get the node indices that surround the cell
          IntVector ni[8];
          Vector d_S[8];
          double S[8];

          patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S);

          for (int k = 0; k < 8; k++){
	    if(patch->containsNode(ni[k])){
	       Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
						d_S[k].z()*oodx[2]);
	       internalforce[ni[k]] -=
			(div * (pstress[idx] - Id*p_pressure[idx]) * pvol[idx]);
               gstress[ni[k]] += pstress[idx] * pmass[idx] * S[k];
               gstressglobal[ni[k]] += pstress[idx] * pmass[idx] * S[k];
	     }
          }
      }
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        gstress[*iter] /= gmass[*iter];
    }

    IntVector low = patch-> getInteriorNodeLowIndex();
    IntVector hi  = patch-> getInteriorNodeHighIndex();
    for(Patch::FaceType face = Patch::startFace;
        face <= Patch::endFace; face=Patch::nextFace(face)){

        // I assume we have the patch variable
        // Check if the face is on the boundary
         Patch::BCType bc_type = patch->getBCType(face);
         if (bc_type == Patch::None) {
           // We are on the boundary, i.e. not on an interior patch
           // boundary, so do the traction accumulation . . .
           if(face==Patch::zminus){
             int K=low.z();
             for (int i = low.x(); i<hi.x(); i++) {
              for (int j = low.y(); j<hi.y(); j++) {
		integralTraction +=
			gstress[IntVector(i,j,K)](3,3)*dx.x()*dx.y();
		if(fabs(gstress[IntVector(i,j,K)](3,3)) > 1.e-12){
		  integralArea+=dx.x()*dx.y();
                }
              }
             }
	   }
        } // end of if (bc_type == Patch::None)
    }

    new_dw->put(internalforce, lb->gInternalForceLabel,   matlindex, patch);
    new_dw->put(gstress,       lb->gStressForSavingLabel, matlindex, patch);
  }
  if(integralArea > 0.){
    integralTraction=integralTraction/integralArea;
  }
  else{
    integralTraction=0.;
  }
  new_dw->put(sum_vartype(integralTraction), lb->NTractionZMinusLabel);

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
    gstressglobal[*iter] /= gmassglobal[*iter];
  }
  new_dw->put(gstressglobal,  lb->gStressForSavingLabel, 
	      d_sharedState->getAllInOneMatl()->get(0), patch);
  }
}

void SerialMPM::computeInternalHeatRate(const ProcessorGroup*,
				        const PatchSubset* patches,
					const MaterialSubset* ,
				        DataWarehouse* old_dw,
				        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing computeInternalHeatRate on patch " << patch->getID()
         <<"\t\t MPM"<< endl;
  #endif
    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      double thermalConductivity = mpm_matl->getThermalConductivity();
      
      constParticleVariable<Point>  px;
      constParticleVariable<double> pvol;
      ParticleVariable<Vector> pTemperatureGradient;
      constNCVariable<double>       gTemperature;
      NCVariable<double>       internalHeatRate;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);

      old_dw->get(px,           lb->pXLabel,              pset);
      new_dw->get(pvol,         lb->pVolumeDeformedLabel, pset);
      new_dw->get(gTemperature, lb->gTemperatureLabel,    matlindex, patch,
						Ghost::AroundCells, 2);

      new_dw->allocate(internalHeatRate, lb->gInternalHeatRateLabel,
			matlindex, patch);
      new_dw->allocate(pTemperatureGradient,lb->pTemperatureGradientLabel,pset);
  
      internalHeatRate.initialize(0.);

      // First compute the temperature gradient at each particle
      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;

         // Get the node indices that surround the cell
         IntVector ni[8];
         Vector d_S[8];

         patch->findCellAndShapeDerivatives(px[idx], ni, d_S);

	 pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
         for (int k = 0; k < 8; k++){
             for (int j = 0; j<3; j++) {
               pTemperatureGradient[idx](j) += 
                    gTemperature[ni[k]] * d_S[k](j) * oodx[j];
             }
         }
      }


      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;
  
         // Get the node indices that surround the cell
         IntVector ni[8];
         Vector d_S[8];
         patch->findCellAndShapeDerivatives(px[idx], ni, d_S);

         for (int k = 0; k < 8; k++){
	   if(patch->containsNode(ni[k])){
             Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
						d_S[k].z()*oodx[2]);
	     internalHeatRate[ni[k]] -= Dot( div, pTemperatureGradient[idx]) * 
	                                pvol[idx] * thermalConductivity;
	   }
         }
      }

      new_dw->put(internalHeatRate, lb->gInternalHeatRateLabel,matlindex,patch);

    }  // End of loop over materials
  }  // End of loop over patches
}


void SerialMPM::solveEquationsMotion(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset*,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing solveEquationsMotion on patch " << patch->getID()
         <<"\t\t\t MPM"<< endl;
  #endif
    Vector gravity = d_sharedState->getGravity();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label() );
    delt_vartype doMechOld;
    old_dw->get(doMechOld, lb->doMechLabel);

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Get required variables for this patch
      constNCVariable<Vector> internalforce;
      constNCVariable<Vector> externalforce;
      constNCVariable<Vector> gradPAccNC;  // for MPMICE
      constNCVariable<Vector> AccArchesNC;  // for MPMArches

      new_dw->get(internalforce, lb->gInternalForceLabel, matlindex, patch,
							   Ghost::None, 0);
      new_dw->get(externalforce, lb->gExternalForceLabel, matlindex, patch,
							   Ghost::None, 0);

      constNCVariable<double> mass;
      new_dw->get(mass, lb->gMassLabel,       matlindex,patch, Ghost::None,0);

      if(d_with_ice){
         new_dw->get(gradPAccNC,lb->gradPAccNCLabel,    matlindex, patch,
							   Ghost::None, 0);
      }
      else{
  	 NCVariable<Vector> gradPAccNC_create;
	 new_dw->allocate(gradPAccNC_create, lb->gradPAccNCLabel,
			  matlindex, patch);
	 gradPAccNC_create.initialize(Vector(0.,0.,0.));
	 gradPAccNC = gradPAccNC_create; // reference created data
      }
      if(d_with_arches){
         new_dw->get(AccArchesNC,lb->AccArchesNCLabel,    matlindex, patch,
							   Ghost::None, 0);
      }
      else{
  	 NCVariable<Vector> AccArchesNC_create;
	 new_dw->allocate(AccArchesNC_create, lb->AccArchesNCLabel,
			  matlindex, patch);
	 AccArchesNC_create.initialize(Vector(0.,0.,0.));
	 AccArchesNC = AccArchesNC_create; // reference created data	 
      }

      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocate(acceleration, lb->gAccelerationLabel, matlindex, patch);
      acceleration.initialize(Vector(0.,0.,0.));

      if(doMechOld < -1.5){
       // Do the computation of a = F/m for nodes where m!=0.0
       // You need if(mass>small_num) so you don't get pressure
       // acceleration where there isn't any mass. 3.30.01 
       for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
         acceleration[*iter] =
		(internalforce[*iter] + externalforce[*iter])/mass[*iter] +
                 gravity + gradPAccNC[*iter] + AccArchesNC[*iter];
       }
      }
   
      // Put the result in the datawarehouse
      new_dw->put(acceleration, lb->gAccelerationLabel, matlindex, patch);
    }
    new_dw->put(doMechOld, lb->doMechLabel);
  }
}

void SerialMPM::solveHeatEquations(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* ,
				   DataWarehouse* /*old_dw*/,
				   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing solveHeatEquations on patch " << patch->getID()
         <<"\t\t\t MPM"<< endl;
  #endif
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      double specificHeat = mpm_matl->getSpecificHeat();
     
      // Get required variables for this patch
      constNCVariable<double> mass,externalHeatRate,gvolume;
      constNCVariable<double> thermalContactHeatExchangeRate;
      NCVariable<double> internalHeatRate;
      
      new_dw->get(mass, lb->gMassLabel,       dwindex, patch, Ghost::None, 0);
      new_dw->get(gvolume, lb->gVolumeLabel,    dwindex, patch, Ghost::None, 0);
      new_dw->getCopy(internalHeatRate, lb->gInternalHeatRateLabel,
		      dwindex, patch, Ghost::None, 0);
      new_dw->get(externalHeatRate, lb->gExternalHeatRateLabel,
		  dwindex, patch, Ghost::None, 0);

      new_dw->get(thermalContactHeatExchangeRate,
                  lb->gThermalContactHeatExchangeRateLabel,
                  dwindex, patch, Ghost::None, 0);

      Vector dx = patch->dCell();
      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){
	BoundCondBase* temp_bcs;
        if (patch->getBCType(face) == Patch::None) {
	   temp_bcs = patch->getBCValues(dwindex,"Temperature",face);
        } else
          continue;

	if (temp_bcs != 0) {
            TemperatureBoundCond* bc =
                       dynamic_cast<TemperatureBoundCond*>(temp_bcs);
            if (bc->getKind() == "Neumann"){
	      double value = bc->getValue();

             IntVector low = patch->getInteriorNodeLowIndex();
             IntVector hi  = patch->getInteriorNodeHighIndex();     
              if(face==Patch::xplus || face==Patch::xminus){
                int I;
                if(face==Patch::xminus){ I=low.x(); }
                if(face==Patch::xplus){ I=hi.x()-1; }
                for (int j = low.y(); j<hi.y(); j++) {
                  for (int k = low.z(); k<hi.z(); k++) {
                    internalHeatRate[IntVector(I,j,k)] +=
				value*(2.0*gvolume[IntVector(I,j,k)]/dx.x());
                  }
                }
              }
              if(face==Patch::yplus || face==Patch::yminus){
                int J;
                if(face==Patch::yminus){ J=low.y(); }
                if(face==Patch::yplus){ J=hi.y()-1; }
                for (int i = low.x(); i<hi.x(); i++) {
                  for (int k = low.z(); k<hi.z(); k++) {
                    internalHeatRate[IntVector(i,J,k)] +=
				value*(2.0*gvolume[IntVector(i,J,k)]/dx.y());
                  }
                }
              }
              if(face==Patch::zplus || face==Patch::zminus){
                int K;
                if(face==Patch::zminus){ K=low.z(); }
                if(face==Patch::zplus){ K=hi.z()-1; }
                for (int i = low.x(); i<hi.x(); i++) {
                  for (int j = low.y(); j<hi.y(); j++) {
                    internalHeatRate[IntVector(i,j,K)] +=
				value*(2.0*gvolume[IntVector(i,j,K)]/dx.z());
                  }
                }
              }
            }
          }
        
      }

      // Create variables for the results
      NCVariable<double> temperatureRate;
      new_dw->allocate(temperatureRate,lb->gTemperatureRateLabel,dwindex,patch);
      temperatureRate.initialize(0.0);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	  temperatureRate[*iter] = (internalHeatRate[*iter]
		                 +  externalHeatRate[*iter]) /
				  (mass[*iter] * specificHeat);
            temperatureRate[*iter]+=thermalContactHeatExchangeRate[*iter];
      }

      // Put the result in the datawarehouse
      new_dw->put(temperatureRate, lb->gTemperatureRateLabel, dwindex, patch);
    }
  }
}


void SerialMPM::integrateAcceleration(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing integrateAcceleration on patch " << patch->getID()
         <<"\t\t\t MPM"<< endl;
  #endif
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      // Get required variables for this patch
      constNCVariable<Vector>        acceleration;
      constNCVariable<Vector>        velocity;
      delt_vartype delT;

      new_dw->get(acceleration, lb->gAccelerationLabel,  dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(velocity,     lb->gVelocityLabel,      dwindex, patch,
		  Ghost::None, 0);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      // Create variables for the results
      NCVariable<Vector> velocity_star;
      new_dw->allocate(velocity_star, lb->gVelocityStarLabel, dwindex, patch);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	velocity_star[*iter] = velocity[*iter] + acceleration[*iter] * delT;
      }

      // Put the result in the datawarehouse
      new_dw->put( velocity_star, lb->gVelocityStarLabel, dwindex, patch);
    }
  }
}

void SerialMPM::integrateTemperatureRate(const ProcessorGroup*,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing integrateTemperatureRate on patch " << patch->getID()
         << "\t\t MPM"<< endl;
  #endif
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();

      constNCVariable<double> temperature;
      constNCVariable<double> temperatureRate;
      delt_vartype delT;
 
      new_dw->get(temperature, lb->gTemperatureLabel, dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(temperatureRate, lb->gTemperatureRateLabel,
			dwindex, patch, Ghost::None, 0);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      NCVariable<double> temperatureStar;
      new_dw->allocate(temperatureStar,lb->gTemperatureStarLabel,dwindex,patch);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        temperatureStar[*iter] = temperature[*iter] +
				 temperatureRate[*iter] * delT;
      }

      new_dw->put( temperatureStar, lb->gTemperatureStarLabel, dwindex, patch );
    }
  }
}

void SerialMPM::setGridBoundaryConditions(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset* ,
						DataWarehouse* old_dw,
						DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing setGridBoundaryConditions on patch " << patch->getID()
         <<"\t\t MPM"<< endl;
  #endif
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    
    delt_vartype delT;            
    old_dw->get(delT, d_sharedState->get_delt_label() );
                      
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;
      NCVariable<double> gTemperatureRate;
      constNCVariable<double> gTemperatureNoBC;
      
      new_dw->getModifiable(gacceleration,    lb->gAccelerationLabel,
			    dwindex, patch);
      new_dw->getModifiable(gvelocity_star,   lb->gVelocityStarLabel,
			    dwindex, patch);
      new_dw->getModifiable(gTemperatureRate, lb->gTemperatureRateLabel,
			    dwindex, patch);
      new_dw->get(gTemperatureNoBC, lb->gTemperatureNoBCLabel,
			dwindex, patch, Ghost::None, 0);
     // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles
      IntVector offset(0,0,0);
      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){
        BoundCondBase *vel_bcs, *temp_bcs, *sym_bcs;
        if (patch->getBCType(face) == Patch::None) {
	   vel_bcs  = patch->getBCValues(dwindex,"Velocity",face);
	   temp_bcs = patch->getBCValues(dwindex,"Temperature",face);
	   sym_bcs  = patch->getBCValues(dwindex,"Symmetric",face);
        } else
          continue;
         //__________________________________
         // Velocity and Acceleration
	  if (vel_bcs != 0) {
	    VelocityBoundCond* bc = 
	      dynamic_cast<VelocityBoundCond*>(vel_bcs);
	    //cout << "Velocity bc value = " << bc->getValue() << endl;
	    if (bc->getKind() == "Dirichlet") {
	      gvelocity_star.fillFace(patch, face,bc->getValue(),offset);
	      gacceleration.fillFace( patch, face,Vector(0.0,0.0,0.0),offset);
	    }
	  }
	  if (sym_bcs != 0) {
	     gvelocity_star.fillFaceNormal(patch, face,offset);
	     gacceleration.fillFaceNormal( patch, face,offset);
	  }
         //__________________________________
         // Temperature BC
	  if (temp_bcs != 0) {
	    TemperatureBoundCond* bc = 
	      dynamic_cast<TemperatureBoundCond*>(temp_bcs);
	    if (bc->getKind() == "Dirichlet") {
	      //cout << "Temperature bc value = " << bc->getValue() << endl;
              
            IntVector low = patch->getInteriorNodeLowIndex();
            IntVector hi  = patch->getInteriorNodeHighIndex();
	     double boundTemp = bc->getValue();
	     if(face==Patch::xplus || face==Patch::xminus){
		int I;
		if(face==Patch::xminus){ I=low.x(); }
		if(face==Patch::xplus){  I=hi.x()-1; }
		for (int j = low.y(); j<hi.y(); j++) { 
		  for (int k = low.z(); k<hi.z(); k++) {
		    gTemperatureRate[IntVector(I,j,k)] +=
		      (boundTemp - gTemperatureNoBC[IntVector(I,j,k)])/delT;
		  }
		}
	     }
	     if(face==Patch::yplus || face==Patch::yminus){
	       int J;
	       if(face==Patch::yminus){ J=low.y(); }
	       if(face==Patch::yplus){  J=hi.y()-1; }
	       for (int i = low.x(); i<hi.x(); i++) {
		  for (int k = low.z(); k<hi.z(); k++) {
		    gTemperatureRate[IntVector(i,J,k)] +=
		      (boundTemp - gTemperatureNoBC[IntVector(i,J,k)])/delT;
		  }
	       }
	     }
	     if(face==Patch::zplus || face==Patch::zminus){
	       int K;
	       if(face==Patch::zminus){ K=low.z(); }
	       if(face==Patch::zplus){  K=hi.z()-1; }
	       for (int i = low.x(); i<hi.x(); i++) {
		  for (int j = low.y(); j<hi.y(); j++) {
		    gTemperatureRate[IntVector(i,j,K)] +=
		      (boundTemp - gTemperatureNoBC[IntVector(i,j,K)])/delT;
		  }
	       }
	     }
	   }  // if(dirichlet)
	   if (bc->getKind() == "Neumann") {
	      //cout << "bc value = " << bc->getValue() << endl;
	   }
	 }  //if(temp_bc}
      }  // patch face loop

    } // matl loop
  }  // patch loop
}


void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset* ,
						DataWarehouse* old_dw,
						DataWarehouse* new_dw)
{

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing interpolateToParticlesAndUpdate on patch " << patch->getID()
         <<"\t MPM"<< endl;
  #endif
    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
    Vector vel(0.0,0.0,0.0);
    Vector acc(0.0,0.0,0.0);
  
    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    Vector CMX(0.0,0.0,0.0);
    Vector CMV(0.0,0.0,0.0);
    double ke=0;
    double massLost=0;
    int numMPMMatls=d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew;
      constParticleVariable<Vector> pvelocity, pexternalForce;
      ParticleVariable<Vector> pvelocitynew, pexternalForceNew;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      ParticleVariable<double> pmassNew,pvolumeNew,pTemperatureNew;
      //ParticleVariable<double> pToughness;

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<double> gTemperatureRate, gTemperature, gTemperatureNoBC;
      constNCVariable<double> dTdt, massBurnFraction, frictionalTempRate;

      delt_vartype delT;

      ParticleSubset* pset = old_dw->getParticleSubset(dwindex, patch);

      ParticleSubset* delete_particles = scinew ParticleSubset
	(pset->getParticleSet(),false,dwindex,patch);
    
      old_dw->get(px,                    lb->pXLabel,                    pset);
      old_dw->get(pmass,                 lb->pMassLabel,                 pset);
      old_dw->get(pvolume,               lb->pVolumeLabel,               pset);
      old_dw->get(pexternalForce,        lb->pExternalForceLabel,        pset);
      old_dw->get(pTemperature,          lb->pTemperatureLabel,          pset);
      old_dw->get(pvelocity,             lb->pVelocityLabel,             pset);
      new_dw->allocate(pTemperatureNew,  lb->pTemperatureLabel_preReloc, pset);
      new_dw->allocate(pvelocitynew,     lb->pVelocityLabel_afterUpdate, pset);
      new_dw->allocate(pxnew,            lb->pXLabel_preReloc,           pset);
      new_dw->allocate(pmassNew,         lb->pMassLabel_preReloc,        pset);
      new_dw->allocate(pvolumeNew,       lb->pVolumeLabel_preReloc,      pset);
      new_dw->allocate(pexternalForceNew,lb->pExternalForceLabel_preReloc,pset);
      pexternalForceNew.copyData(pexternalForce);

      new_dw->get(gvelocity_star,     lb->gVelocityStarLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gacceleration,      lb->gAccelerationLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gTemperatureRate,   lb->gTemperatureRateLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gTemperature,       lb->gTemperatureLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gTemperatureNoBC,   lb->gTemperatureNoBCLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(frictionalTempRate, lb->frictionalWorkLabel,
			dwindex, patch, Ghost::AroundCells, 1);

      if(d_with_ice){
        new_dw->get(dTdt, lb->dTdt_NCLabel,dwindex,patch,Ghost::AroundCells,1);
        new_dw->get(massBurnFraction, lb->massBurnFractionLabel,
		    dwindex,patch,Ghost::AroundCells,1);
      }
      else{
	NCVariable<double> dTdt_create, massBurnFraction_create;	
        new_dw->allocate(dTdt_create, lb->dTdt_NCLabel,
			 dwindex,patch,Ghost::AroundCells,1);
        new_dw->allocate(massBurnFraction_create, lb->massBurnFractionLabel,
			 dwindex,patch,Ghost::AroundCells,1);
        dTdt_create.initialize(0.);
        massBurnFraction_create.initialize(0.);
	dTdt = dTdt_create; // reference created data
	massBurnFraction = massBurnFraction_create; // reference created data
      }

      old_dw->get(delT, d_sharedState->get_delt_label() );

      double Cp=mpm_matl->getSpecificHeat();
      double rho_init=mpm_matl->getInitialDensity();

      IntVector ni[8];

    
      if(mpm_matl->getFractureModel()) {
        constParticleVariable<int> pConnectivity;
        new_dw->get(pConnectivity,  lb->pConnectivityLabel,  pset);
	
	constParticleVariable<Vector> pDisplacement;
	ParticleVariable<Vector> pDisplacement_new;
	old_dw->get(pDisplacement, lb->pDisplacementLabel,  pset);
        new_dw->allocate(pDisplacement_new,lb->pDisplacementLabel_preReloc,
			 pset);


        for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();
	    iter++){
	  particleIndex idx = *iter;

          double S[8];

          patch->findCellAndWeights(px[idx], ni, S);

          Connectivity connectivity(pConnectivity[idx]);
  	  int conn[8];
	  connectivity.getInfo(conn);
      	  Connectivity::modifyWeights(conn,S,Connectivity::connect);

          double tempRate = 0;
          // Accumulate the contribution from each surrounding vertex
          for(int k = 0; k < 8; k++) {
	     if( conn[k] ) {
                tempRate += (gTemperatureRate[ni[k]] + dTdt[ni[k]] +
			     frictionalTempRate[ni[k]])     * S[k];
             }
	  }
	  
          vel = Vector(0.0,0.0,0.0);
          acc = Vector(0.0,0.0,0.0);
          for(int k = 0; k < 8; k++) {
	     if( conn[k] ) {
	        vel += gvelocity_star[ni[k]]  * S[k];
   	        acc += gacceleration[ni[k]]   * S[k];
             }
	  }
	  
          // Update the particle's position and velocity
          pTemperatureNew[idx] = pTemperature[idx] + tempRate * delT;
          thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
	
	  int constraint = mpm_matl->getFractureModel()->getConstraint();
          if( constraint > 0) {
	    vel[constraint] = 0.;
	    acc[constraint] = 0.;
	  }
	  
	  pxnew[idx]        = px[idx]        + vel * delT;
          pvelocitynew[idx] = pvelocity[idx] + acc * delT;

	  pmassNew[idx]   = pmass[idx];
          pvolumeNew[idx] = pvolume[idx];

          ke += .5*pmass[idx]*pvelocitynew[idx].length2();
	  CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
	  CMV += pvelocitynew[idx]*pmass[idx];

	  pDisplacement_new[idx] = pDisplacement[idx] + vel * delT;
        }

        new_dw->put(pDisplacement_new, lb->pDisplacementLabel_preReloc);
      }
      else {  // Interpolate to particles if no fracture is involved
        for(ParticleSubset::iterator iter = pset->begin();
	    iter != pset->end(); iter++){
	  particleIndex idx = *iter;

          double S[8];
          Vector d_S[8];

          // Get the node indices that surround the cell
          patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S);

          vel = Vector(0.0,0.0,0.0);
          acc = Vector(0.0,0.0,0.0);
          double tempRate = 0;
          double burnFraction = 0;

          // Accumulate the contribution from each surrounding vertex
          for (int k = 0; k < 8; k++) {
	      vel      += gvelocity_star[ni[k]]  * S[k];
   	      acc      += gacceleration[ni[k]]   * S[k];
              tempRate += (gTemperatureRate[ni[k]] + dTdt[ni[k]] +
			   frictionalTempRate[ni[k]])     * S[k];
              burnFraction += massBurnFraction[ni[k]] * S[k];
          }

          // Update the particle's position and velocity
          pxnew[idx]           = px[idx] + vel * delT;
          pvelocitynew[idx]    = pvelocity[idx] + acc * delT;
          pTemperatureNew[idx] = pTemperature[idx] + tempRate * delT;
    
          double rho;
	  if(pvolume[idx] > 0.){
	    rho = pmass[idx]/pvolume[idx];
	  }
	  else{
	    rho = rho_init;
	  }
          pmassNew[idx]        = Max(pmass[idx]*(1.    - burnFraction),0.);
          pvolumeNew[idx]      = pmassNew[idx]/rho;
#if 1
	  if(pmassNew[idx] <= 3.e-15){
	    delete_particles->addParticle(idx);
	    pvelocitynew[idx] = Vector(0.,0.,0);
	    pxnew[idx] = px[idx];
	  }
#endif

          thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
          ke += .5*pmass[idx]*pvelocitynew[idx].length2();
	  CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
	  CMV += pvelocitynew[idx]*pmass[idx];
          massLost += (pmass[idx] - pmassNew[idx]);
        }
      }
      
      // Store the new result
      new_dw->put(pxnew,           lb->pXLabel_preReloc);
      new_dw->put(pvelocitynew,    lb->pVelocityLabel_afterUpdate);
      new_dw->put(pexternalForceNew, lb->pExternalForceLabel_preReloc);
      new_dw->put(pmassNew,        lb->pMassLabel_preReloc);
      new_dw->put(pvolumeNew,      lb->pVolumeLabel_preReloc);
      new_dw->put(pTemperatureNew, lb->pTemperatureLabel_preReloc);
      new_dw->deleteParticles(delete_particles);

      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      old_dw->get(pids, lb->pParticleIDLabel, pset);
      new_dw->allocate(pids_new, lb->pParticleIDLabel_preReloc, pset);
      pids_new.copyData(pids);
      new_dw->put(pids_new, lb->pParticleIDLabel_preReloc);
    }
    // DON'T MOVE THESE!!!
    new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
    new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
    new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);

//  cout << "Solid mass lost this timestep = " << massLost << endl;
//  cout << "Solid momentum after advection = " << CMV << endl;

//  cout << "THERMAL ENERGY " << thermal_energy << endl;
  }
}

void SerialMPM::computeCrackExtension(
                   const ProcessorGroup*,
		   const PatchSubset* patches,
		   const MaterialSubset* ,
		   DataWarehouse* old_dw,
		   DataWarehouse* new_dw)
{
#ifdef DOING
  cout <<"Doing computeCrackExtension "<<"\t\t\t\t MPM"<< endl;
#endif
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    mpm_matl->getFractureModel()->computeCrackExtension(
	  patches, mpm_matl, old_dw, new_dw);
  }
}

void SerialMPM::computeFracture(
                   const ProcessorGroup*,
		   const PatchSubset* patches,
		   const MaterialSubset* ,
		   DataWarehouse* old_dw,
		   DataWarehouse* new_dw)
{
#ifdef DOING
  cout <<"Doing computeFracture "<<"\t\t\t\t MPM"<< endl;
#endif
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    mpm_matl->getFractureModel()->computeFracture(
	  patches, mpm_matl, old_dw, new_dw);
  }
}

void SerialMPM::computeBoundaryContact(
                   const ProcessorGroup*,
		   const PatchSubset* patches,
		   const MaterialSubset* ,
		   DataWarehouse* old_dw,
		   DataWarehouse* new_dw)
{
#ifdef DOING
  cout <<"Doing computeBoundaryContact "<<"\t\t\t\t MPM"<< endl;
#endif
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    mpm_matl->getFractureModel()->
       computeBoundaryContact(patches, mpm_matl, old_dw, new_dw);
  }
}

void SerialMPM::carryForwardVariables( const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset*,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  #ifdef DOING
    cout <<"Doing carryForwardVariables on patch " << patch->getID()
         <<"\t\t\t MPM"<< endl;
  #endif
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      int matlindex = mpm_matl->getDWIndex();
        
      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);

      //stress
      constParticleVariable<Matrix3> pStress;
      ParticleVariable<Matrix3> pStress_new;
      if(mpm_matl->getFractureModel()) {
        new_dw->get(pStress, lb->pStressLabel_afterFracture, pset);
      }
      else {
        new_dw->get(pStress, lb->pStressLabel_afterStrainRate, pset);
      }
      new_dw->allocate(pStress_new, lb->pStressLabel_preReloc, pset);
      pStress_new.copyData(pStress);
      new_dw->put(pStress_new, lb->pStressLabel_preReloc);

      //velocity
      constParticleVariable<Vector> pVelocity;
      ParticleVariable<Vector> pVelocity_new;
      if(mpm_matl->getFractureModel()) {
        new_dw->get(pVelocity, lb->pVelocityLabel_afterFracture, pset);
      }
      else {
        new_dw->get(pVelocity, lb->pVelocityLabel_afterUpdate, pset);
      }
      new_dw->allocate(pVelocity_new, lb->pVelocityLabel_preReloc, pset);
      pVelocity_new.copyData(pVelocity);
      new_dw->put(pVelocity_new, lb->pVelocityLabel_preReloc);
    }
  }
}

void SerialMPM::setSharedState(SimulationStateP& ssp)
{
  d_sharedState = ssp;
}
