#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h> // 
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMPhysicalModules.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/HeatConduction/HeatConduction.h>
#include <Packages/Uintah/CCA/Components/MPM/Fracture/Fracture.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/Fracture/Connectivity.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/SymmetryBoundCond.h>
#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/NotFinished.h>

#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace SCIRun;

using namespace std;

SerialMPM::SerialMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  d_nextOutputTime=0.;
  d_fracture = false;
  d_SMALL_NUM_MPM=1e-200;
}

SerialMPM::~SerialMPM()
{
  delete lb;
  MPMPhysicalModules::kill();
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
       lb->registerPermanentParticleState(m,lb->pCrackNormal1Label,
					 lb->pCrackNormal1Label_preReloc); 
       lb->registerPermanentParticleState(m,lb->pCrackNormal2Label,
					 lb->pCrackNormal2Label_preReloc); 
       lb->registerPermanentParticleState(m,lb->pCrackNormal3Label,
					 lb->pCrackNormal3Label_preReloc); 
       lb->registerPermanentParticleState(m,lb->pToughnessLabel,
					  lb->pToughnessLabel_preReloc); 
       lb->registerPermanentParticleState(m,lb->pIsBrokenLabel,
					  lb->pIsBrokenLabel_preReloc); 
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
     lb->registerPermanentParticleState(m,lb->pDeformationMeasureLabel,
					lb->pDeformationMeasureLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pStressLabel,
					lb->pStressLabel_preReloc);
     
     mpm_matl->getConstitutiveModel()->addParticleState(lb->d_particleState[m],
					lb->d_particleState_preReloc[m]);
   }
}

void SerialMPM::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched)
{
  Task* t = scinew Task("SerialMPM::actuallyInitialize",
			this, &SerialMPM::actuallyInitialize);
  t->computes(lb->pXLabel);
  t->computes(lb->pMassLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  if(d_fracture){
    t->computes(lb->pIsBrokenLabel);
    t->computes(lb->pCrackNormal1Label);
    t->computes(lb->pCrackNormal2Label);
    t->computes(lb->pCrackNormal3Label);
    t->computes(lb->pToughnessLabel);
  }
  t->computes(d_sharedState->get_delt_label());
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
}

void SerialMPM::scheduleComputeStableTimestep(const LevelP&,
					      SchedulerP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
}

void SerialMPM::scheduleTimeAdvance(double , double ,
				    const LevelP&         level,
				    SchedulerP&     sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  //The next line is used for data analyze, please do not move.  --tan
  if(d_analyze) d_analyze->performAnalyze(sched, patches, matls);

  if(d_fracture) {
    scheduleSetPositions(                 sched, patches, matls);
    scheduleComputeBoundaryContact(       sched, patches, matls);
    scheduleComputeConnectivity(          sched, patches, matls);
  }
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
      
  if (MPMPhysicalModules::thermalContactModel) {
    scheduleComputeHeatExchange(          sched, patches, matls);
  }
  scheduleExMomInterpolated(              sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);
  scheduleComputeInternalForce(           sched, patches, matls);
  scheduleComputeInternalHeatRate(        sched, patches, matls);
  scheduleSolveEquationsMotion(           sched, patches, matls);
  scheduleSolveHeatEquations(             sched, patches, matls);
  scheduleIntegrateAcceleration(          sched, patches, matls);
  // scheduleIntegrateTemperatureRate(    sched, patches, matls);
  scheduleExMomIntegrated(                sched, patches, matls);
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);

  if(d_fracture) {
    scheduleComputeFracture(              sched, patches, matls);
  }
  scheduleCarryForwardVariables(          sched, patches, matls);
    
  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc, 
				    lb->d_particleState_preReloc,
				    lb->pXLabel, lb->d_particleState, matls);
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

  t->requires(Task::OldDW, lb->pXLabel, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormal1Label, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormal2Label, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormal3Label, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pIsBrokenLabel, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pVolumeLabel, Ghost::AroundCells, 1);

  t->requires(Task::NewDW, lb->pXXLabel, Ghost::None);

  t->computes(lb->pTouchNormalLabel);

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

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task( "SerialMPM::computeConnectivity",
			  this,&SerialMPM::computeConnectivity);

  t->requires(Task::OldDW, lb->pXLabel, Ghost::AroundCells, 1);
  t->requires(Task::NewDW, lb->pXXLabel, Ghost::None);
  t->requires(Task::OldDW, lb->pVolumeLabel, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pIsBrokenLabel, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormal1Label, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormal2Label, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormal3Label, Ghost::AroundCells, 1);
  t->requires(Task::NewDW, lb->pTouchNormalLabel, Ghost::AroundCells, 1);

  t->computes(lb->pConnectivityLabel);
  t->computes(lb->pContactNormalLabel);

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
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureNoBCLabel);
  t->computes(lb->gExternalHeatRateLabel);

  if(d_fracture) {
    t->requires(Task::NewDW,lb->pContactNormalLabel, Ghost::AroundNodes, 1 );
    t->requires(Task::NewDW, lb->pConnectivityLabel, Ghost::AroundNodes, 1 );
    t->computes(lb->gMassContactLabel);
  }
     
  t->computes(lb->gMassLabel);
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
		        MPMPhysicalModules::thermalContactModel,
		        &ThermalContact::computeHeatExchange);

  MPMPhysicalModules::thermalContactModel->addComputesAndRequires(t, patches,
								  matls);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleExMomInterpolated(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* t = scinew Task("Contact::exMomInterpolated",
		    MPMPhysicalModules::contactModel,
		    &Contact::exMomInterpolated);

  MPMPhysicalModules::contactModel->
    addComputesAndRequiresInterpolated(t, patches, matls);
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

  t->requires(Task::NewDW,lb->gMassLabel,                 Ghost::None);
  t->requires(Task::NewDW,lb->pStressAfterStrainRateLabel,Ghost::AroundNodes,1);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,       Ghost::AroundNodes,1);
  t->requires(Task::OldDW,lb->pMassLabel,                 Ghost::AroundNodes,1);

  if(d_sharedState->getNumMatls() != d_sharedState->getNumMPMMatls()){
    t->requires(Task::NewDW, lb->pPressureLabel,          Ghost::AroundNodes,1);
  }

  if(d_fracture) {
    t->requires(Task::NewDW, lb->pConnectivityLabel, Ghost::AroundNodes, 1 );
    t->requires(Task::NewDW, lb->pContactNormalLabel, Ghost::AroundNodes, 1 );
  }

  t->computes(lb->gInternalForceLabel);
  t->computes(lb->gStressForSavingLabel);
  t->computes(lb->NTractionZMinusLabel);
  t->computes(lb->gStressForSavingLabel);

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

  if(d_fracture)
    t->requires(Task::NewDW, lb->gMassContactLabel,   Ghost::None);
  else
    t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
      
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  if(d_sharedState->getNumMatls() != d_sharedState->getNumMPMMatls()){
    t->requires(Task::NewDW, lb->gradPressNCLabel,  Ghost::None);
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

  if(d_fracture)
    t->requires(Task::NewDW, lb->gMassContactLabel,   Ghost::None);
  else
    t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);

  t->requires(Task::NewDW, lb->gVolumeLabel,           Ghost::None);
  t->requires(Task::NewDW, lb->gInternalHeatRateLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalHeatRateLabel, Ghost::None);

  if(MPMPhysicalModules::thermalContactModel) {
    t->requires(Task::NewDW, lb->gThermalContactHeatExchangeRateLabel,
		Ghost::None);
  }
		
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

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gAccelerationLabel,    Ghost::None);
  t->requires(Task::NewDW, lb->gMomExedVelocityLabel, Ghost::None);

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
		   MPMPhysicalModules::contactModel,
		   &Contact::exMomIntegrated);
  MPMPhysicalModules::contactModel->addComputesAndRequiresIntegrated(t,
								     patches,
								     matls);
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

  int numMPMMatls = d_sharedState->getNumMPMMatls();
  int numALLMatls = d_sharedState->getNumMatls();
  Task* t=scinew Task("SerialMPM::interpolateToParticlesAndUpdate",
		    this, &SerialMPM::interpolateToParticlesAndUpdate);


  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gMomExedAccelerationLabel, Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->gMomExedVelocityStarLabel, Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,     Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->gTemperatureLabel,         Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel,     Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pXLabel,                   Ghost::None);
  t->requires(Task::OldDW, lb->pExternalForceLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,                Ghost::None);
  t->requires(Task::OldDW, lb->pParticleIDLabel,          Ghost::None);
  t->requires(Task::OldDW, lb->pTemperatureLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->pVelocityLabel,            Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,                Ghost::None);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,      Ghost::None);

  
  if(numMPMMatls!=numALLMatls){
    t->requires(Task::NewDW, lb->dTdt_NCLabel,            Ghost::AroundCells,1);
  }

  if(d_fracture) {
    t->requires(Task::NewDW, lb->pConnectivityLabel,      Ghost::None);
    t->requires(Task::NewDW, lb->pContactNormalLabel,     Ghost::None);
  }

  t->computes(lb->pVelocityLabel_preReloc);
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

  t->requires(Task::OldDW, lb->pXLabel,            Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pIsBrokenLabel,     Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormal1Label, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormal2Label, Ghost::AroundCells, 1);
  t->requires(Task::OldDW, lb->pCrackNormal3Label, Ghost::AroundCells, 1);

  t->requires(Task::NewDW, lb->pXXLabel,                    Ghost::None);
  t->requires(Task::OldDW, lb->pVolumeLabel,                Ghost::None);
  t->requires(Task::NewDW, lb->pStressAfterStrainRateLabel, Ghost::None);
  t->requires(Task::NewDW, lb->pStrainEnergyLabel,          Ghost::None);
  t->requires(Task::OldDW, lb->pToughnessLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->pRotationRateLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->pConnectivityLabel,          Ghost::None);

  t->requires(Task::NewDW, lb->gStressForSavingLabel, Ghost::AroundCells, 1);

  t->computes(lb->pStressAfterFractureReleaseLabel);
  t->computes(lb->pIsBrokenLabel_preReloc);
  t->computes(lb->pCrackNormal1Label_preReloc);
  t->computes(lb->pCrackNormal2Label_preReloc);
  t->computes(lb->pCrackNormal3Label_preReloc);
  t->computes(lb->pToughnessLabel_preReloc);
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

  if(d_fracture)
    t->requires(Task::NewDW, lb->pStressAfterFractureReleaseLabel,
		Ghost::None);
  else
      t->requires(Task::NewDW, lb->pStressAfterStrainRateLabel,
		Ghost::None);			 

  t->computes(lb->pStressLabel_preReloc);
  sched->addTask(t, patches, matls);
}

void SerialMPM::actuallyInitialize(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse*,
				   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    PerPatch<long> NAPID(0);
    if(new_dw->exists(lb->ppNAPIDLabel, 0, patch))
      new_dw->get(NAPID,lb->ppNAPIDLabel, 0, patch);

    for(int m=0;m<matls->size();m++){
       NOT_FINISHED("not quite right - mapping of matls, use matls->get()");
       MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
       particleIndex numParticles = mpm_matl->countParticles(patch);

       mpm_matl->createParticles(numParticles, NAPID, patch, new_dw);

       NAPID=NAPID + numParticles;

       mpm_matl->getConstitutiveModel()->initializeCMData(patch,
						mpm_matl, new_dw);
       if(mpm_matl->getFractureModel()) {
	 mpm_matl->getFractureModel()->initializeFractureModelData( patch,
						mpm_matl, new_dw);
       }       

       int dwindex = mpm_matl->getDWIndex();

       MPMPhysicalModules::contactModel->
			initializeContact(patch,dwindex,new_dw);
    }
    new_dw->put(NAPID, lb->ppNAPIDLabel, 0, patch);
  }
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
		   const MaterialSubset* matls,
		   DataWarehouse* old_dw,
		   DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    mpm_matl->getFractureModel()->computeConnectivity(
	  patches, mpm_matl, old_dw, new_dw);
  }
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorGroup*,
					   const PatchSubset* patches,
					   const MaterialSubset* matls,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int numMatls = d_sharedState->getNumMPMMatls();
    int numALLMatls = d_sharedState->getNumMatls();

    NCVariable<double> totalgmass;
    new_dw->allocate(totalgmass,lb->gMassLabel,numALLMatls, patch);
    totalgmass.initialize(d_SMALL_NUM_MPM);

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle data
      ParticleVariable<Point>  px;
      ParticleVariable<double> pmass, pvolume, pTemperature;
      ParticleVariable<Vector> pvelocity, pexternalforce;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      old_dw->get(pexternalforce, lb->pExternalForceLabel, pset);
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);

      // Create arrays for the grid data
      NCVariable<double> gmass,gmassContact;
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

      if(mpm_matl->getFractureModel()) {  // Do interpolation with fracture

        new_dw->allocate(gmassContact,lb->gMassContactLabel, matlindex, patch);
        gmassContact.initialize(d_SMALL_NUM_MPM);

        ParticleVariable<int> pConnectivity;
        ParticleVariable<Vector> pContactNormal;

        new_dw->get(pConnectivity, lb->pConnectivityLabel, pset);
        new_dw->get(pContactNormal, lb->pContactNormalLabel, pset);

        for(ParticleSubset::iterator iter = pset->begin();
						  iter != pset->end(); iter++){
	  particleIndex idx = *iter;

	  // Get the node indices that surround the cell
	  IntVector ni[8];
	  double S_connect[8],S_contact[8];

  	  patch->findCellAndWeights(px[idx], ni, S_connect);
	  for(int k = 0; k < 8; k++) S_contact[k] = S_connect[k]; //make a copy

          Connectivity connectivity(pConnectivity[idx]);
  	  int conn[8];
	  connectivity.getInfo(conn);
      	  connectivity.modifyWeights(conn,S_connect,Connectivity::connect);
      	  connectivity.modifyWeights(conn,S_contact,Connectivity::contact);

          for(int k = 0; k < 8; k++) {
	    if( patch->containsNode(ni[k]) ) {
	      if( conn[k] == Connectivity::connect || 
	          conn[k] == Connectivity::contact) {
	        totalgmass[ni[k]]     += pmass[idx]          * S_connect[k];
	        gmass[ni[k]]          += pmass[idx]          * S_connect[k];
	        totalmass += pmass[idx] * S_connect[k];
	        gmassContact[ni[k]]   += pmass[idx]          * S_contact[k];
	        gTemperature[ni[k]]   += pTemperature[idx] * 
	        pmass[idx] * S_contact[k];
	        gvolume[ni[k]]        += pvolume[idx]        * S_contact[k];
	      }

	      if( conn[k] == Connectivity::connect ) {
	       gexternalforce[ni[k]] += pexternalforce[idx] * S_contact[k];
	       gvelocity[ni[k]]      += pvelocity[idx] *pmass[idx]*S_contact[k];
	      }
	      else if( conn[k] == Connectivity::contact ) {
	       gexternalforce[ni[k]] += pContactNormal[idx] * 
		 ( Dot(pContactNormal[idx],pexternalforce[idx]) * S_contact[k]);
	       gvelocity[ni[k]]      += pContactNormal[idx] * 
                 ( Dot(pContactNormal[idx],pvelocity[idx]) * 
                 pmass[idx] * S_contact[k] );
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
	       totalgmass[ni[k]]     += pmass[idx]          * S[k];
	       gmass[ni[k]]          += pmass[idx]          * S[k];
	       gvolume[ni[k]]        += pvolume[idx]        * S[k];
	       gexternalforce[ni[k]] += pexternalforce[idx] * S[k];
	       gvelocity[ni[k]]      += pvelocity[idx]    * pmass[idx] * S[k];
	       gTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];

	       totalmass += pmass[idx] * S[k];
	    }
	  }
        }
      }

      if(mpm_matl->getFractureModel()) {  // Do interpolation with fracture
         for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
	   gvelocity[*iter] /= gmassContact[*iter];
           gTemperatureNoBC[*iter] = gTemperature[*iter]/gmassContact[*iter];
           gTemperature[*iter] /= gmassContact[*iter];
	 }
       }
       else {  // Do interpolation without fracture
         for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
	   gvelocity[*iter] /= gmass[*iter];
           gTemperatureNoBC[*iter] = gTemperature[*iter]/gmass[*iter];
           gTemperature[*iter] /= gmass[*iter];
	 }
       }

      // Apply grid boundary conditions to the velocity before storing the data

      IntVector offset = 
	patch->getInteriorCellLowIndex() - patch->getCellLowIndex();
      // cout << "offset = " << offset << endl;
      for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
	vector<BoundCondBase* > bcs;
	bcs = patch->getBCValues(face);

	for (int i = 0; i<(int)bcs.size(); i++ ) {
	  string bcs_type = bcs[i]->getType();
	  if (bcs_type == "Velocity") {
	    VelocityBoundCond* bc = 
	      dynamic_cast<VelocityBoundCond*>(bcs[i]);
	    if (bc->getKind() == "Dirichlet") {
	      //cout << "Velocity bc value = " << bc->getValue() << endl;
	      gvelocity.fillFace(face,bc->getValue(),offset);
	    }
	  }
	  if (bcs_type == "Symmetric") {
	     gvelocity.fillFaceNormal(face,offset);
	  }
	  if (bcs_type == "Temperature") {
            TemperatureBoundCond* bc =
              dynamic_cast<TemperatureBoundCond*>(bcs[i]);
            if (bc->getKind() == "Dirichlet") {
              gTemperature.fillFace(face,bc->getValue(),offset);
	    }
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

      if(mpm_matl->getFractureModel()) {
        new_dw->put(gmassContact, lb->gMassContactLabel,  matlindex, patch);
      }
    }  // End loop over materials
    new_dw->put(totalgmass,       lb->gMassLabel,          numALLMatls, patch);
  }  // End loop over patches
}

void SerialMPM::computeStressTensor(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* ,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw)
{
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

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      int matlindex = mpm_matl->getDWIndex();
        
      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);

      ParticleVariable<Point> pX;
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

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    Matrix3 Id;
    Id.Identity();

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numALLMatls = d_sharedState->getNumMatls();

    double integralTraction = 0.;
    double integralArea = 0.;

    NCVariable<Matrix3>       gstressglobal;
    NCVariable<double>        gmassglobal;
    new_dw->get(gmassglobal,  lb->gMassLabel,numALLMatls,patch, Ghost::None, 0);
    new_dw->allocate(gstressglobal,lb->gStressForSavingLabel,numALLMatls,patch);

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle position, volume
      // and the constitutive model
      ParticleVariable<Point>   px;
      ParticleVariable<Point>   pxonpatch;
      ParticleVariable<double>  pvol, pmass;
      ParticleVariable<Matrix3> pstress;
      NCVariable<Vector>        internalforce;
      NCVariable<Matrix3>       gstress;
      NCVariable<double>        gmass;

      ParticleVariable<double>  p_pressure;;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel, pset);
      old_dw->get(pmass,   lb->pMassLabel, pset);
      new_dw->get(pvol,    lb->pVolumeDeformedLabel, pset);
      new_dw->get(pstress, lb->pStressAfterStrainRateLabel, pset);
      new_dw->get(gmass,   lb->gMassLabel, matlindex, patch, Ghost::None, 0);

      new_dw->allocate(gstress,lb->gStressForSavingLabel, matlindex, patch);
      new_dw->allocate(internalforce, lb->gInternalForceLabel,
						matlindex, patch);

      if(numMPMMatls==numALLMatls){
	new_dw->allocate(p_pressure,lb->pPressureLabel, pset);
	for(ParticleSubset::iterator iter = pset->begin();
                                     iter != pset->end(); iter++){
	   p_pressure[*iter]=0.0;
	}
      }
      else {
        new_dw->get(p_pressure,lb->pPressureLabel, pset);
      }

      internalforce.initialize(Vector(0,0,0));

      if(mpm_matl->getFractureModel()) {
        ParticleVariable<int>    pConnectivity;
        ParticleVariable<Vector> pContactNormal;
        new_dw->get(pConnectivity,  lb->pConnectivityLabel,  pset);
        new_dw->get(pContactNormal, lb->pContactNormalLabel, pset);

        for(ParticleSubset::iterator iter = pset->begin();
						iter != pset->end(); iter++){
           particleIndex idx = *iter;
  
           // Get the node indices that surround the cell
           IntVector ni[8];

           Vector d_S_connect[8],d_S_contact[8];
	   double   S_connect[8],  S_contact[8];

           patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, 
							S_connect, d_S_connect);
	   //make a copy
	   for(int k = 0; k < 8; k++) {
	     S_contact[k] = S_connect[k];
	     d_S_contact[k] = d_S_connect[k];
	   }

           Connectivity connectivity(pConnectivity[idx]);
  	   int conn[8];
	   connectivity.getInfo(conn);
	 
           connectivity.modifyShapeDerivatives(
				      conn,d_S_connect,Connectivity::connect);
      	   connectivity.modifyWeights(conn,S_connect,Connectivity::connect);
           connectivity.modifyShapeDerivatives(
				      conn,d_S_contact,Connectivity::contact);
      	   connectivity.modifyWeights(conn,S_contact,Connectivity::contact);

           for(int k = 0; k < 8; k++) {
	     if( patch->containsNode(ni[k]) ) {
	       if( conn[k] == Connectivity::connect ) {
                 gstress[ni[k]] += pstress[idx] * pmass[idx] * S_connect[k];
                 gstressglobal[ni[k]] += pstress[idx] * pmass[idx]*S_connect[k];
	       }

               if(conn[k] == Connectivity::connect) {
	         Vector div(d_S_contact[k].x()*oodx[0],
	                    d_S_contact[k].y()*oodx[1],
			    d_S_contact[k].z()*oodx[2]);
	         internalforce[ni[k]] -=
		     (div * (pstress[idx] - Id*p_pressure[idx]) * pvol[idx]);
	       }
	       else if(conn[k] == Connectivity::contact) {
                 Vector div(d_S_contact[k].x()*oodx[0],
	                    d_S_contact[k].y()*oodx[1],
		            d_S_contact[k].z()*oodx[2]);
	         internalforce[ni[k]] -= pContactNormal[idx] *
		   Dot( div * ((pstress[idx] - Id*p_pressure[idx]) * pvol[idx]),
		        pContactNormal[idx] );
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

    IntVector offset = 
	patch->getInteriorCellLowIndex() - patch->getCellLowIndex();
    IntVector low = gstress.getLowIndex() + offset;
    IntVector hi  = gstress.getHighIndex() - offset;

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
  new_dw->put(gstressglobal,  lb->gStressForSavingLabel, numALLMatls, patch);
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
      NCVariable<double>       gTemperature;
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

    Vector zero(0.,0.,0.);
    Vector gravity = d_sharedState->getGravity();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label() );

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Get required variables for this patch
      NCVariable<Vector> internalforce;
      NCVariable<Vector> externalforce;
      NCVariable<Vector> gradPressNC;  // for MPMICE

      new_dw->get(internalforce, lb->gInternalForceLabel, matlindex, patch,
							   Ghost::None, 0);
      new_dw->get(externalforce, lb->gExternalForceLabel, matlindex, patch,
							   Ghost::None, 0);

      NCVariable<double> mass;
      if(mpm_matl->getFractureModel())
        new_dw->get(mass, lb->gMassContactLabel,matlindex,patch, Ghost::None,0);
      else
        new_dw->get(mass, lb->gMassLabel,       matlindex,patch, Ghost::None,0);

      if(d_sharedState->getNumMatls() != d_sharedState->getNumMPMMatls()){
         new_dw->get(gradPressNC,lb->gradPressNCLabel,    matlindex, patch,
							   Ghost::None, 0);
      }
      else{
	 new_dw->allocate(gradPressNC,lb->gradPressNCLabel, matlindex, patch);
	 gradPressNC.initialize(zero);
      }

      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocate(acceleration, lb->gAccelerationLabel, matlindex, patch);
      acceleration.initialize(zero);

      // Do the computation of a = F/m for nodes where m!=0.0
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	  acceleration[*iter] =
		(internalforce[*iter] + externalforce[*iter] +
			gradPressNC[*iter]/delT)/ mass[*iter] + gravity;
      }

      // Put the result in the datawarehouse
      new_dw->put(acceleration, lb->gAccelerationLabel, matlindex, patch);
    }
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

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      double specificHeat = mpm_matl->getSpecificHeat();
     
      // Get required variables for this patch
      NCVariable<double> mass,internalHeatRate,externalHeatRate,gvolume;
      NCVariable<double> thermalContactHeatExchangeRate;

      if(mpm_matl->getFractureModel())
        new_dw->get(mass, lb->gMassContactLabel,dwindex, patch, Ghost::None, 0);
      else
        new_dw->get(mass, lb->gMassLabel,       dwindex, patch, Ghost::None, 0);
	
      new_dw->get(gvolume, lb->gVolumeLabel,    dwindex, patch, Ghost::None, 0);
      new_dw->get(internalHeatRate, lb->gInternalHeatRateLabel,
					        dwindex, patch, Ghost::None, 0);
      new_dw->get(externalHeatRate, lb->gExternalHeatRateLabel,
					        dwindex, patch, Ghost::None, 0);

      if(MPMPhysicalModules::thermalContactModel) {
        new_dw->get(thermalContactHeatExchangeRate,
                  lb->gThermalContactHeatExchangeRateLabel, 
                  dwindex, patch, Ghost::None, 0);
      }

      Vector dx = patch->dCell();
      for(Patch::FaceType face = Patch::startFace;
        face <= Patch::endFace; face=Patch::nextFace(face)){
        vector<BoundCondBase* > bcs;
        bcs = patch->getBCValues(face);
        for (int i = 0; i<(int)bcs.size(); i++ ) {
          string bcs_type = bcs[i]->getType();
          if (bcs_type == "Temperature") {
            TemperatureBoundCond* bc =
                       dynamic_cast<TemperatureBoundCond*>(bcs[i]);
            if (bc->getKind() == "Neumann"){
	      double value = bc->getValue();
	      IntVector offset = 
		patch->getInteriorCellLowIndex() - patch->getCellLowIndex();
              IntVector low = internalHeatRate.getLowIndex() + offset;
              IntVector hi = internalHeatRate.getHighIndex() - offset;
	     
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
      }

      // Create variables for the results
      NCVariable<double> temperatureRate;
      new_dw->allocate(temperatureRate,lb->gTemperatureRateLabel,dwindex,patch);
      temperatureRate.initialize(0.0);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	  temperatureRate[*iter] = (internalHeatRate[*iter]
		                 +  externalHeatRate[*iter]) /
				  (mass[*iter] * specificHeat);
          if(MPMPhysicalModules::thermalContactModel) {
            temperatureRate[*iter]+=thermalContactHeatExchangeRate[*iter];
          }
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

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      // Get required variables for this patch
      NCVariable<Vector>        acceleration;
      NCVariable<Vector>        velocity;
      delt_vartype delT;

      new_dw->get(acceleration, lb->gAccelerationLabel,    dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(velocity,     lb->gMomExedVelocityLabel, dwindex, patch,
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
      new_dw->allocate(temperatureStar,lb->gTemperatureStarLabel,dwindex,patch);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        temperatureStar[*iter] = temperature[*iter] +
				 temperatureRate[*iter] * delT;
      }

      new_dw->put( temperatureStar, lb->gTemperatureStarLabel, dwindex, patch );
    }
  }
}

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset* ,
						DataWarehouse* old_dw,
						DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
    Vector vel(0.0,0.0,0.0);
    Vector acc(0.0,0.0,0.0);
  
    double tempRate;
  
    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    Vector CMX(0.0,0.0,0.0);
    Vector CMV(0.0,0.0,0.0);
    double ke=0;
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    int numALLMatls=d_sharedState->getNumMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      ParticleVariable<Point>  px, pxnew;
      ParticleVariable<Vector> pvelocity, pvelocitynew, pexternalForce;
      ParticleVariable<double> pmass, pmassNew,pvolume,pvolumeNew;
      ParticleVariable<double> pToughness;
      ParticleVariable<double> pTemperature, pTemperatureNew; 

      // Get the arrays of grid data on which the new part. values depend
      NCVariable<Vector> gvelocity_star, gacceleration;
      NCVariable<double> gTemperatureRate, gTemperature, gTemperatureNoBC;
      NCVariable<double> dTdt;

      delt_vartype delT;

      ParticleSubset* pset = old_dw->getParticleSubset(dwindex, patch);
    
      old_dw->get(px,                    lb->pXLabel,                    pset);
      old_dw->get(pmass,                 lb->pMassLabel,                 pset);
      old_dw->get(pvolume,               lb->pVolumeLabel,               pset);
      old_dw->get(pexternalForce,        lb->pExternalForceLabel,        pset);
      old_dw->get(pTemperature,          lb->pTemperatureLabel,          pset);
      old_dw->get(pvelocity,             lb->pVelocityLabel,             pset);
      new_dw->allocate(pTemperatureNew,  lb->pTemperatureLabel_preReloc, pset);
      new_dw->allocate(pvelocitynew,     lb->pVelocityLabel,             pset);
      new_dw->allocate(pxnew,            lb->pXLabel_preReloc,           pset);
      new_dw->allocate(pmassNew,         lb->pMassLabel_preReloc,        pset);
      new_dw->allocate(pvolumeNew,       lb->pVolumeLabel_preReloc,      pset);

      new_dw->get(gvelocity_star,   lb->gMomExedVelocityStarLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gacceleration,    lb->gMomExedAccelerationLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gTemperatureRate, lb->gTemperatureRateLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gTemperature,     lb->gTemperatureLabel,
			dwindex, patch, Ghost::AroundCells, 1);
      new_dw->get(gTemperatureNoBC, lb->gTemperatureNoBCLabel,
			dwindex, patch, Ghost::AroundCells, 1);

      if(numMPMMatls!=numALLMatls){
        new_dw->get(dTdt, lb->dTdt_NCLabel, dwindex,patch,Ghost::AroundCells,1);
      }
      else{
        new_dw->allocate(dTdt, lb->dTdt_NCLabel,dwindex,patch,IntVector(1,1,1));
        dTdt.initialize(0.);
      }

      old_dw->get(delT, d_sharedState->get_delt_label() );

      double Cp=mpm_matl->getSpecificHeat();

      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles
      IntVector offset = 
	patch->getInteriorCellLowIndex() - patch->getCellLowIndex();
      for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
	vector<BoundCondBase* > bcs;
	bcs = patch->getBCValues(face);
	//cout << "number of bcs on face " << face << " = " 
	//     << bcs.size() << endl;

	for (int i = 0; i<(int)bcs.size(); i++ ) {
	  string bcs_type = bcs[i]->getType();
	  if (bcs_type == "Velocity") {
	    VelocityBoundCond* bc = 
	      dynamic_cast<VelocityBoundCond*>(bcs[i]);
	    //cout << "Velocity bc value = " << bc->getValue() << endl;
	    if (bc->getKind() == "Dirichlet") {
	      gvelocity_star.fillFace(face,bc->getValue(),offset);
	      gacceleration.fillFace(face,Vector(0.0,0.0,0.0),offset);
	    }
	  }
	  if (bcs_type == "Symmetric") {
	     gvelocity_star.fillFaceNormal(face,offset);
	     gacceleration.fillFaceNormal(face,offset);
	  }
	  if (bcs_type == "Temperature") {
	    TemperatureBoundCond* bc = 
	      dynamic_cast<TemperatureBoundCond*>(bcs[i]);
	    if (bc->getKind() == "Dirichlet") {
	      //cout << "Temperature bc value = " << bc->getValue() << endl;
	      IntVector low = gTemperature.getLowIndex() + offset;
	      IntVector hi = gTemperature.getHighIndex() - offset;
	      double boundTemp = bc->getValue();
	      if(face==Patch::xplus || face==Patch::xminus){
		int I;
		if(face==Patch::xminus){ I=low.x(); }
		if(face==Patch::xplus){ I=hi.x()-1; }
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
		if(face==Patch::yplus){ J=hi.y()-1; }
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
		if(face==Patch::zplus){ K=hi.z()-1; }
		for (int i = low.x(); i<hi.x(); i++) {
		  for (int j = low.y(); j<hi.y(); j++) {
		    gTemperatureRate[IntVector(i,j,K)] +=
		      (boundTemp - gTemperatureNoBC[IntVector(i,j,K)])/delT;
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

      IntVector ni[8];
    
      if(mpm_matl->getFractureModel()) {
        ParticleVariable<int> pConnectivity;
        ParticleVariable<Vector> pContactNormal;
        new_dw->get(pConnectivity,  lb->pConnectivityLabel,  pset);
        new_dw->get(pContactNormal, lb->pContactNormalLabel, pset);

        for(ParticleSubset::iterator iter = pset->begin();
						iter != pset->end(); iter++){
	  particleIndex idx = *iter;

          double S_connect[8],S_contact[8];
          Vector d_S_connect[8],d_S_contact[8];

          patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, 
							S_connect, d_S_connect);
	   //make a copy
	  for(int k = 0; k < 8; k++) {
	     S_contact[k] = S_connect[k];
	     d_S_contact[k] = d_S_connect[k];
	  }

          Connectivity connectivity(pConnectivity[idx]);
  	  int conn[8];
	  connectivity.getInfo(conn);
	 
          connectivity.modifyShapeDerivatives(
				     conn,d_S_connect,Connectivity::connect);
      	  connectivity.modifyWeights(conn,S_connect,Connectivity::connect);
          connectivity.modifyShapeDerivatives(
				     conn,d_S_contact,Connectivity::contact);
      	  connectivity.modifyWeights(conn,S_contact,Connectivity::contact);

	  int numConnectedNodes = 0;

          vel = Vector(0.0,0.0,0.0);
          acc = Vector(0.0,0.0,0.0);

          tempRate = 0;
	
          // Accumulate the contribution from each surrounding vertex
          for(int k = 0; k < 8; k++) {
	     if( conn[k] == Connectivity::connect || 
	         conn[k] == Connectivity::contact) {
                tempRate += (gTemperatureRate[ni[k]] + dTdt[ni[k]])
							* S_connect[k];
		numConnectedNodes++;
             }

	     if( conn[k] == Connectivity::connect ) {
	        vel += gvelocity_star[ni[k]]  * S_contact[k];
   	        acc += gacceleration[ni[k]]   * S_contact[k];
             }
	     else if( conn[k] == Connectivity::contact ) {
	        vel += pContactNormal[idx] *
	          ( Dot(pContactNormal[idx], gvelocity_star[ni[k]]) * 
		    S_contact[k] );
   	        acc += pContactNormal[idx] *
	          ( Dot(pContactNormal[idx], gacceleration[ni[k]]) * 
		    S_contact[k] );
	     }
	  }

          // Update the particle's position and velocity
          pTemperatureNew[idx] = pTemperature[idx] + tempRate * delT;
          thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
	
	  if(numConnectedNodes != 0) {
            pxnew[idx]        = px[idx]        + vel * delT;
            pvelocitynew[idx] = pvelocity[idx] + acc * delT;
          }
	  else {        
   	    //for isolated particles in fracture
            pxnew[idx]      =  px[idx] + pvelocity[idx] * delT;
            pvelocitynew[idx] = pvelocity[idx] +
	    pexternalForce[idx] / (pmass[idx] * delT);
          }
	  pmassNew[idx]   = pmass[idx];
          pvolumeNew[idx] = pvolume[idx];

          ke += .5*pmass[idx]*pvelocitynew[idx].length2();
	  CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
	  CMV += pvelocitynew[idx]*pmass[idx];
        }
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
          tempRate = 0;

          // Accumulate the contribution from each surrounding vertex
          for (int k = 0; k < 8; k++) {
	      vel      += gvelocity_star[ni[k]]  * S[k];
   	      acc      += gacceleration[ni[k]]   * S[k];
              tempRate += (gTemperatureRate[ni[k]] + dTdt[ni[k]]) * S[k];
          }

          // Update the particle's position and velocity
          pxnew[idx]      = px[idx] + vel * delT;
          pvelocitynew[idx] = pvelocity[idx] + acc * delT;
          pTemperatureNew[idx] = pTemperature[idx] + tempRate * delT;
          pmassNew[idx]        = pmass[idx];
          pvolumeNew[idx]      = pvolume[idx];

          thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
          ke += .5*pmass[idx]*pvelocitynew[idx].length2();
	  CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
	  CMV += pvelocitynew[idx]*pmass[idx];
        }
      }

      // Store the new result
      new_dw->put(pxnew,           lb->pXLabel_preReloc);
      new_dw->put(pvelocitynew,    lb->pVelocityLabel_preReloc);
      new_dw->put(pexternalForce,  lb->pExternalForceLabel_preReloc);
      new_dw->put(pmassNew,        lb->pMassLabel_preReloc);
      new_dw->put(pvolumeNew,      lb->pVolumeLabel_preReloc);
      new_dw->put(pTemperatureNew, lb->pTemperatureLabel_preReloc);

      ParticleVariable<long> pids;
      old_dw->get(pids, lb->pParticleIDLabel, pset);
      new_dw->put(pids, lb->pParticleIDLabel_preReloc);
    }
    // DON'T MOVE THESE!!!
    new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
    new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
    new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);

//  cout << "Solid momentum after advection = " << CMV << endl;

//  cout << "THERMAL ENERGY " << thermal_energy << endl;
  }
}

void SerialMPM::computeFracture(
                   const ProcessorGroup*,
		   const PatchSubset* patches,
		   const MaterialSubset* ,
		   DataWarehouse* old_dw,
		   DataWarehouse* new_dw)
{
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
        new_dw->get(pStress, lb->pStressAfterStrainRateLabel,      pset);
      }

      new_dw->put(pStress, lb->pStressLabel_preReloc);
    }
  }
}
