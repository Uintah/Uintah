#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/Burn/HEBurn.h>
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

     if(mpm_matl->getHEBurnModel()->getBurns()){
       //     lb->registerPermanentParticleState(m,lb->pSurfLabel,
       //lb->pSurfLabel_preReloc);
       lb->registerPermanentParticleState(m,lb->pIsIgnitedLabel,
					  lb->pIsIgnitedLabel_preReloc);
     }
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

   int numMatls = d_sharedState->getNumMPMMatls();

   for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){

     const Patch* patch=*iter;
     {
	 Task* t = scinew Task("SerialMPM::actuallyInitialize", patch, dw, dw,
			       this, &SerialMPM::actuallyInitialize);
	 for(int m = 0; m < numMatls; m++){
           MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
           int idx = mpm_matl->getDWIndex();
	   t->computes(dw, lb->pXLabel,             idx, patch);
	   t->computes(dw, lb->pMassLabel,          idx, patch);
	   t->computes(dw, lb->pVolumeLabel,        idx, patch);
	   t->computes(dw, lb->pTemperatureLabel,   idx, patch);
	   t->computes(dw, lb->pVelocityLabel,      idx, patch);
	   t->computes(dw, lb->pExternalForceLabel, idx, patch);
	   t->computes(dw, lb->pParticleIDLabel,    idx, patch);
	   if(d_fracture){
	      t->computes(dw, lb->pIsBrokenLabel,           idx, patch);
	      t->computes(dw, lb->pCrackNormal1Label,       idx, patch);
	      t->computes(dw, lb->pCrackNormal2Label,       idx, patch);
	      t->computes(dw, lb->pCrackNormal3Label,       idx, patch);
	      t->computes(dw, lb->pToughnessLabel,          idx, patch);
	   }
	 }
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
   for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;
    
      //The next line is used for data analyze, please do not move.  --tan
      if(d_analyze) d_analyze->performAnalyze(patch, sched, old_dw, new_dw);

      if(d_fracture) {
         scheduleSetPositions(patch,sched,old_dw,new_dw);
	 scheduleComputeBoundaryContact(patch,sched,old_dw,new_dw);
         scheduleComputeConnectivity(patch,sched,old_dw,new_dw);
      }
      scheduleInterpolateParticlesToGrid(patch,sched,old_dw,new_dw);
      
      if (MPMPhysicalModules::thermalContactModel) {
         scheduleComputeHeatExchange(patch,sched,old_dw,new_dw);
      }
      scheduleExMomInterpolated(patch,sched,old_dw,new_dw);
      scheduleComputeStressTensor(patch,sched,old_dw,new_dw);
      scheduleComputeInternalForce(patch,sched,old_dw,new_dw);
      scheduleComputeInternalHeatRate(patch,sched,old_dw,new_dw);
      scheduleSolveEquationsMotion(patch,sched,old_dw,new_dw);
      scheduleSolveHeatEquations(patch,sched,old_dw,new_dw);
      scheduleIntegrateAcceleration(patch,sched,old_dw,new_dw);
      scheduleIntegrateTemperatureRate(patch,sched,old_dw,new_dw);
      scheduleExMomIntegrated(patch,sched,old_dw,new_dw);
      scheduleInterpolateToParticlesAndUpdate(patch,sched,old_dw,new_dw);
      scheduleComputeMassRate(patch,sched,old_dw,new_dw);

      if(d_fracture) {
         scheduleComputeFracture(patch,sched,old_dw,new_dw);
      }
      scheduleCarryForwardVariables(patch,sched,old_dw,new_dw);

#if 0
      if(t + dt >= d_nextOutputTime) {
	scheduleInterpolateParticlesForSaving(patch,sched,old_dw,new_dw);
      }
#endif
    }
    
    if(t + dt >= d_nextOutputTime)
    {
	d_nextOutputTime += d_outputInterval;
    }

   int numMatls = d_sharedState->getNumMPMMatls();

   sched->scheduleParticleRelocation(level, old_dw, new_dw,
				     lb->pXLabel_preReloc, 
				     lb->d_particleState_preReloc,
				     lb->pXLabel, lb->d_particleState,
				     numMatls);
}

void SerialMPM::scheduleSetPositions(const Patch* patch,
					      SchedulerP& sched,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task( "SerialMPM::setPositions",
			  patch, old_dw, new_dw,
			  this,&SerialMPM::setPositions);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
    int idx = matl->getDWIndex();
    t->requires(old_dw, lb->pXLabel, idx, patch, Ghost::None);
    t->computes(new_dw, lb->pXXLabel, idx, patch);
  }
  sched->addTask(t);
}

void SerialMPM::scheduleComputeBoundaryContact(const Patch* patch,
                                               SchedulerP& sched,
					       DataWarehouseP& old_dw,
					       DataWarehouseP& new_dw)
{
 /*
  * computeBoundaryContact
  *   in(P.X, P.VOLUME, P.NEWISBROKEN, P.NEWCRACKSURFACENORMAL)
  *   operation(computeBoundaryContact)
  * out(P.STRESS) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task( "SerialMPM::computeBoundaryContact",
			  patch, old_dw, new_dw,
			  this,&SerialMPM::computeBoundaryContact);

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
    int idx = matl->getDWIndex();
    t->requires(old_dw, lb->pXLabel,
        idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pCrackNormal1Label,
	idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pCrackNormal2Label,
	idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pCrackNormal3Label,
	idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pIsBrokenLabel,
        idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pVolumeLabel,
        idx, patch, Ghost::AroundCells, 1);

    t->requires(new_dw, lb->pXXLabel, idx, patch, Ghost::None);

    t->computes(new_dw, lb->pContactNormalLabel, idx, patch);
  }
  sched->addTask(t);
}

void SerialMPM::scheduleComputeConnectivity(const Patch* patch,
					      SchedulerP& sched,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{
 /*
  * computeConnectivity
  *   in(P.X, P.VOLUME, P.ISBROKEN, P.CRACKSURFACENORMAL)
  *   operation(compute the visibility information of particles to the
  *   related nodes)
  * out(P.VISIBILITY) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task( "SerialMPM::computeConnectivity",
			  patch, old_dw, new_dw,
			  this,&SerialMPM::computeConnectivity);

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
    int idx = matl->getDWIndex();
    t->requires(old_dw, lb->pXLabel,        idx, patch, Ghost::AroundCells, 1);
    t->requires(new_dw, lb->pXXLabel,       idx, patch, Ghost::None);
    t->requires(old_dw, lb->pVolumeLabel,   idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pIsBrokenLabel, 
       idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pCrackNormal1Label,
       idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pCrackNormal2Label,
       idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pCrackNormal3Label,
       idx, patch, Ghost::AroundCells, 1);
    t->requires(new_dw, lb->pContactNormalLabel,
       idx, patch, Ghost::AroundCells, 1);

    t->computes(new_dw, lb->pConnectivityLabel, idx, patch);
  }
  sched->addTask(t);
}

void SerialMPM::scheduleInterpolateParticlesToGrid(const Patch* patch,
						   SchedulerP& sched,
						   DataWarehouseP& old_dw,
						   DataWarehouseP& new_dw)
{
  /* interpolateParticlesToGrid
   *   in(P.MASS, P.VELOCITY, P.NAT_X)
   *   operation(interpolate the P.MASS and P.VEL to the grid
   *             using P.NAT_X and some shape function evaluations)
   *   out(G.MASS, G.VELOCITY) */


  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("SerialMPM::interpolateParticlesToGrid",
		    patch, old_dw, new_dw,
		    this,&SerialMPM::interpolateParticlesToGrid);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    int idx = mpm_matl->getDWIndex();
    t->requires(old_dw, lb->pMassLabel,         idx,patch,Ghost::AroundNodes,1);
    t->requires(old_dw, lb->pVolumeLabel,       idx,patch,Ghost::AroundNodes,1);
    
    t->requires(old_dw, lb->pVelocityLabel,     idx,patch,Ghost::AroundNodes,1);
    t->requires(old_dw, lb->pXLabel,            idx,patch,Ghost::AroundNodes,1);
    t->requires(old_dw, lb->pExternalForceLabel,idx,patch,Ghost::AroundNodes,1);
    t->requires(old_dw, lb->pTemperatureLabel,  idx,patch,Ghost::AroundNodes,1);
//    t->requires(old_dw, lb->pExternalHeatRateLabel,
//						idx,patch,Ghost::AroundNodes,1);

    t->computes(new_dw, lb->gMassLabel,            idx, patch);
    t->computes(new_dw, lb->gVolumeLabel,          idx, patch);
    t->computes(new_dw, lb->gVelocityLabel,        idx, patch);
    t->computes(new_dw, lb->gExternalForceLabel,   idx, patch);
    t->computes(new_dw, lb->gTemperatureLabel,     idx, patch);
    t->computes(new_dw, lb->gExternalHeatRateLabel,idx, patch);

    if(mpm_matl->getFractureModel()) {
       t->requires(new_dw,lb->pContactNormalLabel,idx, patch,
		Ghost::AroundNodes, 1 );
       t->requires(new_dw, lb->pConnectivityLabel, idx, patch,
		Ghost::AroundNodes, 1 );
       t->computes(new_dw, lb->gMassContactLabel,  idx, patch);
    }
  }
     
  t->computes(new_dw, lb->gMassLabel,            numMatls, patch);
  t->computes(new_dw, lb->TotalMassLabel);
  sched->addTask(t);
}

void SerialMPM::scheduleComputeHeatExchange(const Patch* patch,
					    SchedulerP& sched,
					    DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw)
{
 /* computeHeatExchange
  *   in(G.MASS, G.TEMPERATURE, G.EXTERNAL_HEAT_RATE)
  *   operation(peform heat exchange which will cause each of
  *   velocity fields to exchange heat according to 
  *   the temperature differences)
  *   out(G.EXTERNAL_HEAT_RATE) */


  int numMatls = d_sharedState->getNumMPMMatls();
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

void SerialMPM::scheduleExMomInterpolated(const Patch* patch,
					  SchedulerP& sched,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw)
{
  Task* t = scinew Task("Contact::exMomInterpolated",
		    patch, old_dw, new_dw,
		    MPMPhysicalModules::contactModel,
		    &Contact::exMomInterpolated);

  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
	MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
	MPMPhysicalModules::contactModel->
	addComputesAndRequiresInterpolated(t, mpm_matl, patch, old_dw, new_dw);
  }
  sched->addTask(t);
}

void SerialMPM::scheduleComputeStressTensor(const Patch* patch,
					    SchedulerP& sched,
					    DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("SerialMPM::computeStressTensor",
		    patch, old_dw, new_dw,
		    this, &SerialMPM::computeStressTensor);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patch, old_dw, new_dw);
  }
	 
  t->computes(new_dw, d_sharedState->get_delt_label());
  t->computes(new_dw, lb->StrainEnergyLabel);

  sched->addTask(t);
}

void SerialMPM::scheduleComputeInternalForce(const Patch* patch,
					     SchedulerP& sched,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw)
{
 /*
  * computeInternalForce
  *   in(P.CONMOD, P.NAT_X, P.VOLUME)
  *   operation(evaluate the divergence of the stress (stored in
  *   P.CONMOD) using P.NAT_X and the gradients of the
  *   shape functions)
  * out(G.F_INTERNAL) */

  int numMPMMatls = d_sharedState->getNumMPMMatls();
  int numALLMatls = d_sharedState->getNumMatls();

  Task* t = scinew Task("SerialMPM::computeInternalForce",
		    patch, old_dw, new_dw,
		    this, &SerialMPM::computeInternalForce);

  t->requires( new_dw, lb->gMassLabel, numMPMMatls, patch, Ghost::None);
  for(int m = 0; m < numMPMMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    int idx = mpm_matl->getDWIndex();
	    
    t->requires( new_dw, lb->pStressAfterStrainRateLabel, idx, patch,
			 Ghost::AroundNodes, 1);
    t->requires( new_dw, lb->pVolumeDeformedLabel, idx, patch,
			 Ghost::AroundNodes, 1);
    t->requires( old_dw, lb->pMassLabel, idx, patch, Ghost::AroundNodes, 1);
    t->requires( new_dw, lb->gMassLabel, idx, patch, Ghost::None);

    if(numMPMMatls!=numALLMatls){
      t->requires(new_dw, lb->pPressureLabel, idx, patch,Ghost::AroundNodes, 1);
    }

    if(mpm_matl->getFractureModel()) {
       t->requires(new_dw, lb->pConnectivityLabel, idx, patch,
			Ghost::AroundNodes, 1 );
       t->requires(new_dw, lb->pContactNormalLabel, idx, patch,
			Ghost::AroundNodes, 1 );
    }

    t->computes(new_dw, lb->gInternalForceLabel,   idx, patch);
    t->computes(new_dw, lb->gStressForSavingLabel, idx, patch);
  }
  t->computes(new_dw, lb->NTractionZMinusLabel);
  t->computes(new_dw, lb->gStressForSavingLabel, numMPMMatls, patch);

  sched->addTask(t);
}

void SerialMPM::scheduleComputeInternalHeatRate(const Patch* patch,
					        SchedulerP& sched,
					        DataWarehouseP& old_dw,
					        DataWarehouseP& new_dw)
{  
  /*
   * computeInternalHeatRate
   * in(P.X, P.VOLUME, P.TEMPERATUREGRADIENT)
   * operation(evaluate the grid internal heat rate using 
   *   P.TEMPERATUREGRADIENT and the gradients of the shape functions)
   * out(G.INTERNALHEATRATE) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("SerialMPM::computeInternalHeatRate",
		    patch, old_dw, new_dw,
		    this, &SerialMPM::computeInternalHeatRate);

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    int idx = mpm_matl->getDWIndex();

    t->requires(old_dw, lb->pXLabel, idx, patch, Ghost::AroundNodes, 1 );
    t->requires(new_dw, lb->pVolumeDeformedLabel,
				     idx, patch, Ghost::AroundNodes, 1 );
    t->requires(old_dw, lb->pTemperatureGradientLabel,
				     idx, patch, Ghost::AroundNodes, 1);

    if(mpm_matl->getFractureModel()) {
      t->requires(new_dw, lb->pConnectivityLabel, idx,patch,Ghost::AroundNodes,1);
    }

    t->computes( new_dw, lb->gInternalHeatRateLabel, idx, patch );
  }

  sched->addTask(t);
}

void SerialMPM::scheduleSolveEquationsMotion(const Patch* patch,
					     SchedulerP& sched,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw)
{
  /* solveEquationsMotion
   *   in(G.MASS, G.F_INTERNAL)
   *   operation(acceleration = f/m)
   *   out(G.ACCELERATION) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("SerialMPM::solveEquationsMotion",
		    patch, old_dw, new_dw,
		    this, &SerialMPM::solveEquationsMotion);

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    int idx = mpm_matl->getDWIndex();
    
    if(mpm_matl->getFractureModel())
      t->requires( new_dw, lb->gMassContactLabel,   idx, patch, Ghost::None);
    else
      t->requires( new_dw, lb->gMassLabel,          idx, patch, Ghost::None);
      
    t->requires( new_dw, lb->gInternalForceLabel, idx, patch, Ghost::None);
    t->requires( new_dw, lb->gExternalForceLabel, idx, patch, Ghost::None);
    if(d_sharedState->getNumMatls() != d_sharedState->getNumMPMMatls()){
        t->requires( new_dw, lb->gradPressNCLabel,idx, patch, Ghost::None);
    }

    t->computes( new_dw, lb->gAccelerationLabel,  idx, patch);
  }

  sched->addTask(t);
}

void SerialMPM::scheduleSolveHeatEquations(const Patch* patch,
					   SchedulerP& sched,
					   DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
  /* solveHeatEquations
   *   in(G.MASS, G.INTERNALHEATRATE, G.EXTERNALHEATRATE)
   *   out(G.TEMPERATURERATE) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("SerialMPM::solveHeatEquations",
			    patch, old_dw, new_dw,
			    this, &SerialMPM::solveHeatEquations);

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    int idx = mpm_matl->getDWIndex();

    if(mpm_matl->getFractureModel())
      t->requires( new_dw, lb->gMassContactLabel,   idx, patch, Ghost::None);
    else
      t->requires( new_dw, lb->gMassLabel,          idx, patch, Ghost::None);

    t->requires( new_dw, lb->gVolumeLabel,           idx, patch, Ghost::None);
    t->requires( new_dw, lb->gInternalHeatRateLabel, idx, patch, Ghost::None);
    t->requires( new_dw, lb->gExternalHeatRateLabel, idx, patch, Ghost::None);

    if(MPMPhysicalModules::thermalContactModel) {
       t->requires(new_dw, lb->gThermalContactHeatExchangeRateLabel,
						     idx, patch, Ghost::None);
    }
		
    t->computes( new_dw, lb->gTemperatureRateLabel, idx, patch);
  }

  sched->addTask(t);
}

void SerialMPM::scheduleIntegrateAcceleration(const Patch* patch,
					      SchedulerP& sched,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{
  /* integrateAcceleration
   *   in(G.ACCELERATION, G.VELOCITY)
   *   operation(v* = v + a*dt)
   *   out(G.VELOCITY_STAR) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("SerialMPM::integrateAcceleration",
			    patch, old_dw, new_dw,
			    this, &SerialMPM::integrateAcceleration);

  t->requires(old_dw, d_sharedState->get_delt_label() );

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
    int idx = matl->getDWIndex();
    t->requires(new_dw, lb->gAccelerationLabel,    idx, patch, Ghost::None);
    t->requires(new_dw, lb->gMomExedVelocityLabel, idx, patch, Ghost::None);

    t->computes(new_dw, lb->gVelocityStarLabel, idx, patch );
  }
		     
  sched->addTask(t);
}

void SerialMPM::scheduleIntegrateTemperatureRate(const Patch* patch,
					         SchedulerP& sched,
					         DataWarehouseP& old_dw,
					         DataWarehouseP& new_dw)
{
  /* integrateTemperatureRate
   *   in(G.TEMPERATURE, G.TEMPERATURERATE)
   *   operation(t* = t + t_rate * dt)
   *   out(G.TEMPERATURE_STAR) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("SerialMPM::integrateTemperatureRate",
		    patch, old_dw, new_dw,
		    this, &SerialMPM::integrateTemperatureRate);

  t->requires( old_dw, d_sharedState->get_delt_label() );

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
    int idx = matl->getDWIndex();

    t->requires( new_dw, lb->gTemperatureLabel,     idx, patch, Ghost::None);
    t->requires( new_dw, lb->gTemperatureRateLabel, idx, patch, Ghost::None);
		     
    t->computes( new_dw, lb->gTemperatureStarLabel, idx, patch );
  }
		     
  sched->addTask(t);
}

void SerialMPM::scheduleExMomIntegrated(const Patch* patch,
				        SchedulerP& sched,
				        DataWarehouseP& old_dw,
				        DataWarehouseP& new_dw)
{
  /* exMomIntegrated
   *   in(G.MASS, G.VELOCITY_STAR, G.ACCELERATION)
   *   operation(peform operations which will cause each of
   *		  velocity fields to feel the influence of the
   *		  the others according to specific rules)
   *   out(G.VELOCITY_STAR, G.ACCELERATION) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("Contact::exMomIntegrated",
		   patch, old_dw, new_dw,
		   MPMPhysicalModules::contactModel,
		   &Contact::exMomIntegrated);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    MPMPhysicalModules::contactModel->addComputesAndRequiresIntegrated(
					t, mpm_matl, patch, old_dw, new_dw);
  }

  sched->addTask(t);
}

void SerialMPM::scheduleInterpolateToParticlesAndUpdate(const Patch* patch,
						        SchedulerP& sched,
						        DataWarehouseP& old_dw,
						        DataWarehouseP& new_dw)

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
		    patch, old_dw, new_dw,
		    this, &SerialMPM::interpolateToParticlesAndUpdate);


  t->requires(old_dw, d_sharedState->get_delt_label() );

  for(int m = 0; m < numMPMMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    int idx = mpm_matl->getDWIndex();
    t->requires(new_dw, lb->gMomExedAccelerationLabel,idx, patch,
			Ghost::AroundCells, 1);
    t->requires(new_dw, lb->gMomExedVelocityStarLabel,idx, patch,
			Ghost::AroundCells, 1);
    t->requires(new_dw, lb->gTemperatureRateLabel,    idx, patch,
			Ghost::AroundCells, 1);
    t->requires(new_dw, lb->gTemperatureLabel,        idx, patch,
			Ghost::AroundCells, 1);
    t->requires(new_dw, lb->gTemperatureStarLabel,    idx, patch,
			Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pXLabel,              idx, patch, Ghost::None);
    t->requires(old_dw, lb->pExternalForceLabel,  idx, patch, Ghost::None);
    t->requires(old_dw, lb->pMassLabel,           idx, patch, Ghost::None);
    t->requires(old_dw, lb->pParticleIDLabel,     idx, patch, Ghost::None);
    t->requires(old_dw, lb->pTemperatureLabel,    idx, patch, Ghost::None);
    t->requires(old_dw, lb->pVelocityLabel,       idx, patch, Ghost::None);

    if(numMPMMatls!=numALLMatls){
      t->requires(new_dw, lb->dTdt_NCLabel, idx, patch, Ghost::AroundCells,1);
    }

    if(mpm_matl->getFractureModel()) {
       t->requires(new_dw, lb->pConnectivityLabel,    idx, patch, Ghost::None);
       t->requires(new_dw, lb->pContactNormalLabel,   idx, patch, Ghost::None);
    }

    t->computes(new_dw, lb->pVelocityLabel_preReloc,            idx, patch);
    t->computes(new_dw, lb->pXLabel_preReloc,                   idx, patch);
    t->computes(new_dw, lb->pExternalForceLabel_preReloc,       idx, patch);
    t->computes(new_dw, lb->pParticleIDLabel_preReloc,          idx, patch);
    t->computes(new_dw, lb->pTemperatureRateLabel_preReloc,     idx, patch);
    t->computes(new_dw, lb->pTemperatureLabel_preReloc,         idx, patch);
    t->computes(new_dw, lb->pTemperatureGradientLabel_preReloc, idx, patch);
  }

  t->computes(new_dw, lb->KineticEnergyLabel);
  t->computes(new_dw, lb->CenterOfMassPositionLabel);
  t->computes(new_dw, lb->CenterOfMassVelocityLabel);
  sched->addTask(t);
}

void SerialMPM::scheduleComputeMassRate(const Patch* patch,
				        SchedulerP& sched,
				        DataWarehouseP& old_dw,
				        DataWarehouseP& new_dw)
{
 /* computeMassRate
  * in(P.TEMPERATURE_RATE)
  * operation(based on the heat flux history, determine if
  * each of the particles has ignited, adjust the mass of those
  * particles which are burning)
  * out(P.IGNITED) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task *t = scinew Task("SerialMPM::computeMassRate",
                         patch, old_dw, new_dw,
                         this, &SerialMPM::computeMassRate);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    HEBurn* heb = mpm_matl->getHEBurnModel();
    heb->addComputesAndRequires(t, mpm_matl, patch, old_dw, new_dw);
    if(!d_burns){
      d_burns=heb->getBurns();
    }
  }
  sched->addTask(t);
}

void SerialMPM::scheduleComputeFracture(const Patch* patch,
					      SchedulerP& sched,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{
 /*
  * computeFracture
  *   in(P.X, P.VOLUME, P.ISBROKEN, P.CRACKSURFACENORMAL)
  *   operation(compute the visibility information of particles to the
  *   related nodes)
  * out(P.VISIBILITY) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task( "SerialMPM::computeFracture",
			  patch, old_dw, new_dw,
			  this,&SerialMPM::computeFracture);

  t->requires(old_dw, d_sharedState->get_delt_label() );

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
    int idx = matl->getDWIndex();
    t->requires(old_dw, lb->pXLabel,        idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pIsBrokenLabel, idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pCrackNormal1Label, idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pCrackNormal2Label, idx, patch, Ghost::AroundCells, 1);
    t->requires(old_dw, lb->pCrackNormal3Label, idx, patch, Ghost::AroundCells, 1);

    t->requires(new_dw, lb->pXXLabel,       idx, patch, Ghost::None);
    t->requires(old_dw, lb->pVolumeLabel,   idx, patch, Ghost::None);
    t->requires(new_dw, lb->pStressAfterStrainRateLabel, idx, patch, Ghost::None);
    t->requires(new_dw, lb->pStrainEnergyLabel, idx, patch, Ghost::None);
    t->requires(old_dw, lb->pToughnessLabel, idx, patch, Ghost::None);
    t->requires(new_dw, lb->pRotationRateLabel, idx, patch, Ghost::None);
    t->requires(new_dw, lb->pConnectivityLabel, idx, patch, Ghost::None);

    t->requires(new_dw, lb->gStressForSavingLabel, idx, patch,
			Ghost::AroundCells, 1);

    t->computes(new_dw, lb->pStressAfterFractureReleaseLabel, idx, patch);
    t->computes(new_dw, lb->pIsBrokenLabel_preReloc, idx, patch);
    t->computes(new_dw, lb->pCrackNormal1Label_preReloc, idx, patch);
    t->computes(new_dw, lb->pCrackNormal2Label_preReloc, idx, patch);
    t->computes(new_dw, lb->pCrackNormal3Label_preReloc, idx, patch);
    t->computes(new_dw, lb->pToughnessLabel_preReloc, idx, patch);
  }
  sched->addTask(t);
}

void SerialMPM::scheduleCarryForwardVariables(const Patch* patch,
					      SchedulerP& sched,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{
  /* carryForwardVariables
   * in(p.x,p.stressBeforeFractureRelease,p.isNewlyBroken,
   *   p.crackSurfaceNormal)
   *   operation(check the stress on each particle to see
   *   if the microcrack will initiate and/or grow)
   * out(p.stress) */

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("SerialMPM::carryForwardVariables",
		         patch, old_dw, new_dw,
		         this,&SerialMPM::carryForwardVariables);

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    int idx = mpm_matl->getDWIndex();

    if(d_fracture) {
      t->requires( new_dw, lb->pStressAfterFractureReleaseLabel,
		idx, patch, Ghost::None);
    }
    else {
      t->requires( new_dw, lb->pStressAfterStrainRateLabel,
		idx, patch, Ghost::None);			 
    }

    t->computes(new_dw, lb->pStressLabel_preReloc,   idx, patch);
  }
  sched->addTask(t);
}

void SerialMPM::scheduleInterpolateParticlesForSaving(const Patch* patch,
					 	      SchedulerP& sched,
						      DataWarehouseP& old_dw,
						      DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task *t = scinew Task("SerialMPM::interpolateParticlesForSaving",
                         patch, old_dw, new_dw,
                         this, &SerialMPM::interpolateParticlesForSaving);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    int idx = mpm_matl->getDWIndex();
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
  sched->addTask(t);
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
       mpm_matl->getHEBurnModel()->initializeBurnModelData(patch,
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


void SerialMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
					      const Patch*,
					      DataWarehouseP&,
					      DataWarehouseP&)
{
}

void SerialMPM::computeConnectivity(
                   const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    mpm_matl->getFractureModel()->computeConnectivity(
	  patch, mpm_matl, old_dw, new_dw);
  }
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorGroup*,
					   const Patch* patch,
					   DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
  static Vector zero(0.,0.,0.);
  
  int numMatls = d_sharedState->getNumMPMMatls();
  int numALLMatls = d_sharedState->getNumMatls();

  NCVariable<double> totalgmass;
  new_dw->allocate(totalgmass,lb->gMassLabel,numALLMatls, patch);
  totalgmass.initialize(0);

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle data
      ParticleVariable<Point> px;
      ParticleVariable<double> pmass;
      ParticleVariable<double> pvolume;
      ParticleVariable<Vector> pvelocity;
      ParticleVariable<Vector> pexternalforce;
      ParticleVariable<double> pTemperature;

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

      new_dw->allocate(gmass,  lb->gMassLabel,        matlindex, patch);
      new_dw->allocate(gvolume,lb->gVolumeLabel,      matlindex, patch);
      new_dw->allocate(gvelocity,lb->gVelocityLabel,    matlindex, patch);
      new_dw->allocate(gTemperature, lb->gTemperatureLabel, matlindex, patch);
      new_dw->allocate(gexternalforce,lb->gExternalForceLabel,matlindex, 
		       patch);
      new_dw->allocate(gexternalheatrate, lb->gExternalHeatRateLabel,
		       matlindex, patch);

      gmass.initialize(0);
      gvolume.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);
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
      gmassContact.initialize(0);

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
	         conn[k] == Connectivity::contact) 
             {
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
	       gvelocity[ni[k]]      += pvelocity[idx] * pmass[idx] * S_contact[k];
	     }
	     else if( conn[k] == Connectivity::contact ) {
	       gexternalforce[ni[k]] += pContactNormal[idx] * 
		 ( Dot(pContactNormal[idx],pexternalforce[idx]) * S_contact[k] );
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

   //    cout << "Particle momentum before intToGrid = " << total_mom << endl;

      for(NodeIterator iter = patch->getNodeIterator();	!iter.done(); iter++)
      {
         if(mpm_matl->getFractureModel()) {  // Do interpolation with fracture
	   if(gmassContact[*iter] >= 1.e-10) {
	      gvelocity[*iter] /= gmassContact[*iter];
              gTemperature[*iter] /= gmassContact[*iter];
	   }
	 }
         else {  // Do interpolation without fracture
	   if(gmass[*iter] >= 1.e-10) {
	      gvelocity[*iter] /= gmass[*iter];
              gTemperature[*iter] /= gmass[*iter];
	   }
	 }
      }

      // Apply grid boundary conditions to the velocity
      // before storing the data

      //      cout << "Patch id = " << patch->getID() << endl;
      IntVector offset = 
	patch->getInteriorCellLowIndex() - patch->getCellLowIndex();
      // cout << "offset = " << offset << endl;
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
	    if (bc->getKind() == "Dirichlet") {
	      //cout << "Velocity bc value = " << bc->getValue() << endl;
	      gvelocity.fillFace(face,bc->getValue(),offset);
	    }
	  }
	  if (bcs_type == "Symmetric") {
	     gvelocity.fillFaceNormal(face,offset);
	  }
	}
      }

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);

      new_dw->put(gmass,         lb->gMassLabel,          matlindex, patch);
      new_dw->put(gvolume,       lb->gVolumeLabel,        matlindex, patch);
      new_dw->put(gvelocity,     lb->gVelocityLabel,      matlindex, patch);
      new_dw->put(gexternalforce,lb->gExternalForceLabel, matlindex, patch);
      new_dw->put(gTemperature,  lb->gTemperatureLabel,   matlindex, patch);
      new_dw->put(gexternalheatrate,lb->gExternalHeatRateLabel,
							  matlindex, patch);

      if(mpm_matl->getFractureModel()) {
        new_dw->put(gmassContact, lb->gMassContactLabel,  matlindex, patch);
      }
  }
  new_dw->put(totalgmass,         lb->gMassLabel,          numALLMatls, patch);
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
      HEBurn* heb = mpm_matl->getHEBurnModel();
      heb->computeMassRate(patch, mpm_matl, old_dw, new_dw);
   }
}

void SerialMPM::setPositions( const ProcessorGroup*,
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw)
{
  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    int matlindex = mpm_matl->getDWIndex();
        
    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);

    ParticleVariable<Point> pX;
    ParticleVariable<Point> pXX;
    
    old_dw->get(pX, lb->pXLabel, pset);
    new_dw->allocate(pXX, lb->pXXLabel, pset);
    
    for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++)
    {
      pXX[*iter] = pX[*iter];
    }

    new_dw->put(pXX, lb->pXXLabel);
  }
}
	    
void SerialMPM::computeInternalForce(const ProcessorGroup*,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{
  static Vector zero(0.,0.,0.);

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
  new_dw->allocate(gstressglobal,lb->gStressForSavingLabel, numALLMatls, patch);

  for(int m = 0; m < numMPMMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle position, volume
      // and the constitutive model
      ParticleVariable<Point>   px;
      ParticleVariable<double>  pvol;
      ParticleVariable<double>  pmass;
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

      new_dw->get(pvol,   lb->pVolumeDeformedLabel, pset);
      new_dw->get(gmass,  lb->gMassLabel, matlindex, patch, Ghost::None, 0);
      new_dw->get(pstress,lb->pStressAfterStrainRateLabel, pset);

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
      ParticleVariable<int> pConnectivity;
      ParticleVariable<Vector> pContactNormal;
      new_dw->get(pConnectivity, lb->pConnectivityLabel, pset);
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
	 
         connectivity.modifyShapeDerivatives(conn,d_S_connect,Connectivity::connect);
      	 connectivity.modifyWeights(conn,S_connect,Connectivity::connect);
         connectivity.modifyShapeDerivatives(conn,d_S_contact,Connectivity::contact);
      	 connectivity.modifyWeights(conn,S_contact,Connectivity::contact);

         for(int k = 0; k < 8; k++) {
	   if( patch->containsNode(ni[k]) ) {
	     if( conn[k] == Connectivity::connect )
             {
               gstress[ni[k]] += pstress[idx] * pmass[idx] * S_connect[k];
               gstressglobal[ni[k]] += pstress[idx] * pmass[idx] * S_connect[k];
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
//			(div * (pstress[idx]) * pvol[idx]);
             gstress[ni[k]] += pstress[idx] * pmass[idx] * S[k];
             gstressglobal[ni[k]] += pstress[idx] * pmass[idx] * S[k];
	   }
         }
      }
    }

      for(NodeIterator iter = patch->getNodeIterator();
				!iter.done(); iter++) {
         if(gmass[*iter] >= 1.e-10){
            gstress[*iter] /= gmass[*iter];
         }
//       cout << "internalForce = " << internalforce[*iter] << endl;
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

  for(NodeIterator iter = patch->getNodeIterator();
      !iter.done(); iter++) {
    if(gmassglobal[*iter] >= 1.e-10){
      gstressglobal[*iter] /= gmassglobal[*iter];
    }
  }
  new_dw->put(gstressglobal,  lb->gStressForSavingLabel, numALLMatls, patch);
}

void SerialMPM::computeInternalHeatRate(const ProcessorGroup*,
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
	     internalHeatRate[ni[k]] -= Dot( div, pTemperatureGradient[idx] ) * 
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
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{
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
        new_dw->get(mass,  lb->gMassContactLabel, matlindex, patch, Ghost::None, 0);
      else
        new_dw->get(mass,  lb->gMassLabel, matlindex, patch, Ghost::None, 0);

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

      // Do the computation of a = F/m for nodes where m!=0.0
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	if(mass[*iter]>0.0){
	  acceleration[*iter] =
		(internalforce[*iter] + externalforce[*iter] +
				gradPressNC[*iter]/delT)/ mass[*iter] + gravity;
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
      NCVariable<double> mass,internalHeatRate,externalHeatRate,gvolume;
      NCVariable<double> thermalContactHeatExchangeRate;

      if(mpm_matl->getFractureModel())
        new_dw->get(mass,    lb->gMassContactLabel,   dwindex, patch, Ghost::None, 0);
      else
        new_dw->get(mass,    lb->gMassLabel,   dwindex, patch, Ghost::None, 0);
	
      new_dw->get(gvolume, lb->gVolumeLabel, dwindex, patch, Ghost::None, 0);
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
//              cout << "bc value = " << bc->getValue() << endl;
	      double value = bc->getValue();
	      IntVector offset = 
		patch->getInteriorCellLowIndex() - patch->getCellLowIndex();
              IntVector low = internalHeatRate.getLowIndex() + offset;
              IntVector hi = internalHeatRate.getHighIndex() - offset;
	     
              if(face==Patch::xplus || face==Patch::xminus){
                int I=-1234;
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
                int J=-1234;
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
                int K=-1234;
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
      new_dw->allocate(temperatureRate, lb->gTemperatureRateLabel,
         dwindex, patch);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	if(mass[*iter]>0.0){
	  temperatureRate[*iter] = (internalHeatRate[*iter]
		                 +  externalHeatRate[*iter]) /
				  (mass[*iter] * specificHeat);
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
    ParticleVariable<double> pmass;
    ParticleVariable<double> pToughness;
    ParticleVariable<double> pTemperature, pTemperatureNew, pTemperatureRate; 
    ParticleVariable<Vector> pTemperatureGradient; 

    // Get the arrays of grid data on which the new part. values depend
    NCVariable<Vector> gvelocity_star;
    NCVariable<Vector> gacceleration;
    NCVariable<double> gTemperatureRate,gTemperatureStar, gTemperature;
    NCVariable<double> dTdt;

    delt_vartype delT;

    ParticleSubset* pset = old_dw->getParticleSubset(dwindex, patch);
    
    old_dw->get(px,                       lb->pXLabel, pset);
    old_dw->get(pmass,                    lb->pMassLabel, pset);
    old_dw->get(pexternalForce,           lb->pExternalForceLabel, pset);
    old_dw->get(pTemperature,             lb->pTemperatureLabel, pset);
    old_dw->get(pvelocity,                lb->pVelocityLabel, pset);
    new_dw->allocate(pTemperatureNew,     lb->pTemperatureLabel_preReloc, pset);
    new_dw->allocate(pTemperatureRate,    lb->pTemperatureRateLabel, pset);
    new_dw->allocate(pTemperatureGradient,lb->pTemperatureGradientLabel, pset);
    new_dw->allocate(pvelocitynew,        lb->pVelocityLabel, pset);
    new_dw->allocate(pxnew,               lb->pXLabel_preReloc, pset);

    new_dw->get(gvelocity_star,   lb->gMomExedVelocityStarLabel,
			dwindex, patch, Ghost::AroundCells, 1);
    new_dw->get(gacceleration,    lb->gMomExedAccelerationLabel,
			dwindex, patch, Ghost::AroundCells, 1);
    new_dw->get(gTemperatureRate, lb->gTemperatureRateLabel,
			dwindex, patch, Ghost::AroundCells, 1);
    new_dw->get(gTemperatureStar, lb->gTemperatureStarLabel,
			dwindex, patch, Ghost::AroundCells, 1);
    new_dw->get(gTemperature,     lb->gTemperatureLabel,
			dwindex, patch, Ghost::AroundCells, 1);

    if(numMPMMatls!=numALLMatls){
      new_dw->get(dTdt, lb->dTdt_NCLabel, dwindex, patch, Ghost::AroundCells,1);
    }
    else{
      new_dw->allocate(dTdt, lb->dTdt_NCLabel,dwindex,patch,IntVector(1,1,1));
      dTdt.initialize(0.);
    }

    old_dw->get(delT, d_sharedState->get_delt_label() );

    double Cp=mpm_matl->getSpecificHeat();
    //double ThCnd = mpm_matl->getThermalConductivity();

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
	      gTemperatureStar.fillFace(face,bc->getValue(),offset);
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

    IntVector ni[8];
    
    if(mpm_matl->getFractureModel()) {
      ParticleVariable<int> pConnectivity;
      ParticleVariable<Vector> pContactNormal;
      new_dw->get(pConnectivity, lb->pConnectivityLabel, pset);
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
	 
        connectivity.modifyShapeDerivatives(conn,d_S_connect,Connectivity::connect);
      	connectivity.modifyWeights(conn,S_connect,Connectivity::connect);
        connectivity.modifyShapeDerivatives(conn,d_S_contact,Connectivity::contact);
      	connectivity.modifyWeights(conn,S_contact,Connectivity::contact);

	int numConnectedNodes = 0;

        vel = Vector(0.0,0.0,0.0);
        acc = Vector(0.0,0.0,0.0);

        pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
        tempRate = 0;
	
        // Accumulate the contribution from each surrounding vertex
        for(int k = 0; k < 8; k++) {
	     if( conn[k] == Connectivity::connect || 
	         conn[k] == Connectivity::contact) 
             {
                tempRate += (gTemperatureRate[ni[k]] + dTdt[ni[k]])
							* S_connect[k];
                for (int j = 0; j<3; j++) {
                  pTemperatureGradient[idx](j) += 
                    gTemperatureStar[ni[k]] * d_S_connect[k](j) * oodx[j];
                }
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
        pTemperatureRate[idx] = tempRate;
        pTemperatureNew[idx] = pTemperature[idx] + tempRate * delT;
        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
	
	if(numConnectedNodes != 0) {
          pxnew[idx]      = px[idx] + vel * delT;
          pvelocitynew[idx] = pvelocity[idx] + acc * delT;
        }
	else {        
   	  //for isolated particles in fracture
          pxnew[idx]      =  px[idx] + pvelocity[idx] * delT;
          pvelocitynew[idx] = pvelocity[idx] +
	     pexternalForce[idx] / (pmass[idx] * delT);
        }

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

        pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
        tempRate = 0;

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < 8; k++) {
	      vel += gvelocity_star[ni[k]]  * S[k];
   	      acc += gacceleration[ni[k]]   * S[k];
	   
              tempRate += (gTemperatureRate[ni[k]] + dTdt[ni[k]]) * S[k];
              for (int j = 0; j<3; j++) {
                pTemperatureGradient[idx](j) += 
                   gTemperatureStar[ni[k]] * d_S[k](j) * oodx[j];
              }
        }

        // Update the particle's position and velocity
        pxnew[idx]      = px[idx] + vel * delT;
        pvelocitynew[idx] = pvelocity[idx] + acc * delT;
        pTemperatureRate[idx] = tempRate;
        pTemperatureNew[idx] = pTemperature[idx] + tempRate * delT;

        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
        ke += .5*pmass[idx]*pvelocitynew[idx].length2();
	CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
	CMV += pvelocitynew[idx]*pmass[idx];
      }
    }

      // Store the new result
      new_dw->put(pxnew,          lb->pXLabel_preReloc);
      new_dw->put(pvelocitynew,   lb->pVelocityLabel_preReloc);
      new_dw->put(pexternalForce, lb->pExternalForceLabel_preReloc);

      ParticleVariable<long> pids;
      old_dw->get(pids, lb->pParticleIDLabel, pset);
      new_dw->put(pids, lb->pParticleIDLabel_preReloc);

      new_dw->put(pTemperatureRate,     lb->pTemperatureRateLabel_preReloc);
      new_dw->put(pTemperatureNew,      lb->pTemperatureLabel_preReloc);
      new_dw->put(pTemperatureGradient, lb->pTemperatureGradientLabel_preReloc);
  }
  // DON'T MOVE THESE!!!
  new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
  new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
  new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);

  cout << "Solid momentum after advection = " << CMV << endl;

//  cout << "THERMAL ENERGY " << thermal_energy << endl;
}

void SerialMPM::computeFracture(
                   const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    mpm_matl->getFractureModel()->computeFracture(
	  patch, mpm_matl, old_dw, new_dw);
  }
}

void SerialMPM::computeBoundaryContact(
                   const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    mpm_matl->getFractureModel()->
       computeBoundaryContact(patch, mpm_matl, old_dw, new_dw);
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

    new_dw->put(pStress, lb->pStressLabel_preReloc);
  }
}
	    
void SerialMPM::interpolateParticlesForSaving(const ProcessorGroup*,
				    	      const Patch*,
					      DataWarehouseP&,
					      DataWarehouseP&)
{
#if 0
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
	 
		 patch->findCellAndWeights(px[idx], ni, S);

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
	 
		 patch->findCellAndWeights(px[idx], ni, S);

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
	 
		 patch->findCellAndWeights(px[idx], ni, S);

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
#endif
}
