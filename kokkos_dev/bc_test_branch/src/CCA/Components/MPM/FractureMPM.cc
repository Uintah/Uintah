/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/FractureMPM.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Contact/Contact.h>
#include <CCA/Components/MPM/Contact/ContactFactory.h>
#include <CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/HeatConduction/HeatConduction.h>
#include <CCA/Components/Regridder/PerPatchVars.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <iostream>
#include <fstream>

#undef KUMAR
//#define KUMAR

using namespace Uintah;

using namespace std;

static DebugStream cout_doing("MPM", false);
static DebugStream cout_dbg("FractureMPM", false);
static DebugStream cout_convert("MPMConv", false);
static DebugStream cout_heat("MPMHeat", false);
static DebugStream amr_doing("AMRMPM", false);

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

static Vector face_norm(Patch::FaceType f)
{
  switch(f) {
  case Patch::xminus: return Vector(-1,0,0);
  case Patch::xplus:  return Vector( 1,0,0);
  case Patch::yminus: return Vector(0,-1,0);
  case Patch::yplus:  return Vector(0, 1,0);
  case Patch::zminus: return Vector(0,0,-1);
  case Patch::zplus:  return Vector(0,0, 1);
  default:
    return Vector(0,0,0); // oops !
  }
}

FractureMPM::FractureMPM(const ProcessorGroup* myworld) :
  SerialMPM(myworld)
{
  crackModel          = 0;
}

FractureMPM::~FractureMPM()
{
  delete crackModel;
}

void FractureMPM::problemSetup(const ProblemSpecP& prob_spec, 
                               const ProblemSpecP& restart_prob_spec,GridP& grid,
                               SimulationStateP& sharedState)
{
  SerialMPM::problemSetup(prob_spec,restart_prob_spec,grid,sharedState);

  // for FractureMPM
  dataArchiver = dynamic_cast<Output*>(getPort("output"));
  crackModel =  scinew Crack(prob_spec,sharedState,dataArchiver,lb,flags);

}

void
FractureMPM::materialProblemSetup(const ProblemSpecP& prob_spec,
                                  SimulationStateP& sharedState,
                                  MPMFlags* flags)
{
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps =  
    prob_spec->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {

    //Create and register as an MPM material
    MPMMaterial *mat = scinew MPMMaterial(ps, sharedState, flags);
    sharedState->registerMPMMaterial(mat);

    // If new particles are to be created, create a copy of each material
    // without the associated geometry
    if (flags->d_createNewParticles) {
      MPMMaterial *mat_copy = scinew MPMMaterial();
      mat_copy->copyWithoutGeom(ps,mat, flags);
      sharedState->registerMPMMaterial(mat_copy);
    }
  }
}

void FractureMPM::scheduleInitialize(const LevelP& level,
                                     SchedulerP& sched)
{
  Task* t = scinew Task("FractureMPM::actuallyInitialize",
                        this, &FractureMPM::actuallyInitialize);

  MaterialSubset* zeroth_matl = scinew MaterialSubset();
  zeroth_matl->add(0);
  zeroth_matl->addReference();

  t->computes(lb->partCountLabel);
  t->computes(lb->pXLabel);
  t->computes(lb->pDispLabel);
  t->computes(lb->pMassLabel);
  t->computes(lb->pFiberDirLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pTempPreviousLabel); // for thermal stress
  t->computes(lb->pdTdtLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pDispGradsLabel);
  t->computes(lb->pStrainEnergyDensityLabel);
  t->computes(d_sharedState->get_delt_label(),level.get_rep());
  t->computes(lb->pCellNAPIDLabel,zeroth_matl);

  // Debugging Scalar
  if (flags->d_with_color) {
    t->computes(lb->pColorLabel);
  }

  if (flags->d_useLoadCurves) {
    // Computes the load curve ID associated with each particle
    t->computes(lb->pLoadCurveIDLabel);
  }

  if (flags->d_reductionVars->accStrainEnergy) {
    // Computes accumulated strain energy
    t->computes(lb->AccStrainEnergyLabel);
  }

  // artificial damping coeff initialized to 0.0
  if (cout_dbg.active())
    cout_doing << "Artificial Damping Coeff = " << flags->d_artificialDampCoeff 
               << " 8 or 27 = " << flags->d_8or27 << endl;

  int numMPM = d_sharedState->getNumMPMMatls();
  const PatchSet* patches = level->eachPatch();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
  }

  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  schedulePrintParticleCount(level,sched);
    
  // for FractureMPM: Descritize crack plane into triangular elements
  t = scinew Task("Crack:CrackDiscretization",
                   crackModel, &Crack::CrackDiscretization);
  crackModel->addComputesAndRequiresCrackDiscretization(t,
                  level->eachPatch(), d_sharedState->allMPMMaterials());
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...

  if (flags->d_useLoadCurves) {
    // Schedule the initialization of pressure BCs per particle
    scheduleInitializePressureBCs(level, sched);
  }
}

void FractureMPM::scheduleInitializeAddedMaterial(const LevelP& level,
                                                SchedulerP& sched)
{
  if (cout_doing.active())
    cout_doing << "Doing FractureMPM::scheduleInitializeAddedMaterial " << endl;

  Task* t = scinew Task("FractureMPM::actuallyInitializeAddedMaterial",
                  this, &FractureMPM::actuallyInitializeAddedMaterial);

  int numALLMatls = d_sharedState->getNumMatls();
  int numMPMMatls = d_sharedState->getNumMPMMatls();
  MaterialSubset* add_matl = scinew MaterialSubset();
  cout << "Added Material = " << numALLMatls-1 << endl;
  add_matl->add(numALLMatls-1);
  add_matl->addReference();

  t->computes(lb->partCountLabel,          add_matl);
  t->computes(lb->pXLabel,                 add_matl);
  t->computes(lb->pDispLabel,              add_matl);
  t->computes(lb->pMassLabel,              add_matl);
  t->computes(lb->pVolumeLabel,            add_matl);
  t->computes(lb->pTemperatureLabel,       add_matl);
  t->computes(lb->pTempPreviousLabel,      add_matl); // for thermal stress
  t->computes(lb->pdTdtLabel,  add_matl);
  t->computes(lb->pVelocityLabel,          add_matl);
  t->computes(lb->pExternalForceLabel,     add_matl);
  t->computes(lb->pParticleIDLabel,        add_matl);
  t->computes(lb->pDeformationMeasureLabel,add_matl);
  t->computes(lb->pStressLabel,            add_matl);
  t->computes(lb->pSizeLabel,              add_matl);

  if (flags->d_reductionVars->accStrainEnergy) {
    // Computes accumulated strain energy
    t->computes(lb->AccStrainEnergyLabel);
  }

  const PatchSet* patches = level->eachPatch();

  MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(numMPMMatls-1);
  ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
  cm->addInitialComputesAndRequires(t, mpm_matl, patches);

  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  // The task will have a reference to add_matl
  if (add_matl->removeReference())
    delete add_matl; // shouln't happen, but...
}

void FractureMPM::scheduleInitializePressureBCs(const LevelP& level,
                                              SchedulerP& sched)
{
  MaterialSubset* loadCurveIndex = scinew MaterialSubset();
  int nofPressureBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure") loadCurveIndex->add(nofPressureBCs++);
  }
  if (nofPressureBCs > 0) {

    // Create a task that calculates the total number of particles
    // associated with each load curve.
    Task* t = scinew Task("FractureMPM::countMaterialPointsPerLoadCurve",
                          this, &FractureMPM::countMaterialPointsPerLoadCurve);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel, Ghost::None);
    t->computes(lb->materialPointsPerLoadCurveLabel, loadCurveIndex,
                Task::OutOfDomain);
    sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

    // Create a task that calculates the force to be associated with
    // each particle based on the pressure BCs
    t = scinew Task("FractureMPM::initializePressureBC",
                    this, &FractureMPM::initializePressureBC);
    t->requires(Task::NewDW, lb->pXLabel, Ghost::None);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel, Ghost::None);
    t->requires(Task::NewDW, lb->materialPointsPerLoadCurveLabel, loadCurveIndex, Task::OutOfDomain, Ghost::None);
    t->modifies(lb->pExternalForceLabel);
    sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
  }
}

void FractureMPM::scheduleComputeStableTimestep(const LevelP&,
                                              SchedulerP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
}

void
FractureMPM::scheduleTimeAdvance(const LevelP & level,
                                 SchedulerP   & sched)
{
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels()))
    return;

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  scheduleApplyExternalLoads(             sched, patches, matls);
  scheduleParticleVelocityField(          sched, patches, matls);//for FractureMPM 
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  scheduleComputeHeatExchange(            sched, patches, matls);
  scheduleAdjustCrackContactInterpolated( sched, patches, matls);//for FractureMPM
  scheduleExMomInterpolated(              sched, patches, matls);
  scheduleComputeContactArea(             sched, patches, matls);
  scheduleComputeInternalForce(           sched, patches, matls);
  scheduleComputeAndIntegrateAcceleration(sched, patches, matls);
  scheduleAdjustCrackContactIntegrated(   sched, patches, matls);//for FractureMPM
  scheduleExMomIntegrated(                sched, patches, matls);
  scheduleSetGridBoundaryConditions(      sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);
  if(flags->d_doExplicitHeatConduction){
    scheduleComputeInternalHeatRate(      sched, patches, matls);
    scheduleComputeNodalHeatFlux(         sched, patches, matls);
    scheduleSolveHeatEquations(           sched, patches, matls);
    scheduleIntegrateTemperatureRate(     sched, patches, matls);
  }
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls); 
  scheduleCalculateFractureParameters(    sched, patches, matls);//for FractureMPM 
  scheduleDoCrackPropagation(             sched, patches, matls);//for FractureMPM
  scheduleMoveCracks(                     sched, patches, matls);//for FractureMPM
  scheduleUpdateCrackFront(               sched, patches, matls);//for FractureMPM

  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    d_sharedState->d_particleState_preReloc,
                                    lb->pXLabel, 
                                    d_sharedState->d_particleState,
                                    lb->pParticleIDLabel, matls);
}

void FractureMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
 /*
  * applyExternalLoads
  *   in(p.externalForce, p.externalheatrate)
  *   out(p.externalForceNew, p.externalheatrateNew) */
  Task* t=scinew Task("FractureMPM::applyExternalLoads",
                    this, &FractureMPM::applyExternalLoads);

  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->computes(             lb->pExtForceLabel_preReloc);
  if (flags->d_useLoadCurves) {
    t->requires(Task::OldDW, lb->pXLabel, Ghost::None);
    t->requires(Task::OldDW, lb->pLoadCurveIDLabel,    Ghost::None);
    t->computes(             lb->pLoadCurveIDLabel_preReloc);
  }

//  t->computes(Task::OldDW, lb->pExternalHeatRateLabel_preReloc);

  sched->addTask(t, patches, matls);
}

// Determine velocity field for each particle-node pair
void FractureMPM::scheduleParticleVelocityField(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{
  Task* t = scinew Task("Crack::ParticleVelocityField", crackModel,
                        &Crack::ParticleVelocityField);

  crackModel->addComputesAndRequiresParticleVelocityField(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
  /* interpolateParticlesToGrid
   *   in(P.MASS, P.VELOCITY, P.NAT_X)
   *   operation(interpolate the P.MASS and P.VEL to the grid
   *             using P.NAT_X and some shape function evaluations)
   *   out(G.MASS, G.VELOCITY) */


  Task* t = scinew Task("FractureMPM::interpolateParticlesToGrid",
                        this,&FractureMPM::interpolateParticlesToGrid);

 
  Ghost::GhostType  gan = Ghost::AroundNodes;
  t->requires(Task::OldDW, lb->pMassLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pVolumeLabel,           gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,         gan,NGP);
  t->requires(Task::OldDW, lb->pXLabel,                gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      gan,NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,             gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,gan,NGP);
//t->requires(Task::OldDW, lb->pExternalHeatRateLabel, gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,   gan,NGP);



  t->computes(lb->gMassLabel);
  t->computes(lb->gMassLabel,        d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
  t->computes(lb->gTemperatureLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
  t->computes(lb->gVolumeLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);  
  t->computes(lb->gVelocityLabel,    d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
  t->computes(lb->gSp_volLabel);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureNoBCLabel);
  t->computes(lb->gTemperatureRateLabel);
  t->computes(lb->gExternalHeatRateLabel);
  t->computes(lb->gNumNearParticlesLabel);
  t->computes(lb->TotalMassLabel);
 
  // for FractureMPM
  t->requires(Task::OldDW, lb->pDispLabel,  gan, NGP);
  t->requires(Task::NewDW, lb->pgCodeLabel, gan, NGP);
  t->computes(lb->GMassLabel);
  t->computes(lb->GSp_volLabel);
  t->computes(lb->GVolumeLabel);
  t->computes(lb->GVelocityLabel);
  t->computes(lb->GExternalForceLabel);
  t->computes(lb->GTemperatureLabel);
  t->computes(lb->GTemperatureNoBCLabel);
  t->computes(lb->GExternalHeatRateLabel);
  t->computes(lb->gDisplacementLabel);
  t->computes(lb->GDisplacementLabel);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleComputeHeatExchange(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
 /* computeHeatExchange
  *   in(G.MASS, G.TEMPERATURE, G.EXTERNAL_HEAT_RATE)
  *   operation(peform heat exchange which will cause each of
  *   velocity fields to exchange heat according to 
  *   the temperature differences)
  *   out(G.EXTERNAL_HEAT_RATE) */

  if (cout_doing.active())
    cout_doing << getpid() << " Doing FractureMPM::ThermalContact::computeHeatExchange " << endl;

  Task* t = scinew Task("ThermalContact::computeHeatExchange",
                        thermalContactModel,
                        &ThermalContact::computeHeatExchange);

  thermalContactModel->addComputesAndRequires(t, patches, matls);
  sched->addTask(t, patches, matls);
}

// Check crack contact and adjust grid velocity field
void FractureMPM::scheduleAdjustCrackContactInterpolated(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  Task* t = scinew Task("Crack::AdjustCrackContactInterpolated",
                    crackModel,&Crack::AdjustCrackContactInterpolated);

  crackModel->addComputesAndRequiresAdjustCrackContactInterpolated(t,
                                                     patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleExMomInterpolated(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  contactModel->addComputesAndRequiresInterpolated(sched, patches, matls);
}

/////////////////////////////////////////////////////////////////////////
/*!  **WARNING** In addition to the stresses and deformations, the internal
 *               heat rate in the particles (pdTdtLabel)
 *               is computed here */
/////////////////////////////////////////////////////////////////////////
void FractureMPM::scheduleComputeStressTensor(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  // for thermal stress analysis
  scheduleComputeParticleTempFromGrid(sched, patches, matls);

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("FractureMPM::computeStressTensor",
                    this, &FractureMPM::computeStressTensor);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches);
    const MaterialSubset* matlset = mpm_matl->thisMaterial();
    t->computes(lb->p_qLabel_preReloc, matlset);
  }

  t->computes(d_sharedState->get_delt_label(),getLevel(patches));
  t->computes(lb->StrainEnergyLabel);

  sched->addTask(t, patches, matls);

  if (flags->d_reductionVars->accStrainEnergy) {
    scheduleComputeAccStrainEnergy(sched, patches, matls);
  } 
  if(flags->d_artificial_viscosity){
    scheduleComputeArtificialViscosity(sched, patches, matls);
  }
}

// Compute particle temperature by interpolating grid temperature
// for thermal stress analysis
void FractureMPM::scheduleComputeParticleTempFromGrid(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls)
{
  Ghost::GhostType gac = Ghost::AroundCells;
  Ghost::GhostType gan = Ghost::AroundNodes;
  Task* t = scinew Task("FractureMPM::computeParticleTempFromGrid",
                        this, &FractureMPM::computeParticleTempFromGrid);
  t->requires(Task::OldDW, lb->pXLabel,                 gan, NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,              gan, NGP);
  t->requires(Task::NewDW, lb->gTemperatureLabel,       gac, NGN);
  t->requires(Task::NewDW, lb->GTemperatureLabel,       gac, NGN);
  t->requires(Task::NewDW,lb->pgCodeLabel,              gan, NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,gan,NGP);
  t->computes(lb->pTempCurrentLabel);
  sched->addTask(t, patches, matls);
}

// Compute the accumulated strain energy
void FractureMPM::scheduleComputeAccStrainEnergy(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  Task* t = scinew Task("FractureMPM::computeAccStrainEnergy",
                        this, &FractureMPM::computeAccStrainEnergy);
  t->requires(Task::OldDW, lb->AccStrainEnergyLabel);
  t->requires(Task::NewDW, lb->StrainEnergyLabel);
  t->computes(lb->AccStrainEnergyLabel);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleComputeArtificialViscosity(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
  Task* t = scinew Task("FractureMPM::computeArtificialViscosity",
                    this, &FractureMPM::computeArtificialViscosity);

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, lb->pXLabel,                 Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,              Ghost::None);
  t->requires(Task::NewDW, lb->pVolumeLabel,            Ghost::None);
  t->requires(Task::OldDW, lb->pSizeLabel,              Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,      gac, NGN);
  t->requires(Task::NewDW, lb->GVelocityStarLabel,      gac, NGN);  // for FractureMPM
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,Ghost::None);
  t->computes(lb->p_qLabel);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleComputeContactArea(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  /*
   * computeContactArea */
  if(d_bndy_traction_faces.size()>0) {
    Task* t = scinew Task("FractureMPM::computeContactArea",
                          this, &FractureMPM::computeContactArea);

    Ghost::GhostType  gnone = Ghost::None;
    t->requires(Task::NewDW, lb->gVolumeLabel, gnone);
    t->requires(Task::NewDW, lb->GVolumeLabel, gnone); // for FractureMPM
    for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
        ftit!=d_bndy_traction_faces.end();ftit++) {
      int iface = (int)(*ftit);
      t->computes(lb->BndyContactCellAreaLabel[iface]);
    }
    sched->addTask(t, patches, matls);
  }
}

void FractureMPM::scheduleComputeInternalForce(SchedulerP& sched,
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

  Task* t = scinew Task("FractureMPM::computeInternalForce",
                    this, &FractureMPM::computeInternalForce);


  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gMassLabel, gnone);
  t->requires(Task::NewDW,lb->gMassLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain, gnone);
  t->requires(Task::OldDW,lb->pStressLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pVolumeLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::OldDW,lb->pMassLabel,                 gan,NGP);
  t->requires(Task::OldDW,lb->pSizeLabel,                 gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,   gan,NGP);


  // for FractureMPM
  t->requires(Task::NewDW,lb->pgCodeLabel,                gan,NGP);
  t->requires(Task::NewDW,lb->GMassLabel, gnone); 
  t->computes(lb->GInternalForceLabel);
  t->computes(lb->TotalVolumeDeformedLabel);

  if(flags->d_with_ice){
    t->requires(Task::NewDW, lb->pPressureLabel,          gan,NGP);
  }

  if(flags->d_artificial_viscosity){
    t->requires(Task::NewDW, lb->p_qLabel,                gan,NGP);
  }

  t->computes(lb->gInternalForceLabel);

  for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
      ftit!=d_bndy_traction_faces.end();ftit++) {
    int iface = (int)(*ftit);
    t->requires(Task::NewDW, lb->BndyContactCellAreaLabel[iface]);
    t->computes(lb->BndyForceLabel[iface]);
    t->computes(lb->BndyContactAreaLabel[iface]);
    t->computes(lb->BndyTractionLabel[iface]);
  }

  t->computes(lb->gStressForSavingLabel);
  t->computes(lb->gStressForSavingLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleComputeInternalHeatRate(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{ 
  heatConductionModel->scheduleComputeInternalHeatRate(sched,patches,matls);
}

void FractureMPM::scheduleSolveHeatEquations(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  heatConductionModel->scheduleSolveHeatEquations(sched,patches,matls);
}

void FractureMPM::scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeAndIntegrateAcceleration");

  Task* t = scinew Task("MPM::computeAndIntegrateAcceleration",
                        this, &FractureMPM::computeAndIntegrateAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,          Ghost::None);

  t->requires(Task::NewDW, lb->GMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->GVelocityLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->GInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->GExternalForceLabel, Ghost::None);

  t->computes(lb->gVelocityStarLabel);
  t->computes(lb->gAccelerationLabel);

  t->computes(lb->GAccelerationLabel);
  t->computes(lb->GVelocityStarLabel);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleIntegrateTemperatureRate(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  heatConductionModel->scheduleIntegrateTemperatureRate(sched,patches,matls);
}

// Check crack contact and adjust nodal velocities and accelerations
void FractureMPM::scheduleAdjustCrackContactIntegrated(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{ 
  Task* t = scinew Task("Crack::AdjustCrackContactIntegrated",
                    crackModel,&Crack::AdjustCrackContactIntegrated);
  
  crackModel->addComputesAndRequiresAdjustCrackContactIntegrated(t,
                                                      patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleExMomIntegrated(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  contactModel->addComputesAndRequiresIntegrated(sched, patches, matls);
}

void FractureMPM::scheduleSetGridBoundaryConditions(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)

{
  Task* t=scinew Task("FractureMPM::setGridBoundaryConditions",
                    this, &FractureMPM::setGridBoundaryConditions);
                  
  const MaterialSubset* mss = matls->getUnion();
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::NewDW, lb->gVelocityLabel,   Ghost::None);

  // for FractureMPM
  t->modifies(             lb->GAccelerationLabel,     mss);
  t->modifies(             lb->GVelocityStarLabel,     mss);
  t->requires(Task::NewDW, lb->GVelocityLabel,   Ghost::None);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                         const PatchSet* patches,
                                                         const MaterialSet* matls)

{
 /*
  * interpolateToParticlesAndUpdate
  *   in(G.ACCELERATION, G.VELOCITY_STAR, P.NAT_X)
  *   operation(interpolate acceleration and v* to particles and
  *   integrate these to get new particle velocity and position)
  * out(P.VELOCITY, P.X, P.NAT_X) */

  Task* t=scinew Task("FractureMPM::interpolateToParticlesAndUpdate",
                      this, &FractureMPM::interpolateToParticlesAndUpdate);




  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType   gac = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gAccelerationLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureLabel,      gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->frictionalWorkLabel,    gac,NGN);
  t->requires(Task::OldDW, lb->pXLabel,                gnone);
  t->requires(Task::OldDW, lb->pMassLabel,             gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      gnone);
  t->requires(Task::OldDW, lb->pVelocityLabel,         gnone);
  t->requires(Task::OldDW, lb->pDispLabel,             gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,             gnone);
  t->modifies(lb->pVolumeLabel_preReloc);
  // for thermal stress analysis
  t->requires(Task::NewDW, lb->pTempCurrentLabel,      gnone);
  t->requires(Task::NewDW, lb->pDeformationMeasureLabel_preReloc,        gnone);



  // for FractureMPM
  t->requires(Task::NewDW, lb->GAccelerationLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->GVelocityStarLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->GTemperatureRateLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->GTemperatureLabel,      gac,NGN);
  t->requires(Task::NewDW, lb->GTemperatureNoBCLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->pgCodeLabel,            gnone);

  if(flags->d_with_ice){
    t->requires(Task::NewDW, lb->dTdt_NCLabel,         gac,NGN);
    t->requires(Task::NewDW, lb->massBurnFractionLabel,gac,NGN);
  }

  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pTempPreviousLabel_preReloc); //for thermal stress
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pSizeLabel_preReloc);
  t->computes(lb->pXXLabel);
  t->computes(lb->pKineticEnergyDensityLabel); //for FractureMPM
  
  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->ThermalEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->TotalMomentumLabel);

  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleCalculateFractureParameters(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  // Get nodal solutions
  Task* t = scinew Task("Crack::GetNodalSolutions", crackModel,
                        &Crack::GetNodalSolutions);
  crackModel->addComputesAndRequiresGetNodalSolutions(t,patches, matls);
  sched->addTask(t, patches, matls);

  // cfnset & cfsset
  t = scinew Task("Crack::CrackFrontNodeSubset", crackModel,
                        &Crack::CrackFrontNodeSubset);
  crackModel->addComputesAndRequiresCrackFrontNodeSubset(t,patches, matls);
  sched->addTask(t, patches, matls);
  
  // Compute fracture parameters (J, K,...)
  t = scinew Task("Crack::CalculateFractureParameters", crackModel,
                        &Crack::CalculateFractureParameters);
  crackModel->addComputesAndRequiresCalculateFractureParameters(t, 
                                                    patches, matls);
  sched->addTask(t, patches, matls);
}

// Do crack propgation 
void FractureMPM::scheduleDoCrackPropagation(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{           
  // Propagate crack-front points
  Task* t = scinew Task("Crack::PropagateCrackFrontPoints", crackModel,
                  &Crack::PropagateCrackFrontPoints);
  crackModel->addComputesAndRequiresPropagateCrackFrontPoints(t, patches, matls);
  sched->addTask(t, patches, matls);

  // Construct the new crack-front elems and new crack-front segments.
  // The new crack-front is temporary, and will be updated after moving cracks
  t = scinew Task("Crack::ConstructNewCrackFrontElems", crackModel,
                  &Crack::ConstructNewCrackFrontElems);
  crackModel->addComputesAndRequiresConstructNewCrackFrontElems(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleMoveCracks(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  // Set up cpset -- crack node subset in each patch
  Task* t = scinew Task("Crack::CrackPointSubset", crackModel,
                        &Crack::CrackPointSubset);
  crackModel->addComputesAndRequiresCrackPointSubset(t, patches, matls);
  sched->addTask(t, patches, matls);

  // Move crack points
  t = scinew Task("Crack::MoveCracks", crackModel,
                        &Crack::MoveCracks);
  crackModel->addComputesAndRequiresMoveCracks(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleUpdateCrackFront(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  // Set up cfnset & cfsset -- subset for the temporary crack-front
  // and crack-front segment subset for each patch
  Task* t = scinew Task("Crack::CrackFrontNodeSubset", crackModel,
                        &Crack::CrackFrontNodeSubset);
  crackModel->addComputesAndRequiresCrackFrontNodeSubset(t, patches, matls);
  sched->addTask(t, patches, matls);

  // Recollect crack-front segments, discarding the dead segments,
  // calculating normals, indexes and so on
  t = scinew Task("Crack::RecollectCrackFrontSegments", crackModel,
                        &Crack::RecollectCrackFrontSegments);
  crackModel->addComputesAndRequiresRecollectCrackFrontSegments(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleRefine(const PatchSet* patches,
                                 SchedulerP& sched)
{
  Task* task = scinew Task("FractureMPM::refine", this, &FractureMPM::refine);
  sched->addTask(task, patches, d_sharedState->allMPMMaterials());
  // do nothing for now
}

void FractureMPM::scheduleRefineInterface(const LevelP& /*fineLevel*/,
                                          SchedulerP& /*scheduler*/,
                                          bool, bool)
{
  // do nothing for now
}

void FractureMPM::scheduleCoarsen(const LevelP& /*coarseLevel*/,
                                SchedulerP& /*sched*/)
{
  // do nothing for now
}

/// Schedule to mark flags for AMR regridding
void FractureMPM::scheduleErrorEstimate(const LevelP& coarseLevel,
                                        SchedulerP& sched)
{
  // main way is to count particles, but for now we only want particles on
  // the finest level.  Thus to schedule cells for regridding during the
  // execution, we'll coarsen the flagged cells (see coarsen).

  if (cout_doing.active())
    cout_doing << "FractureMPM::scheduleErrorEstimate on level " << coarseLevel->getIndex() << '\n';


  // Estimate error - this should probably be in it's own schedule,
  // and the simulation controller should not schedule it every time step
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task* task = scinew Task("errorEstimate", this, &FractureMPM::errorEstimate);

  // if the finest level, compute flagged cells
  if (coarseLevel->getIndex() == coarseLevel->getGrid()->numLevels()-1) {
    task->requires(Task::NewDW, lb->pXLabel,     gac, 0);
  }
  else {
    task->requires(Task::NewDW, d_sharedState->get_refineFlag_label(),
                   0, Task::FineLevel, d_sharedState->refineFlagMaterials(),
                   Task::NormalDomain, Ghost::None, 0);
  }
  task->modifies(d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials());
  task->modifies(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials());
  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allMPMMaterials());

}

/// Schedule to mark initial flags for AMR regridding
void FractureMPM::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                               SchedulerP& sched)
{

  if (cout_doing.active())
    cout_doing << "FractureMPM::scheduleErrorEstimate on level " << coarseLevel->getIndex() << '\n';

  // Estimate error - this should probably be in it's own schedule,
  // and the simulation controller should not schedule it every time step
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task* task = scinew Task("errorEstimate", this, &FractureMPM::initialErrorEstimate);
  task->requires(Task::NewDW, lb->pXLabel,     gac, 0);

  task->modifies(d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials());
  task->modifies(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials());
  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allMPMMaterials());
}


void FractureMPM::computeAccStrainEnergy(const ProcessorGroup*,
                                         const PatchSubset*,
                                         const MaterialSubset*,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  // Get the totalStrainEnergy from the old datawarehouse
  max_vartype accStrainEnergy;
  old_dw->get(accStrainEnergy, lb->AccStrainEnergyLabel);

  // Get the incremental strain energy from the new datawarehouse
  sum_vartype incStrainEnergy;
  new_dw->get(incStrainEnergy, lb->StrainEnergyLabel);

  // Add the two a put into new dw
  double totalStrainEnergy =
    (double) accStrainEnergy + (double) incStrainEnergy;
  new_dw->put(max_vartype(totalStrainEnergy), lb->AccStrainEnergyLabel);
}

// Calculate the number of material points per load curve
void FractureMPM::countMaterialPointsPerLoadCurve(const ProcessorGroup*,
                                                  const PatchSubset* patches,
                                                  const MaterialSubset*,
                                                  DataWarehouse* ,
                                                  DataWarehouse* new_dw)
{
  // Find the number of pressure BCs in the problem
  int nofPressureBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure") {
      nofPressureBCs++;

      // Loop through the patches and count
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        int numMPMMatls=d_sharedState->getNumMPMMatls();
        int numPts = 0;
        for(int m = 0; m < numMPMMatls; m++){
          MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
          int dwi = mpm_matl->getDWIndex();

          ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
          constParticleVariable<int> pLoadCurveID;
          new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            if (pLoadCurveID[idx] == (nofPressureBCs)) ++numPts;
          }
        } // matl loop
        new_dw->put(sumlong_vartype(numPts),
                    lb->materialPointsPerLoadCurveLabel, 0, nofPressureBCs-1);
      }  // patch loop
    }
  }
}

// Calculate the number of material points per load curve
void FractureMPM::initializePressureBC(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset*,
                                       DataWarehouse* ,
                                       DataWarehouse* new_dw)
{
  // Get the current time
  double time = 0.0;

  if (cout_dbg.active())
    cout_dbg << "Current Time (Initialize Pressure BC) = " << time << endl;


  // Calculate the force vector at each particle
  int nofPressureBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure") {

      // Get the material points per load curve
      sumlong_vartype numPart = 0;
      new_dw->get(numPart, lb->materialPointsPerLoadCurveLabel,
                  0, nofPressureBCs++);

      // Save the material points per load curve in the PressureBC object
      PressureBC* pbc =
        dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      pbc->numMaterialPoints(numPart);

      if (cout_dbg.active())
      cout_dbg << "    Load Curve = " << nofPressureBCs << " Num Particles = " << numPart << endl;


      // Calculate the force per particle at t = 0.0
      double forcePerPart = pbc->forcePerParticle(time);

      // Loop through the patches and calculate the force vector
      // at each particle
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        int numMPMMatls=d_sharedState->getNumMPMMatls();
        for(int m = 0; m < numMPMMatls; m++){
          MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
          int dwi = mpm_matl->getDWIndex();

          ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
          constParticleVariable<Point>  px;
          new_dw->get(px, lb->pXLabel,             pset);
          constParticleVariable<int> pLoadCurveID;
          new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
          ParticleVariable<Vector> pExternalForce;
          new_dw->getModifiable(pExternalForce, lb->pExternalForceLabel, pset);

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            if (pLoadCurveID[idx] == nofPressureBCs) {
              pExternalForce[idx] = pbc->getForceVector(px[idx], forcePerPart,
                                                        time);
            }
          }
        } // matl loop
      }  // patch loop
    }
  }
}

void FractureMPM::actuallyInitialize(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  particleIndex totalParticles=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing actuallyInitialize on patch " << patch->getID()
                 <<"\t\t\t MPM"<< endl;

    CCVariable<short int> cellNAPID;
    new_dw->allocateAndPut(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
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
                                                         mpm_matl,
                                                         new_dw);
      // scalar used for debugging
      if(flags->d_with_color) {
        ParticleVariable<double> pcolor;
        int index = mpm_matl->getDWIndex();
        ParticleSubset* pset = new_dw->getParticleSubset(index, patch);
        setParticleDefault(pcolor, lb->pColorLabel, pset, new_dw, 0.0);
      }
    }
  }

  if (flags->d_reductionVars->accStrainEnergy) {
    // Initialize the accumulated strain energy
    new_dw->put(max_vartype(0.0), lb->AccStrainEnergyLabel);
  }

  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);
}

void FractureMPM::actuallyInitializeAddedMaterial(const ProcessorGroup*,
                                                  const PatchSubset* patches,
                                                  const MaterialSubset* matls,
                                                  DataWarehouse*,
                                                  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing actuallyInitializeAddedMaterial on patch " << patch->getID() <<"\t\t\t MPM"<< endl;


    int numMPMMatls = d_sharedState->getNumMPMMatls();
    cout << "num MPM Matls = " << numMPMMatls << endl;
    CCVariable<short int> cellNAPID;
    int m=numMPMMatls-1;
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    particleIndex numParticles = mpm_matl->countParticles(patch);

    new_dw->unfinalize();
    mpm_matl->createParticles(numParticles, cellNAPID, patch, new_dw);

    mpm_matl->getConstitutiveModel()->initializeCMData(patch, mpm_matl, new_dw);
    new_dw->refinalize();
  }
}


void FractureMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
                                              const PatchSubset*,
                                              const MaterialSubset*,
                                              DataWarehouse*,
                                              DataWarehouse*)
{
}

void FractureMPM::interpolateParticlesToGrid(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing interpolateParticlesToGrid on patch " << patch->getID()<<"\t\t MPM"<< endl;

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());



    NCVariable<double> gmassglobal,gtempglobal,gvolumeglobal;
    NCVariable<Vector> gvelglobal;
    new_dw->allocateAndPut(gmassglobal, lb->gMassLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gtempglobal, lb->gTemperatureLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gvolumeglobal, lb->gVolumeLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);    
    new_dw->allocateAndPut(gvelglobal, lb->gVelocityLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    gmassglobal.initialize(d_SMALL_NUM_MPM);
    gvolumeglobal.initialize(d_SMALL_NUM_MPM);
    gtempglobal.initialize(0.0);
    gvelglobal.initialize(Vector(0.0));

    Ghost::GhostType  gan = Ghost::AroundNodes;
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      constParticleVariable<Vector> pvelocity, pexternalforce, pdisp;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;


      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);
      old_dw->get(psize,          lb->pSizeLabel,          pset);
      new_dw->get(pexternalforce, lb->pExtForceLabel_preReloc, pset);
      old_dw->get(pDeformationMeasure,  lb->pDeformationMeasureLabel, pset);


      // for FractureMPM
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode,         lb->pgCodeLabel,         pset);
      old_dw->get(pdisp,          lb->pDispLabel,          pset);
      
      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> gexternalforce;
      NCVariable<double> gexternalheatrate;
      NCVariable<double> gTemperature;
      NCVariable<double> gSp_vol;
      NCVariable<double> gTemperatureNoBC;
      NCVariable<double> gTemperatureRate;
      NCVariable<double> gnumnearparticles;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,       dwi,patch);
      new_dw->allocateAndPut(gSp_vol,          lb->gSp_volLabel,     dwi,patch);
      new_dw->allocateAndPut(gvolume,          lb->gVolumeLabel,     dwi,patch);
      new_dw->allocateAndPut(gvelocity,        lb->gVelocityLabel,   dwi,patch);
      new_dw->allocateAndPut(gTemperature,     lb->gTemperatureLabel,dwi,patch);
      new_dw->allocateAndPut(gTemperatureNoBC, lb->gTemperatureNoBCLabel,
                             dwi,patch);
      new_dw->allocateAndPut(gTemperatureRate, lb->gTemperatureRateLabel,
                             dwi,patch);
      new_dw->allocateAndPut(gexternalforce,   lb->gExternalForceLabel,
                             dwi,patch);
      new_dw->allocateAndPut(gexternalheatrate,lb->gExternalHeatRateLabel,
                             dwi,patch);
      new_dw->allocateAndPut(gnumnearparticles,lb->gNumNearParticlesLabel,
                             dwi,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(d_SMALL_NUM_MPM);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);
      gTemperatureNoBC.initialize(0);
      gTemperatureRate.initialize(0);
      gexternalheatrate.initialize(0);
      gnumnearparticles.initialize(0.);
      gSp_vol.initialize(0.);

      // for FractureMPM
      NCVariable<double> Gmass;
      NCVariable<double> Gvolume;
      NCVariable<Vector> Gvelocity;
      NCVariable<Vector> Gexternalforce;
      NCVariable<double> Gexternalheatrate;
      NCVariable<double> GTemperature;
      NCVariable<double> GSp_vol;
      NCVariable<double> GTemperatureNoBC;
      NCVariable<Vector> gdisplacement;
      NCVariable<Vector> Gdisplacement;

      new_dw->allocateAndPut(Gmass,            lb->GMassLabel,       dwi,patch);
      new_dw->allocateAndPut(GSp_vol,          lb->GSp_volLabel,     dwi,patch);
      new_dw->allocateAndPut(Gvolume,          lb->GVolumeLabel,     dwi,patch);
      new_dw->allocateAndPut(Gvelocity,        lb->GVelocityLabel,   dwi,patch);
      new_dw->allocateAndPut(GTemperature,     lb->GTemperatureLabel,dwi,patch);
      new_dw->allocateAndPut(GTemperatureNoBC, lb->GTemperatureNoBCLabel,
                                                                     dwi,patch);
      new_dw->allocateAndPut(Gexternalforce,   lb->GExternalForceLabel,
                                                                     dwi,patch);
      new_dw->allocateAndPut(Gexternalheatrate,lb->GExternalHeatRateLabel,
                                                                     dwi,patch);
      new_dw->allocateAndPut(gdisplacement,   lb->gDisplacementLabel,dwi,patch);
      new_dw->allocateAndPut(Gdisplacement,   lb->GDisplacementLabel,dwi,patch);

      // initialization
      Gmass.initialize(d_SMALL_NUM_MPM);
      Gvolume.initialize(d_SMALL_NUM_MPM);
      Gvelocity.initialize(Vector(0,0,0));
      Gexternalforce.initialize(Vector(0,0,0));
      GTemperature.initialize(0);
      GTemperatureNoBC.initialize(0);
      Gexternalheatrate.initialize(0);
      GSp_vol.initialize(0.);
      gdisplacement.initialize(Vector(0,0,0));
      Gdisplacement.initialize(Vector(0,0,0));

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity
      
      double totalmass = 0;
      Vector total_mom(0.0,0.0,0.0);
      Vector pmom;
      
      double pSp_vol = 1./mpm_matl->getInitialDensity();
      for (ParticleSubset::iterator iter = pset->begin();
                                    iter != pset->end(); 
                                    iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pDeformationMeasure[idx]);

        pmom = pvelocity[idx]*pmass[idx];
        total_mom += pvelocity[idx]*pmass[idx];

        // Add each particles contribution to the local mass & velocity 
        // Must use the node indices
        for(int k = 0; k < flags->d_8or27; k++) {
          if(patch->containsNode(ni[k])) {
            if(pgCode[idx][k]==1) {   // above crack
              gmass[ni[k]]          += pmass[idx]                     * S[k];
              gvelocity[ni[k]]      += pmom                           * S[k];
              gvolume[ni[k]]        += pvolume[idx]                   * S[k];
              gexternalforce[ni[k]] += pexternalforce[idx]            * S[k];
              gTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
              //gexternalheatrate[ni[k]] += pexternalheatrate[idx]      * S[k];
              gnumnearparticles[ni[k]] += 1.0;
              gSp_vol[ni[k]]        += pSp_vol * pmass[idx]           * S[k];
              gdisplacement[ni[k]]  += pdisp[idx] * pmass[idx]        * S[k];
            }
            else if(pgCode[idx][k]==2) {  // below crack
              Gmass[ni[k]]          += pmass[idx]                     * S[k];
              Gvelocity[ni[k]]      += pmom                           * S[k];
              Gvolume[ni[k]]        += pvolume[idx]                   * S[k];
              Gexternalforce[ni[k]] += pexternalforce[idx]            * S[k];
              GTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
              //Gexternalheatrate[ni[k]] += pexternalheatrate[idx]      * S[k];
              GSp_vol[ni[k]]        += pSp_vol * pmass[idx]           * S[k];
              Gdisplacement[ni[k]]  += pdisp[idx] * pmass[idx]        * S[k];
            }
          }
        } // End of loop over k
      } // End of loop over iter

      string interp_type = flags->d_interpolator_type;
      for(NodeIterator iter=patch->getExtraNodeIterator();
                                           !iter.done();iter++){
        IntVector c = *iter; 
        totalmass      += (gmass[c]+Gmass[c]);
        gmassglobal[c] += (gmass[c]+Gmass[c]);
        gvolumeglobal[c]  += (gvolume[c]+Gvolume[c]);
        gvelglobal[c]  += (gvelocity[c]+Gvelocity[c]); 
        gtempglobal[c] += (gTemperature[c]+GTemperature[c]);

        // above crack
        gvelocity[c]       /= gmass[c];
        gTemperature[c]    /= gmass[c];
        gSp_vol[c]         /= gmass[c];
        gdisplacement[c]   /= gmass[c];
        gTemperatureNoBC[c] = gTemperature[c];

        // below crack 
        Gvelocity[c]       /= Gmass[c];
        GTemperature[c]    /= Gmass[c];
        GSp_vol[c]         /= Gmass[c];
        Gdisplacement[c]   /= Gmass[c];
        GTemperatureNoBC[c] = GTemperature[c];
        
      }

      // Apply grid boundary conditions to the velocity before storing the data
      MPMBoundCond bc;
      // above crack
      bc.setBoundaryCondition(patch,dwi,"Velocity",   gvelocity,      interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gvelocity,      interp_type);
      bc.setBoundaryCondition(patch,dwi,"Temperature",gTemperature,   interp_type);
      // below crack
      bc.setBoundaryCondition(patch,dwi,"Velocity",   Gvelocity,      interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  Gvelocity,      interp_type);
      bc.setBoundaryCondition(patch,dwi,"Temperature",GTemperature,   interp_type);

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);

    }  // End loop over materials

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;
      gtempglobal[c] /= gmassglobal[c];
      gvelglobal[c]  /= gmassglobal[c];
    }
    delete interpolator;
  }  // End loop over patches
}

void FractureMPM::computeStressTensor(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* ,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  if (cout_doing.active())
      cout_doing <<"Doing computeStressTensor:FractureMPM: \n" ;          

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){

    if (cout_dbg.active()) {
      cout_dbg << " Patch = " << (patches->get(0))->getID();
      cout_dbg << " Mat = " << m;
    }
    
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);

    if (cout_dbg.active())
      cout_dbg << " MPM_Mat = " << mpm_matl;
    
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

    if (cout_dbg.active())
      cout_dbg << " CM = " << cm;

    cm->setWorld(UintahParallelComponent::d_myworld);
    cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);

    if (cout_dbg.active())
              cout_dbg << " Exit\n" ;    

  }
}

void FractureMPM::computeArtificialViscosity(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* ,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  double C0 = flags->d_artificialViscCoeff1;  
  double C1 = flags->d_artificialViscCoeff2;    
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())    
    cout_doing <<"Doing computeArtificialViscosity on patch " << patch->getID()
               <<"\t\t MPM"<< endl;
    

    // The following scheme for removing ringing behind a shock comes from:
    // VonNeumann, J.; Richtmyer, R. D. (1950): A method for the numerical
    // calculation of hydrodynamic shocks. J. Appl. Phys., vol. 21, pp. 232.
    
    Ghost::GhostType  gac   = Ghost::AroundCells;

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

  
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constNCVariable<Vector> gvelocity;
      ParticleVariable<double> p_q;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Point> px;
      constParticleVariable<double> pmass,pvol;
      constParticleVariable<Matrix3> pDeformationMeasure;


      new_dw->get(gvelocity, lb->gVelocityLabel, dwi,patch, gac, NGN);
      old_dw->get(px,        lb->pXLabel,                      pset);
      old_dw->get(pmass,     lb->pMassLabel,                   pset);
      new_dw->get(pvol,      lb->pVolumeLabel,                 pset);
      old_dw->get(psize,     lb->pSizeLabel,                   pset);
      new_dw->allocateAndPut(p_q,    lb->p_qLabel,             pset);
      old_dw->get(pDeformationMeasure,  lb->pDeformationMeasureLabel, pset);

     
      // for FractureMPM
      constNCVariable<Vector> Gvelocity;
      new_dw->get(Gvelocity, lb->GVelocityLabel, dwi,patch, gac, NGN);
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode,    lb->pgCodeLabel,                  pset);

      Matrix3 velGrad;
      Vector dx = patch->dCell();
      double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
      double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;

      double K = 1./mpm_matl->getConstitutiveModel()->getCompressibility();
      double c_dil;

      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],pDeformationMeasure[idx]);

        // get particle's velocity gradients 
        Vector gvel(0.,0.,0.);
        velGrad.set(0.0);
        for(int k = 0; k < flags->d_8or27; k++) {
          if(pgCode[idx][k]==1) gvel = gvelocity[ni[k]];
          if(pgCode[idx][k]==2) gvel = Gvelocity[ni[k]];
          for(int j = 0; j<3; j++){
            double d_SXoodx = d_S[k][j] * oodx[j];
            for(int i = 0; i<3; i++) {
              velGrad(i,j) += gvel[i] * d_SXoodx;
            }
          }
        }

        Matrix3 D = (velGrad + velGrad.Transpose())*.5;

        double DTrace = D.Trace();
        p_q[idx] = 0.0;
        if(DTrace<0.){
          c_dil = sqrt(K*pvol[idx]/pmass[idx]);
          p_q[idx] = (C0*fabs(c_dil*DTrace*dx_ave) +
                      C1*(DTrace*DTrace*dx_ave*dx_ave))*
                      (pmass[idx]/pvol[idx]);          
        }

      }
    }
    delete interpolator;
  }

}

void FractureMPM::computeContactArea(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  // six indices for each of the faces
  double bndyCArea[6] = {0,0,0,0,0,0};
        
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
        
    if (cout_doing.active())
      cout_doing <<"Doing computeContactArea on patch " << patch->getID() <<"\t\t\t MPM"<< endl;
        
        
    Vector dx = patch->dCell();
    double cellvol = dx.x()*dx.y()*dx.z();
        
    int numMPMMatls = d_sharedState->getNumMPMMatls();
        
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      constNCVariable<double> gvolume,Gvolume;
        
      new_dw->get(gvolume, lb->gVolumeLabel, dwi, patch, Ghost::None, 0);
      new_dw->get(Gvolume, lb->GVolumeLabel, dwi, patch, Ghost::None, 0); // for FractureMPM
        
      for(list<Patch::FaceType>::const_iterator fit(d_bndy_traction_faces.begin()); 
        fit!=d_bndy_traction_faces.end();fit++) {
        Patch::FaceType face = *fit;
        int iface = (int)(face);

        // Check if the face is on an external boundary
        if(patch->getBCType(face)==Patch::Neighbor)
          continue;
        
        // We are on the boundary, i.e. not on an interior patch
        // boundary, and also on the correct side, 
        // so do the traction accumulation . . .
        // loop cells to find boundary areas
        IntVector projlow, projhigh;
        patch->getFaceCells(face, 0, projlow, projhigh);
        // Vector norm = face_norm(face);
        
        for (int i = projlow.x(); i<projhigh.x(); i++) {
          for (int j = projlow.y(); j<projhigh.y(); j++) {
            for (int k = projlow.z(); k<projhigh.z(); k++) {
              IntVector ijk(i,j,k);
              double nodevol = gvolume[ijk]+Gvolume[ijk];
              if(nodevol>0) // FIXME: uses node index to get node volume ...
                {
                  const double celldepth  = dx[iface/2];
                  bndyCArea[iface] += cellvol/celldepth;
                }
            }
          }
        }
        
      } // faces
    } // materials
  } // patches
        
  // be careful only to put the fields that we have built
  // that way if the user asks to output a field that has not been built
  // it will fail early rather than just giving zeros.
  for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
        ftit!=d_bndy_traction_faces.end();ftit++) {
    int iface = (int)(*ftit);
    new_dw->put(sum_vartype(bndyCArea[iface]),lb->BndyContactCellAreaLabel[iface]);
  }
}       

void FractureMPM::computeInternalForce(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* ,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  // node based forces
  Vector bndyForce[6];
  Vector bndyTraction[6];
  for(int iface=0;iface<6;iface++) {
    bndyForce   [iface]  = Vector(0.);
    bndyTraction[iface]  = Vector(0.);
  }
  double partvoldef = 0.;
        
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing computeInternalForce on patch " << patch->getID()
                 <<"\t\t\t MPM"<< endl;

    
    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    double cellvol = dx.x()*dx.y()*dx.z();
    Matrix3 Id;
    Id.Identity();

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());



    int numMPMMatls = d_sharedState->getNumMPMMatls();

    NCVariable<Matrix3>       gstressglobal;
    constNCVariable<double>   gmassglobal;
    new_dw->get(gmassglobal,  lb->gMassLabel,
                d_sharedState->getAllInOneMatl()->get(0), patch, Ghost::None,0);
    new_dw->allocateAndPut(gstressglobal, lb->gStressForSavingLabel, 
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    //gstressglobal.initialize(Matrix3(0.));

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Create arrays for the particle position, volume
      // and the constitutive model
      constParticleVariable<Point>   px;
      constParticleVariable<double>  pvol, pmass;
      constParticleVariable<double>  p_pressure;
      constParticleVariable<double>  p_q;
      constParticleVariable<Matrix3> pstress;
      constParticleVariable<Matrix3>  psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      NCVariable<Vector>             internalforce;
      NCVariable<Matrix3>            gstress;
      constNCVariable<double>        gmass;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel,                      pset);
      old_dw->get(pmass,   lb->pMassLabel,                   pset);
      old_dw->get(pvol,    lb->pVolumeLabel,                 pset);
      old_dw->get(pstress, lb->pStressLabel,                 pset);
      old_dw->get(psize,   lb->pSizeLabel,                   pset);
      new_dw->get(gmass,   lb->gMassLabel, dwi, patch, Ghost::None, 0);
      old_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel, pset);


      new_dw->allocateAndPut(gstress,      lb->gStressForSavingLabel,dwi,patch);
      new_dw->allocateAndPut(internalforce,lb->gInternalForceLabel,  dwi,patch);
      //gstress.initialize(Matrix3(0.));  
 
      // for FractureMPM
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode, lb->pgCodeLabel, pset);
      constNCVariable<double> Gmass;
      new_dw->get(Gmass, lb->GMassLabel, dwi, patch, Ghost::None, 0);
      NCVariable<Vector> Ginternalforce;
      new_dw->allocateAndPut(Ginternalforce,lb->GInternalForceLabel, dwi,patch);

      if(flags->d_with_ice){
        new_dw->get(p_pressure,lb->pPressureLabel, pset);
      }
      else {
        ParticleVariable<double>  p_pressure_create;
        new_dw->allocateTemporary(p_pressure_create,  pset);
        for(ParticleSubset::iterator it = pset->begin();it != pset->end();it++){
          p_pressure_create[*it]=0.0;
        }
        p_pressure = p_pressure_create; // reference created data
      }

      if(flags->d_artificial_viscosity){
        old_dw->get(p_q,lb->p_qLabel, pset);
      }
      else {
        ParticleVariable<double>  p_q_create;
        new_dw->allocateTemporary(p_q_create,  pset);
        for(ParticleSubset::iterator it = pset->begin();it != pset->end();it++){
          p_q_create[*it]=0.0;
        }
        p_q = p_q_create; // reference created data
      }

      internalforce.initialize(Vector(0,0,0));
      Ginternalforce.initialize(Vector(0,0,0));

      Matrix3 stressmass;
      Matrix3 stresspress;

      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx],pDeformationMeasure[idx]);

        stressmass  = pstress[idx]*pmass[idx];
        //stresspress = pstress[idx] + Id*p_pressure[idx];
        stresspress = pstress[idx] + Id*p_pressure[idx] - Id*p_q[idx];
        partvoldef += pvol[idx];

        for (int k = 0; k < flags->d_8or27; k++){
          if(patch->containsNode(ni[k])){
            Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                       d_S[k].z()*oodx[2]);
            if(pgCode[idx][k]==1) {
              internalforce[ni[k]] -= (div * stresspress) * pvol[idx];
            }
            else if(pgCode[idx][k]==2) {
              Ginternalforce[ni[k]] -=(div * stresspress) * pvol[idx];
            }
            gstress[ni[k]] += stressmass * S[k];
          }
        }
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        gstressglobal[c] += gstress[c];
        gstress[c] /= (gmass[c]+Gmass[c]); //add in addtional field
      }
      
      // save boundary forces before apply symmetry boundary condition.
      for(list<Patch::FaceType>::const_iterator fit(d_bndy_traction_faces.begin()); 
          fit!=d_bndy_traction_faces.end();fit++) {
        Patch::FaceType face = *fit;
      
        // Check if the face is on an external boundary
        if(patch->getBCType(face)==Patch::Neighbor)
          continue;
      
        const int iface = (int)face;
      
        // We are on the boundary, i.e. not on an interior patch
        // boundary, and also on the correct side, 
        // so do the traction accumulation . . .
        // loop nodes to find forces
        IntVector projlow, projhigh; 
        patch->getFaceNodes(face, 0, projlow, projhigh);
        Vector norm = face_norm(face);
      
        for (int i = projlow.x(); i<projhigh.x(); i++) { 
          for (int j = projlow.y(); j<projhigh.y(); j++) { 
            for (int k = projlow.z(); k<projhigh.z(); k++) {
              IntVector ijk(i,j,k);
      
              // flip sign so that pushing on boundary gives positive force
              bndyForce[iface] -= internalforce[ijk];
            }
          }
        }
        
        patch->getFaceCells(face, 0, projlow, projhigh);
        for (int i = projlow.x(); i<projhigh.x(); i++) { 
          for (int j = projlow.y(); j<projhigh.y(); j++) { 
            for (int k = projlow.z(); k<projhigh.z(); k++) {
              IntVector ijk(i,j,k);

              double celldepth  = dx[iface/2]; // length in direction perpendicular to boundary
              double dA_c       = cellvol/celldepth; // cell based volume

              for(int ic=0;ic<3;ic++) for(int jc=0;jc<3;jc++) {
                bndyTraction[iface][ic] += gstress[ijk](ic,jc)*norm[jc]*dA_c;
              }

            }
          }
        }

      } // faces
      string interp_type = flags->d_interpolator_type;
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",internalforce,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",Ginternalforce,interp_type);

#ifdef KUMAR
      internalforce.initialize(Vector(0,0,0));
      Ginternalforce.initialize(Vector(0,0,0));
#endif
    }      
      
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      gstressglobal[c] /= gmassglobal[c];
    }
    delete interpolator;
  }
  new_dw->put(sum_vartype(partvoldef), lb->TotalVolumeDeformedLabel);

  // be careful only to put the fields that we have built
  // that way if the user asks to output a field that has not been built
  // it will fail early rather than just giving zeros.
  for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
      ftit!=d_bndy_traction_faces.end();ftit++) {
    int iface = (int)(*ftit);
    new_dw->put(sumvec_vartype(bndyForce[iface]),lb->BndyForceLabel[iface]);
  
    sum_vartype bndyContactCellArea_iface;
    new_dw->get(bndyContactCellArea_iface, lb->BndyContactCellAreaLabel[iface]);
  
    if(bndyContactCellArea_iface>0)
      bndyTraction[iface] /= bndyContactCellArea_iface;
  
    new_dw->put(sumvec_vartype(bndyTraction[iface]),lb->BndyTractionLabel[iface]);
  
    double bndyContactArea_iface = bndyContactCellArea_iface;
    if(bndyTraction[iface].length2()>0)
      bndyContactArea_iface = ::sqrt(bndyForce[iface].length2()/bndyTraction[iface].length2());
    new_dw->put(sum_vartype(bndyContactArea_iface), lb->BndyContactAreaLabel[iface]);
  }  
}

void FractureMPM::computeAndIntegrateAcceleration(const ProcessorGroup*,
                                                  const PatchSubset* patches,
                                                  const MaterialSubset*,
                                                  DataWarehouse* old_dw,
                                                  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing integrateAcceleration on patch " << patch->getID()
                 <<"\t\t\t MPM"<< endl;
    
    Ghost::GhostType  gnone = Ghost::None;
    Vector gravity = flags->d_gravity;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      delt_vartype delT;
      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches));

      // Get required variables for this patch
      constNCVariable<Vector>  velocity;
      constNCVariable<Vector> internalforce;
      constNCVariable<Vector> externalforce;
      constNCVariable<double> mass;

      // for FractureMPM
      constNCVariable<Vector>  Gvelocity;
      new_dw->get(internalforce, lb->gInternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(externalforce, lb->gExternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(mass,          lb->gMassLabel,          dwi, patch, gnone, 0);
      new_dw->get(velocity,      lb->gVelocityLabel,      dwi, patch, gnone, 0);
      new_dw->get(Gvelocity,     lb->GVelocityLabel,      dwi, patch, gnone, 0);

      NCVariable<Vector> acceleration;
      NCVariable<Vector> velocity_star;
      NCVariable<Vector> Gvelocity_star;
      new_dw->allocateAndPut(velocity_star, lb->gVelocityStarLabel, dwi, patch);
      velocity_star.initialize(Vector(0.0));
      new_dw->allocateAndPut(Gvelocity_star,lb->GVelocityStarLabel, dwi, patch);
      Gvelocity_star.initialize(Vector(0.0));

      // Create variables for the results
      new_dw->allocateAndPut(acceleration, lb->gAccelerationLabel, dwi, patch);
      acceleration.initialize(Vector(0.,0.,0.));

      // for FractureMPM
      constNCVariable<double> Gmass;
      constNCVariable<Vector> Ginternalforce;
      constNCVariable<Vector> Gexternalforce;
      new_dw->get(Gmass,         lb->GMassLabel,         dwi, patch, gnone, 0);
      new_dw->get(Ginternalforce,lb->GInternalForceLabel,dwi, patch, gnone, 0);
      new_dw->get(Gexternalforce,lb->GExternalForceLabel,dwi, patch, gnone, 0);

      NCVariable<Vector> Gacceleration;
      new_dw->allocateAndPut(Gacceleration,lb->GAccelerationLabel, dwi, patch);
      Gacceleration.initialize(Vector(0.,0.,0.));

      string interp_type = flags->d_interpolator_type;
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done(); iter++){
        IntVector c = *iter;
        // above crack
        acceleration[c]=(internalforce[c]+externalforce[c])/mass[c]+gravity;
        // below crack
        Gacceleration[c]=(Ginternalforce[c]+Gexternalforce[c])/Gmass[c]+gravity;
       // above crack
       velocity_star[c] = velocity[c] + acceleration[c] * delT;
       // below crack
       Gvelocity_star[c]=Gvelocity[c] + Gacceleration[c] *delT;
      }
    }
  }
}

void FractureMPM::setGridBoundaryConditions(const ProcessorGroup*,
                                            const PatchSubset* patches,
                                            const MaterialSubset* ,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing setGridBoundaryConditions on patch " << patch->getID()
                 <<"\t\t MPM"<< endl;


    int numMPMMatls=d_sharedState->getNumMPMMatls();
    
    delt_vartype delT;            
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
                      
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<Vector> gvelocity;
      
      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,   dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,   dwi,patch);
      new_dw->get(gvelocity,lb->gVelocityLabel,dwi,patch,Ghost::None,0);
      // for FractureMPM
      NCVariable<Vector> Gvelocity_star, Gacceleration;
      constNCVariable<Vector> Gvelocity;
      new_dw->getModifiable(Gacceleration, lb->GAccelerationLabel,   dwi,patch);
      new_dw->getModifiable(Gvelocity_star,lb->GVelocityStarLabel,   dwi,patch);
      new_dw->get(Gvelocity,lb->GVelocityLabel,dwi,patch,Ghost::None,0);
      
      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles
      string interp_type = flags->d_interpolator_type;
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity",gvelocity_star,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Velocity",Gvelocity_star,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocity_star,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",Gvelocity_star,interp_type);

      // Now recompute acceleration as the difference between the velocity
      // interpolated to the grid (no bcs applied) and the new velocity_star
      for(NodeIterator iter = patch->getExtraNodeIterator(); !iter.done();
                                                               iter++){
        IntVector c = *iter;
        gacceleration[c] = (gvelocity_star[c] - gvelocity[c])/delT;
        Gacceleration[c] = (Gvelocity_star[c] - Gvelocity[c])/delT;
      }
    } // matl loop
  }  // patch loop

}

void FractureMPM::applyExternalLoads(const ProcessorGroup* ,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  // Get the current time
  double time = d_sharedState->getElapsedTime();

  if (cout_doing.active())
    cout_doing << "Current Time (applyExternalLoads) = " << time << endl;


  // Calculate the force vector at each particle for each pressure bc
  std::vector<double> forcePerPart;
  std::vector<PressureBC*> pbcP;
  if (flags->d_useLoadCurves) {
    for (int ii = 0; 
             ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
      string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
      if (bcs_type == "Pressure") {

        // Get the material points per load curve
        PressureBC* pbc = 
          dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
        pbcP.push_back(pbc);

        // Calculate the force per particle at current time
        forcePerPart.push_back(pbc->forcePerParticle(time));
      } 
    }
  }

  // Loop thru patches to update external force vector
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing applyExternalLoads on patch " 
                 << patch->getID() << "\t MPM"<< endl;

    
    // Place for user defined loading scenarios to be defined,
    // otherwise pExternalForce is just carried forward.

    int numMPMMatls=d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      if (flags->d_useLoadCurves) {
        bool do_PressureBCs=false;
        for (int ii = 0;
              ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
          string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
          if (bcs_type == "Pressure") {
            do_PressureBCs=true;
          }
        }
        if(do_PressureBCs){
          // Get the particle position data
          constParticleVariable<Point>  px;
          old_dw->get(px, lb->pXLabel, pset);

          // Get the load curve data
          constParticleVariable<int> pLoadCurveID;
          old_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);

          // Get the external force data and allocate new space for
          // external force
          ParticleVariable<Vector> pExternalForce;
          ParticleVariable<Vector> pExternalForce_new;
          old_dw->getModifiable(pExternalForce, lb->pExternalForceLabel, pset);
          new_dw->allocateAndPut(pExternalForce_new, 
                               lb->pExtForceLabel_preReloc,  pset);

          // Iterate over the particles
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            int loadCurveID = pLoadCurveID[idx]-1;
            if (loadCurveID < 0) {
              pExternalForce_new[idx] = pExternalForce[idx];
            } else {
              PressureBC* pbc = pbcP[loadCurveID];
              double force = forcePerPart[loadCurveID];
              pExternalForce_new[idx] = pbc->getForceVector(px[idx],force,time);
            }
          }

          // Recycle the loadCurveIDs
          ParticleVariable<int> pLoadCurveID_new;
          new_dw->allocateAndPut(pLoadCurveID_new, 
                                 lb->pLoadCurveIDLabel_preReloc, pset);
          pLoadCurveID_new.copyData(pLoadCurveID);
        }
      } else {  // Carry forward the old pEF, scale by d_forceIncrementFactor
        // Get the external force data and allocate new space for
        // external force and copy the data
        constParticleVariable<Vector> pExternalForce;
        ParticleVariable<Vector> pExternalForce_new;
        old_dw->get(pExternalForce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pExternalForce_new, 
                              lb->pExtForceLabel_preReloc,  pset);
        
        // Iterate over the particles 
        ParticleSubset::iterator iter = pset->begin();
        for(;iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pExternalForce_new[idx] = 
                  pExternalForce[idx]*flags->d_forceIncrementFactor;
        }
      }
    } // matl loop
  }  // patch loop
}

// for thermal stress analysis
void FractureMPM::computeParticleTempFromGrid(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset*,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constNCVariable<double> gTemperature, GTemperature;
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gTemperature, lb->gTemperatureLabel, dwi,patch, gac, NGP);
      new_dw->get(GTemperature, lb->GTemperatureLabel, dwi,patch, gac, NGP);

      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;

      old_dw->get(px,    lb->pXLabel,    pset);
      old_dw->get(psize, lb->pSizeLabel, pset);
      old_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel, pset);


      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode, lb->pgCodeLabel, pset);
            
      ParticleVariable<double> pTempCur;
      new_dw->allocateAndPut(pTempCur,lb->pTempCurrentLabel,pset);
            
      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++) {
        particleIndex idx = *iter;
        double pTemp=0.0;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx],pDeformationMeasure[idx]);
        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          if(pgCode[idx][k]==1) {
            pTemp += gTemperature[node] * S[k];
          }
          else if(pgCode[idx][k]==2) {
            pTemp += GTemperature[node] * S[k];
          }   
        }
        pTempCur[idx]=pTemp;
      } // End of loop over iter        
    } // End of loop over m
    delete interpolator;
  } // End of loop over p 
}       
        
void FractureMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* ,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active()) {
      cout_doing <<"Doing interpolateToParticlesAndUpdate on patch " 
                 << patch->getID() << "\t MPM"<< endl;
    }
    

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
 

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
 
    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    Vector CMX(0.0,0.0,0.0);
    Vector totalMom(0.0,0.0,0.0);
    double ke=0;
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
    bool combustion_problem=false;

    Material* reactant;
    int RMI = -99;
    reactant = d_sharedState->getMaterialByName("reactant");
    if(reactant != 0){
      RMI = reactant->getDWIndex();
      combustion_problem=true;
    }
    double move_particles=1.;
    if(!flags->d_doGridReset){
      move_particles=0.;
    }    
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew,pxx;
      constParticleVariable<Vector> pvelocity;
      constParticleVariable<Matrix3> psize;
      ParticleVariable<Vector> pvelocitynew;
      ParticleVariable<Matrix3> psizeNew;
      constParticleVariable<double> pmass, pTemperature;
      ParticleVariable<double> pmassNew,pvolume,pTempNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;
      ParticleVariable<double> pkineticEnergyDensity;
      constParticleVariable<Matrix3> pDeformationMeasure;


      // for thermal stress analysis
      constParticleVariable<double> pTempCurrent;
      ParticleVariable<double> pTempPreNew;
      
      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<double> gTemperatureRate, gTemperature, gTemperatureNoBC;
      constNCVariable<double> dTdt, massBurnFrac, frictionTempRate;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,                    lb->pXLabel,                     pset);
      old_dw->get(pdisp,                 lb->pDispLabel,                  pset);
      old_dw->get(pmass,                 lb->pMassLabel,                  pset);
      old_dw->get(pids,                  lb->pParticleIDLabel,            pset);
      old_dw->get(pvelocity,             lb->pVelocityLabel,              pset);
      old_dw->get(pTemperature,          lb->pTemperatureLabel,           pset);
      new_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel_preReloc, pset);


      // for thermal stress analysis
      new_dw->get(pTempCurrent,          lb->pTempCurrentLabel,           pset);
      new_dw->getModifiable(pvolume,     lb->pVolumeLabel_preReloc,       pset);
      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pxx,          lb->pXXLabel,                  pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);
      new_dw->allocateAndPut(pkineticEnergyDensity,
                                          lb->pKineticEnergyDensityLabel, pset);
      // for thermal stress analysis
      new_dw->allocateAndPut(pTempPreNew, lb->pTempPreviousLabel_preReloc, pset);

      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

      pids_new.copyData(pids);
      old_dw->get(psize,               lb->pSizeLabel,                 pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,        pset);
      psizeNew.copyData(psize);
      
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity_star,   lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      new_dw->get(gacceleration,    lb->gAccelerationLabel,   dwi,patch,gac,NGP);
      new_dw->get(gTemperatureRate, lb->gTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(gTemperature,     lb->gTemperatureLabel,    dwi,patch,gac,NGP);
      new_dw->get(gTemperatureNoBC, lb->gTemperatureNoBCLabel,dwi,patch,gac,NGP);
      new_dw->get(frictionTempRate, lb->frictionalWorkLabel,  dwi,patch,gac,NGP);    
      // for FractureMPM
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode, lb->pgCodeLabel, pset);
      constNCVariable<Vector> Gvelocity_star, Gacceleration;
      constNCVariable<double> GTemperatureRate, GTemperature, GTemperatureNoBC;
      new_dw->get(Gvelocity_star,   lb->GVelocityStarLabel,   dwi,patch,gac,NGP);
      new_dw->get(Gacceleration,    lb->GAccelerationLabel,   dwi,patch,gac,NGP);
      new_dw->get(GTemperatureRate, lb->GTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(GTemperature,     lb->GTemperatureLabel,    dwi,patch,gac,NGP);
      new_dw->get(GTemperatureNoBC, lb->GTemperatureNoBCLabel,dwi,patch,gac,NGP);

      if(flags->d_with_ice){
        new_dw->get(dTdt,            lb->dTdt_NCLabel,         dwi,patch,gac,NGP);
        new_dw->get(massBurnFrac,    lb->massBurnFractionLabel,dwi,patch,gac,NGP);
      }
      else{
        NCVariable<double> dTdt_create,massBurnFrac_create;
        new_dw->allocateTemporary(dTdt_create,                     patch,gac,NGP);
        new_dw->allocateTemporary(massBurnFrac_create,             patch,gac,NGP);
        dTdt_create.initialize(0.);
        massBurnFrac_create.initialize(0.);
        dTdt = dTdt_create;                         // reference created data
        massBurnFrac = massBurnFrac_create;         // reference created data
      }

      double Cp=mpm_matl->getSpecificHeat();
      double rho_init=mpm_matl->getInitialDensity();
      double rho_frac_min = 0.;
      if(m == RMI){
        rho_frac_min = .1;
      }

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx], pDeformationMeasure[idx]);

        Vector vel(0.0,0.0,0.0);
        Vector acc(0.0,0.0,0.0);
        double fricTempRate = 0.0;
        double tempRate = 0;
        double burnFraction = 0;

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          fricTempRate = frictionTempRate[node]*flags->d_addFrictionWork;
          if(pgCode[idx][k]==1) {
             vel      += gvelocity_star[node]  * S[k];
             acc      += gacceleration[node]   * S[k];
             tempRate += (gTemperatureRate[node]+dTdt[node]+fricTempRate)*S[k];
             burnFraction += massBurnFrac[node] * S[k];
          }
          else if(pgCode[idx][k]==2) {
             vel      += Gvelocity_star[node]  * S[k];
             acc      += Gacceleration[node]   * S[k];
             tempRate += (GTemperatureRate[node]+dTdt[node]+fricTempRate)*S[k];
             burnFraction += massBurnFrac[node] * S[k];
          }
        }

        // Update the particle's position and velocity
        pxnew[idx]           = px[idx] + vel*delT*move_particles;
        pdispnew[idx]        = pdisp[idx] + vel*delT;
        pvelocitynew[idx]    = pvelocity[idx] + acc*delT;
        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx]    + pdispnew[idx];      
        pTempNew[idx]        = pTemperature[idx] + tempRate*delT;
        pTempPreNew[idx]     = pTempCurrent[idx]; // for thermal stress

        if (cout_heat.active()) {
          cout_heat << "FractureMPM::Particle = " << idx
                    << " T_old = " << pTemperature[idx]
                    << " Tdot = " << tempRate
                    << " dT = " << (tempRate*delT)
                    << " T_new = " << pTempNew[idx] << endl;
        }
        

        double rho;
        if(pvolume[idx] > 0.){
          rho = pmass[idx]/pvolume[idx];
        }
        else{
          rho = rho_init;
        }
        pkineticEnergyDensity[idx]=0.5*rho*pvelocitynew[idx].length2();
        pmassNew[idx]   = Max(pmass[idx]*(1.-burnFraction),0.);
        pvolume[idx]    = pmassNew[idx]/rho;
            
        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
        ke += .5*pmass[idx]*pvelocitynew[idx].length2();
        CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
        totalMom += pvelocitynew[idx]*pmass[idx];
      }

      // Delete particles whose mass is too small (due to combustion)
      // For particles whose new velocity exceeds a maximum set in the input
      // file, set their velocity back to the velocity that it came into
      // this step with
      for(ParticleSubset::iterator iter  = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        if(pmassNew[idx] <= flags->d_min_part_mass){
          delset->addParticle(idx);
        }
        if(pvelocitynew[idx].length() > flags->d_max_vel){
          pvelocitynew[idx]=pvelocity[idx];
        }
      }
      
      new_dw->deleteParticles(delset);
      //__________________________________
      //  particle debugging label-- carry forward
      if (flags->d_with_color) {
        constParticleVariable<double> pColor;
        ParticleVariable<double>pColor_new;
        old_dw->get(pColor, lb->pColorLabel, pset);
        new_dw->allocateAndPut(pColor_new, lb->pColorLabel_preReloc, pset);
        pColor_new.copyData(pColor);
      }
    }

    // DON'T MOVE THESE!!!
    new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
    new_dw->put(sum_vartype(thermal_energy), lb->ThermalEnergyLabel);
    new_dw->put(sumvec_vartype(CMX),         lb->CenterOfMassPositionLabel);
    new_dw->put(sumvec_vartype(totalMom),    lb->TotalMomentumLabel);
    
    // cout << "Solid mass lost this timestep = " << massLost << endl;
    // cout << "Solid momentum after advection = " << totalMom << endl;
    
    // cout << "THERMAL ENERGY " << thermal_energy << endl;
    
    delete interpolator;
  }
}    
      
void
FractureMPM::initialErrorEstimate(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* /*matls*/,
                                  DataWarehouse*,
                                  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (amr_doing.active())
      amr_doing << "Doing FractureMPM::initialErrorEstimate on patch "<< patch->getID()<< endl;

    CCVariable<int> refineFlag;
    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->getModifiable(refineFlag, d_sharedState->get_refineFlag_label(),
                          0, patch);
    new_dw->get(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(),
                0, patch);

    PatchFlag* refinePatch = refinePatchFlag.get().get_rep();


    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Loop over particles
      ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
      constParticleVariable<Point> px;
      new_dw->get(px, lb->pXLabel, pset);

      for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
        refineFlag[patch->getLevel()->getCellIndex(px[*iter])] = true;
        refinePatch->set();
      }
    }
  }
}

void
FractureMPM::errorEstimate(const ProcessorGroup* group,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  // coarsen the errorflag.
  
  if (cout_doing.active())
    cout_doing << "Doing FractureMPM::errorEstimate" << '\n';
  
  const Level* level = getLevel(patches);
  if (level->getIndex() == level->getGrid()->numLevels()-1) {
    // on finest level, we do the same thing as initialErrorEstimate, so call it
    initialErrorEstimate(group, patches, matls, old_dw, new_dw);
  }
  else {
    const Level* fineLevel = level->getFinerLevel().get_rep();
        
    for(int p=0;p<patches->size();p++){
      const Patch* coarsePatch = patches->get(p);
        
      if (amr_doing.active())
        amr_doing << "Doing FractureMPM::errorEstimate on patch " << coarsePatch->getID() << endl;
        
      // Find the overlapping regions...
        
      CCVariable<int> refineFlag;
      PerPatch<PatchFlagP> refinePatchFlag;

      new_dw->getModifiable(refineFlag, d_sharedState->get_refineFlag_label(),
                            0, coarsePatch);
      new_dw->get(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(),
                  0, coarsePatch);
        
      PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
        
      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches);

      for(int i=0;i<finePatches.size();i++){
        const Patch* finePatch = finePatches[i];

        // Get the particle data
        constCCVariable<int> fineErrorFlag;
        new_dw->get(fineErrorFlag, d_sharedState->get_refineFlag_label(), 0,
                    finePatch, Ghost::None, 0);

        IntVector fl(finePatch->getExtraCellLowIndex());
        IntVector fh(finePatch->getExtraCellHighIndex());
        IntVector l(fineLevel->mapCellToCoarser(fl));
        IntVector h(fineLevel->mapCellToCoarser(fh));
        l = Max(l, coarsePatch->getExtraCellLowIndex());
        h = Min(h, coarsePatch->getExtraCellHighIndex());

        for(CellIterator iter(l, h); !iter.done(); iter++){
          IntVector fineStart(level->mapCellToFiner(*iter));

          for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
              !inside.done(); inside++){
            if (fineErrorFlag[fineStart+*inside]) {
              refineFlag[*iter] = 1;
              refinePatch->set();
            }
          }
        }
      }  // fine patch loop
    } // coarse patch loop 
  }
}

void
FractureMPM::refine(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* /*matls*/,
                    DataWarehouse*,
                    DataWarehouse* new_dw)
{
  // just create a particle subset if one doesn't exist
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
        
      if (cout_doing.active()) {
        cout_doing <<"Doing refine on patch "
        << patch->getID() << " material # = " << dwi << endl;
      }
        
      // this is a new patch, so create empty particle variables.
      if (!new_dw->haveParticleSubset(dwi, patch)) {
        ParticleSubset* pset = new_dw->createParticleSubset(0, dwi, patch);
        
        // Create arrays for the particle data
        ParticleVariable<Point>  px;
        ParticleVariable<double> pmass, pvolume, pTemperature;
        ParticleVariable<Vector> pvelocity, pexternalforce, psize, pdisp;
        ParticleVariable<int>    pLoadCurve;
        ParticleVariable<long64> pID;
        ParticleVariable<Matrix3> pdeform, pstress;

        new_dw->allocateAndPut(px,             lb->pXLabel,             pset);
        new_dw->allocateAndPut(pmass,          lb->pMassLabel,          pset);
        new_dw->allocateAndPut(pvolume,        lb->pVolumeLabel,        pset);
        new_dw->allocateAndPut(pvelocity,      lb->pVelocityLabel,      pset);
        new_dw->allocateAndPut(pTemperature,   lb->pTemperatureLabel,   pset);
        new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pID,            lb->pParticleIDLabel,    pset);
        new_dw->allocateAndPut(pdisp,          lb->pDispLabel,          pset);
        new_dw->allocateAndPut(pdeform,        lb->pDeformationMeasureLabel, pset);
        new_dw->allocateAndPut(pstress,        lb->pStressLabel,        pset);
        if (flags->d_useLoadCurves)
                  new_dw->allocateAndPut(pLoadCurve,   lb->pLoadCurveIDLabel,   pset);
        new_dw->allocateAndPut(psize,          lb->pSizeLabel,          pset);

      }
    }
  }

} // end refine()
