#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/NormalForceBC.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

#undef KUMAR
//#define KUMAR

using namespace Uintah;
using namespace SCIRun;

using namespace std;

#define MAX_BASIS 27

static DebugStream cout_doing("MPM", false);
static DebugStream cout_dbg("SerialMPM", false);
static DebugStream cout_heat("MPMHeat", false);
static DebugStream amr_doing("AMRMPM", false);

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;


SerialMPM::SerialMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  flags = scinew MPMFlags();

  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=1e-200;
  d_with_ice    = false;
  contactModel        = 0;
  thermalContactModel = 0;
  d_min_part_mass = 3.e-15;
  d_max_vel = 3.e105;
  NGP     = 1;
  NGN     = 1;
  d_doGridReset = true;
  d_recompile = false;
  dataArchiver = 0;
}

SerialMPM::~SerialMPM()
{
  delete lb;
  delete flags;
  delete contactModel;
  delete thermalContactModel;
  MPMPhysicalBCFactory::clean();
}

void SerialMPM::problemSetup(const ProblemSpecP& prob_spec,GridP&,
                             SimulationStateP& sharedState)
{
  d_sharedState = sharedState;

  ProblemSpecP mpm_soln_ps = prob_spec->findBlock("MPM");

  if(mpm_soln_ps) {
    flags->readMPMFlags(mpm_soln_ps);
    mpm_soln_ps->get("do_grid_reset", d_doGridReset);
    mpm_soln_ps->get("minimum_particle_mass",    d_min_part_mass);
    mpm_soln_ps->get("maximum_particle_velocity",d_max_vel);
    
    std::vector<std::string> bndy_face_txt_list;
    mpm_soln_ps->get("boundary_traction_faces", bndy_face_txt_list);
    
    // convert text representation of face into FaceType
    for(std::vector<std::string>::const_iterator ftit(bndy_face_txt_list.begin());
	ftit!=bndy_face_txt_list.end();ftit++) {
        Patch::FaceType face = Patch::invalidFace;
        for(Patch::FaceType ft=Patch::startFace;ft<=Patch::endFace;ft=Patch::nextFace(ft)) {
          if(Patch::getFaceName(ft)==*ftit) face =  ft;
        }
        if(face!=Patch::invalidFace) {
          d_bndy_traction_faces.push_back(face);
        } else {
          std::cerr << "warning: ignoring unknown face '" << *ftit<< "'" << std::endl;
        }
    }
  }

  if(flags->d_canAddMPMMaterial){
    cout << "Addition of new material for particle failure is possible"<< endl; 
    if(!flags->d_addNewMaterial){
      throw ProblemSetupException("To use material addition, one must specify manual_add_material==true in the input file.");
    }
  }
  
  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(flags->d_8or27==MAX_BASIS){
    NGP=2;
    NGN=2;
  }

  //__________________________________
  // Grab time_integrator, default is explicit
  string integrator_type = "explicit";
  d_integrator = Explicit;
  if (mpm_soln_ps ) {
    mpm_soln_ps->get("time_integrator",flags->d_integrator_type);
    if (flags->d_integrator_type == "implicit"){
      throw ProblemSetupException("Can't use implicit integration with -mpm");
    }
    if (flags->d_integrator_type == "explicit") {
      d_integrator = Explicit;
    }
    if (flags->d_integrator_type == "fracture") {
      d_integrator = Fracture;
      flags->d_fracture = true;
    }
  }

  //  cout << "d_fracture = " << flags->d_fracture << endl;

  MPMPhysicalBCFactory::create(prob_spec);

  contactModel = ContactFactory::create(prob_spec,sharedState,lb,flags);
  thermalContactModel =
    ThermalContactFactory::create(prob_spec, sharedState, lb,flags);

  ProblemSpecP p = prob_spec->findBlock("DataArchiver");
  if(!p->get("outputInterval", d_outputInterval))
    d_outputInterval = 1.0;

  materialProblemSetup(prob_spec, d_sharedState, lb, flags);
}

void SerialMPM::addMaterial(const ProblemSpecP& prob_spec,GridP&,
                            SimulationStateP& sharedState)
{
  // For adding materials mid-Simulation
  d_recompile = true;
  ProblemSpecP mat_ps =  prob_spec->findBlock("AddMaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    //Create and register as an MPM material
    MPMMaterial *mat = scinew MPMMaterial(ps, lb, flags);
    sharedState->registerMPMMaterial(mat);
  }
}

void 
SerialMPM::materialProblemSetup(const ProblemSpecP& prob_spec, 
                                SimulationStateP& sharedState,
                                MPMLabel* lb, MPMFlags* flags)
{
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {

    //Create and register as an MPM material
    MPMMaterial *mat = scinew MPMMaterial(ps, lb, flags);
    sharedState->registerMPMMaterial(mat);

    // If new particles are to be created, create a copy of each material
    // without the associated geometry
    if (flags->d_createNewParticles) {
      MPMMaterial *mat_copy = scinew MPMMaterial();
      mat_copy->copyWithoutGeom(mat, flags);    
      sharedState->registerMPMMaterial(mat_copy);
    }
  }
}

void SerialMPM::scheduleInitialize(const LevelP& level,
                                   SchedulerP& sched)
{
  Task* t = scinew Task("MPM::actuallyInitialize",
                        this, &SerialMPM::actuallyInitialize);

  MaterialSubset* zeroth_matl = scinew MaterialSubset();
  zeroth_matl->add(0);
  zeroth_matl->addReference();

  t->computes(lb->partCountLabel);
  t->computes(lb->pXLabel);
  t->computes(lb->pDispLabel);
  t->computes(lb->pMassLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pInternalHeatRateLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pErosionLabel);
  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->pCellNAPIDLabel,zeroth_matl);
  
  // Debugging Scalar
  if (flags->d_with_color) {
    t->computes(lb->pColorLabel);
  }

  if (flags->d_useLoadCurves) {
    // Computes the load curve ID associated with each particle
    t->computes(lb->pLoadCurveIDLabel);
  }

  if (flags->d_accStrainEnergy) {
    // Computes accumulated strain energy
    t->computes(lb->AccStrainEnergyLabel);
  }

  // artificial damping coeff initialized to 0.0
  cout_dbg << "Artificial Damping Coeff = " << flags->d_artificialDampCoeff 
             << " 8 or 27 = " << flags->d_8or27 << endl;
  if (flags->d_artificialDampCoeff > 0.0) {
    t->computes(lb->pDampingRateLabel); 
    t->computes(lb->pDampingCoeffLabel); 
  }

  int numMPM = d_sharedState->getNumMPMMatls();
  const PatchSet* patches = level->eachPatch();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
  }

  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  schedulePrintParticleCount(level, sched);

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...

  if (flags->d_useLoadCurves) {
    // Schedule the initialization of pressure BCs per particle
    scheduleInitializePressureBCs(level, sched);
  }
}

void SerialMPM::scheduleInitializeAddedMaterial(const LevelP& level,
                                                SchedulerP& sched)
{
  cout_doing << "Doing SerialMPM::scheduleInitializeAddedMaterial " << endl;
  Task* t = scinew Task("SerialMPM::actuallyInitializeAddedMaterial",
                  this, &SerialMPM::actuallyInitializeAddedMaterial);
                                                                                
  int numALLMatls = d_sharedState->getNumMatls();
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
  t->computes(lb->pInternalHeatRateLabel,  add_matl);
  t->computes(lb->pVelocityLabel,          add_matl);
  t->computes(lb->pExternalForceLabel,     add_matl);
  t->computes(lb->pParticleIDLabel,        add_matl);
  t->computes(lb->pDeformationMeasureLabel,add_matl);
  t->computes(lb->pStressLabel,            add_matl);
  t->computes(lb->pSizeLabel,              add_matl);
  t->computes(lb->pErosionLabel,           add_matl);

  if (flags->d_accStrainEnergy) {
    // Computes accumulated strain energy
    t->computes(lb->AccStrainEnergyLabel);
  }

  const PatchSet* patches = level->eachPatch();

  MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(numALLMatls-1);
  ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
  cm->addInitialComputesAndRequires(t, mpm_matl, patches);

  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  // The task will have a reference to add_matl
  if (add_matl->removeReference())
    delete add_matl; // shouln't happen, but...
}

void SerialMPM::schedulePrintParticleCount(const LevelP& level, 
                                           SchedulerP& sched)
{
  Task* t = scinew Task("MPM::printParticleCount",
                        this, &SerialMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
}

void SerialMPM::scheduleInitializePressureBCs(const LevelP& level,
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
    Task* t = scinew Task("MPM::countMaterialPointsPerLoadCurve",
                          this, &SerialMPM::countMaterialPointsPerLoadCurve);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel, Ghost::None);
    t->computes(lb->materialPointsPerLoadCurveLabel, loadCurveIndex,
                Task::OutOfDomain);
    sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

    // Create a task that calculates the force to be associated with
    // each particle based on the pressure BCs
    t = scinew Task("MPM::initializePressureBC",
                    this, &SerialMPM::initializePressureBC);
    t->requires(Task::NewDW, lb->pXLabel, Ghost::None);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel, Ghost::None);
    t->requires(Task::NewDW, lb->materialPointsPerLoadCurveLabel, loadCurveIndex, Task::OutOfDomain, Ghost::None);
    t->modifies(lb->pExternalForceLabel);
    sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
  }
}

void SerialMPM::scheduleComputeStableTimestep(const LevelP&,
                                              SchedulerP&)
{
  // Nothing to do here - delt is computed as a by-product of the
  // consitutive model
}

void
SerialMPM::scheduleTimeAdvance(const LevelP & level,
                               SchedulerP   & sched,
                               int, int ) // AMR Parameters
{
  if (flags->d_finestLevelOnly &&
      level->getIndex() != level->getGrid()->numLevels()-1)
    return;
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  scheduleApplyExternalLoads(             sched, patches, matls);
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  scheduleComputeHeatExchange(            sched, patches, matls);
  scheduleExMomInterpolated(              sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);
  scheduleComputeInternalForce(           sched, patches, matls);
  scheduleComputeInternalHeatRate(        sched, patches, matls);
  scheduleSolveEquationsMotion(           sched, patches, matls);
  scheduleSolveHeatEquations(             sched, patches, matls);
  scheduleIntegrateAcceleration(          sched, patches, matls);
  scheduleIntegrateTemperatureRate(       sched, patches, matls);
  scheduleExMomIntegrated(                sched, patches, matls);
  scheduleSetGridBoundaryConditions(      sched, patches, matls);
  scheduleCalculateDampingRate(           sched, patches, matls);
  scheduleAddNewParticles(                sched, patches, matls);
  scheduleConvertLocalizedParticles(      sched, patches, matls);
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);

  if(flags->d_canAddMPMMaterial){
    //  This checks to see if the model on THIS patch says that it's
    //  time to add a new material
    scheduleCheckNeedAddMPMMaterial(         sched, patches, matls);
                                                                                
    //  This one checks to see if the model on ANY patch says that it's
    //  time to add a new material
    scheduleSetNeedAddMaterialFlag(         sched, level,   matls);
  }

  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    lb->d_particleState_preReloc,
                                    lb->pXLabel, lb->d_particleState,
                                    lb->pParticleIDLabel, matls);
}

void SerialMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
 /*
  * applyExternalLoads
  *   in(p.externalForce, p.externalheatrate)
  *   out(p.externalForceNew, p.externalheatrateNew) */
  Task* t=scinew Task("MPM::applyExternalLoads",
                    this, &SerialMPM::applyExternalLoads);
                  
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

void SerialMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
  /* interpolateParticlesToGrid
   *   in(P.MASS, P.VELOCITY, P.NAT_X)
   *   operation(interpolate the P.MASS and P.VEL to the grid
   *             using P.NAT_X and some shape function evaluations)
   *   out(G.MASS, G.VELOCITY) */


  Task* t = scinew Task("MPM::interpolateParticlesToGrid",
                        this,&SerialMPM::interpolateParticlesToGrid);
  Ghost::GhostType  gan = Ghost::AroundNodes;
  t->requires(Task::OldDW, lb->pMassLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pVolumeLabel,           gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,         gan,NGP);
  t->requires(Task::OldDW, lb->pXLabel,                gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      gan,NGP);
  t->requires(Task::OldDW, lb->pSp_volLabel,           gan,NGP); 
  t->requires(Task::OldDW, lb->pErosionLabel,          gan, NGP);
  
  if(flags->d_8or27==27){
    t->requires(Task::OldDW,lb->pSizeLabel,             gan,NGP);
  }
  //t->requires(Task::OldDW, lb->pExternalHeatRateLabel, gan,NGP);

  t->computes(lb->gMassLabel);
  t->computes(lb->gMassLabel,        d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
  t->computes(lb->gTemperatureLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
  t->computes(lb->gVelocityLabel,    d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
  t->computes(lb->gSp_volLabel);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureNoBCLabel);
  t->computes(lb->gExternalHeatRateLabel);
  t->computes(lb->gNumNearParticlesLabel);
  t->computes(lb->TotalMassLabel);
  t->computes(lb->TotalVolumeDeformedLabel);
  
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

  cout_doing << getpid() << " Doing MPM::ThermalContact::computeHeatExchange "
             << endl;
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

/////////////////////////////////////////////////////////////////////////
/*!  **WARNING** In addition to the stresses and deformations, the internal 
 *               heat rate in the particles (pInternalHeatRateLabel) 
 *               is computed here */
/////////////////////////////////////////////////////////////////////////
void SerialMPM::scheduleComputeStressTensor(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("MPM::computeStressTensor",
                        this, &SerialMPM::computeStressTensor);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches);
  }

  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->StrainEnergyLabel);

  sched->addTask(t, patches, matls);

  // Schedule update of the erosion parameter
  scheduleUpdateErosionParameter(sched, patches, matls);

  if (flags->d_accStrainEnergy) 
    scheduleComputeAccStrainEnergy(sched, patches, matls);

  if(flags->d_artificial_viscosity){
    scheduleComputeArtificialViscosity(   sched, patches, matls);
  }
}

// Compute the accumulated strain energy
void SerialMPM::scheduleUpdateErosionParameter(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls)
{
  Task* t = scinew Task("MPM::updateErosionParameter",
                        this, &SerialMPM::updateErosionParameter);
  t->requires(Task::OldDW, lb->pErosionLabel,          Ghost::None);
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addRequiresDamageParameter(t, mpm_matl, patches);
  }
  t->computes(lb->pErosionLabel_preReloc);
  sched->addTask(t, patches, matls);
}

// Compute the accumulated strain energy
void SerialMPM::scheduleComputeAccStrainEnergy(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls)
{
  Task* t = scinew Task("MPM::computeAccStrainEnergy",
                        this, &SerialMPM::computeAccStrainEnergy);
  t->requires(Task::OldDW, lb->AccStrainEnergyLabel);
  t->requires(Task::NewDW, lb->StrainEnergyLabel);
  t->computes(lb->AccStrainEnergyLabel);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeArtificialViscosity(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
  Task* t = scinew Task("MPM::computeArtificialViscosity",
                        this, &SerialMPM::computeArtificialViscosity);

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, lb->pXLabel,                 Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,              Ghost::None);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,    Ghost::None);

  if(flags->d_8or27==27){
    t->requires(Task::OldDW,lb->pSizeLabel,             Ghost::None);
  }

  t->requires(Task::NewDW,lb->gVelocityLabel, gac, NGN);
  t->computes(lb->p_qLabel);

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

  Task* t = scinew Task("MPM::computeInternalForce",
                        this, &SerialMPM::computeInternalForce);

  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gMassLabel, gnone);
  t->requires(Task::NewDW,lb->gMassLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain, gnone);
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,      gan,NGP);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,       gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::OldDW,lb->pMassLabel,                 gan,NGP);
  if(flags->d_8or27==27){
    t->requires(Task::OldDW, lb->pSizeLabel,              gan,NGP);
  }
  t->requires(Task::NewDW, lb->pErosionLabel_preReloc,    gan, NGP);

  if(d_with_ice){
    t->requires(Task::NewDW, lb->pPressureLabel,          gan,NGP);
  }

  if(flags->d_artificial_viscosity){
    t->requires(Task::NewDW, lb->p_qLabel,                gan,NGP);
  }

  t->computes(lb->gInternalForceLabel);
  
  if (!d_bndy_traction_faces.empty()) {
     
    for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
        ftit!=d_bndy_traction_faces.end();ftit++) {
      t->computes(lb->BndyForceLabel[*ftit]);       // node based
      t->computes(lb->BndyContactAreaLabel[*ftit]); // node based
    }
    
  }
  
  t->computes(lb->gStressForSavingLabel);
  t->computes(lb->gStressForSavingLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);

  sched->addTask(t, patches, matls);
}


void SerialMPM::scheduleComputeInternalHeatRate(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{  
  Task* t = scinew Task("MPM::computeInternalHeatRate",
                        this, &SerialMPM::computeInternalHeatRate);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::OldDW, lb->pXLabel,                         gan, NGP);
  if(flags->d_8or27==27){
    t->requires(Task::OldDW, lb->pSizeLabel,                    gan, NGP);
  }
  t->requires(Task::OldDW, lb->pMassLabel,                      gan, NGP);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,            gan, NGP);
  t->requires(Task::NewDW, lb->pInternalHeatRateLabel_preReloc, gan, NGP);
  t->requires(Task::NewDW, lb->pErosionLabel_preReloc,          gan, NGP);
  t->requires(Task::NewDW, lb->gTemperatureLabel,               gac, 2*NGP);
  t->requires(Task::NewDW, lb->gMassLabel,                      gnone);

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

  Task* t = scinew Task("MPM::solveEquationsMotion",
                        this, &SerialMPM::solveEquationsMotion);

  t->requires(Task::OldDW, d_sharedState->get_delt_label());

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  //Uncomment  the next line to use damping
  //t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);     

  if(d_with_ice){
    t->requires(Task::NewDW, lb->gradPAccNCLabel,   Ghost::None);
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

  Task* t = scinew Task("MPM::solveHeatEquations",
                        this, &SerialMPM::solveHeatEquations);

  const MaterialSubset* mss = matls->getUnion();

  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gMassLabel,                           gnone);
  t->requires(Task::NewDW, lb->gVolumeLabel,                         gnone);
  t->requires(Task::NewDW, lb->gExternalHeatRateLabel,               gnone);
  t->modifies(             lb->gInternalHeatRateLabel,               mss);
  t->requires(Task::NewDW, lb->gThermalContactHeatExchangeRateLabel, gnone);
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

  Task* t = scinew Task("MPM::integrateAcceleration",
                        this, &SerialMPM::integrateAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

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

  Task* t = scinew Task("MPM::integrateTemperatureRate",
                        this, &SerialMPM::integrateTemperatureRate);

  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gTemperatureLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel, Ghost::None);
  t->modifies(             lb->gTemperatureRateLabel, mss);

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
   *              velocity fields to feel the influence of the
   *              the others according to specific rules)
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
  Task* t=scinew Task("MPM::setGridBoundaryConditions",
                      this, &SerialMPM::setGridBoundaryConditions);
                  
  const MaterialSubset* mss = matls->getUnion();
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleCalculateDampingRate(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  /*
   * calculateDampingRate
   *   in(G.VELOCITY_STAR, P.X, P.Size)
   *   operation(Calculate the interpolated particle velocity and
   *             sum the squares of the velocities over particles)
   *   out(sum_vartpe(dampingRate)) 
   */
  if (flags->d_artificialDampCoeff > 0.0) {
    Task* t=scinew Task("MPM::calculateDampingRate", this, 
                        &SerialMPM::calculateDampingRate);
    t->requires(Task::NewDW, lb->gVelocityStarLabel, Ghost::AroundCells, NGN);
    t->requires(Task::OldDW, lb->pXLabel, Ghost::None);
    if(flags->d_8or27==27) 
      t->requires(Task::OldDW, lb->pSizeLabel, Ghost::None);

    t->computes(lb->pDampingRateLabel);
    sched->addTask(t, patches, matls);
  }
}

void SerialMPM::scheduleAddNewParticles(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{

  if (!flags->d_addNewMaterial || flags->d_createNewParticles) return;

  Task* t=scinew Task("MPM::addNewParticles", this, 
                      &SerialMPM::addNewParticles);

  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    mpm_matl->getParticleCreator()->allocateVariablesAddRequires(t, mpm_matl,
                                                                 patches, lb);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->allocateCMDataAddRequires(t,mpm_matl,patches,lb);
    cm->addRequiresDamageParameter(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, matls);
}


void SerialMPM::scheduleConvertLocalizedParticles(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{
  if (!flags->d_createNewParticles || flags->d_addNewMaterial) return;

  Task* t=scinew Task("MPM::convertLocalizedParticles", this, 
                      &SerialMPM::convertLocalizedParticles);

  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m+=2){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    mpm_matl->getParticleCreator()->allocateVariablesAddRequires(t, mpm_matl,
                                                                 patches, lb);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->allocateCMDataAddRequires(t,mpm_matl,patches,lb);
    cm->addRequiresDamageParameter(t, mpm_matl, patches);
  }

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

  Task* t=scinew Task("MPM::interpolateToParticlesAndUpdate",
                      this, &SerialMPM::interpolateToParticlesAndUpdate);


  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gAccelerationLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,           gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureLabel,               gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel,           gac,NGN);
  t->requires(Task::NewDW, lb->frictionalWorkLabel,             gac,NGN);
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                      gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,               gnone);
  t->requires(Task::OldDW, lb->pSp_volLabel,                    gnone); 
  t->requires(Task::OldDW, lb->pVelocityLabel,                  gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                      gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,            gnone);
  t->requires(Task::NewDW, lb->pErosionLabel_preReloc,          gnone);

  // The dampingCoeff (alpha) is 0.0 for standard usage, otherwise
  // it is determined by the damping rate if the artificial damping
  // coefficient Q is greater than 0.0
  if (flags->d_artificialDampCoeff > 0.0) {
    t->requires(Task::OldDW, lb->pDampingCoeffLabel);
    t->requires(Task::NewDW, lb->pDampingRateLabel);
    t->computes(lb->pDampingCoeffLabel);
  }

  if(d_with_ice){
    t->requires(Task::NewDW, lb->dTdt_NCLabel,         gac,NGN);
    t->requires(Task::NewDW, lb->massBurnFractionLabel,gac,NGN);
  }

  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pSp_volLabel_preReloc);
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pSizeLabel_preReloc);
  t->computes(lb->pXXLabel);

  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->ThermalEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->CenterOfMassVelocityLabel);
  
  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }
  
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleRefine(const LevelP& fineLevel, 
                               SchedulerP& sched)
{
  Task* task = scinew Task("SerialMPM::refine", this, &SerialMPM::refine);
  sched->addTask(task, fineLevel->eachPatch(), d_sharedState->allMPMMaterials());
  // do nothing for now
}

void SerialMPM::scheduleRefineInterface(const LevelP& /*fineLevel*/, 
                                        SchedulerP& /*scheduler*/,
                                        int /*step*/, int /*nsteps*/)
{
  // do nothing for now
}

void SerialMPM::scheduleCoarsen(const LevelP& /*coarseLevel*/, 
                                SchedulerP& /*sched*/)
{
  // do nothing for now
}

/// Schedule to mark flags for AMR regridding
void SerialMPM::scheduleErrorEstimate(const LevelP& coarseLevel,
                                      SchedulerP& sched)
{
  // main way is to count particles, but for now we only want particles on
  // the finest level.  Thus to schedule cells for regridding during the 
  // execution, we'll coarsen the flagged cells (see coarsen).

  cout_doing << "SerialMPM::scheduleErrorEstimate on level " << coarseLevel->getIndex() << '\n';
  // Estimate error - this should probably be in it's own schedule,
  // and the simulation controller should not schedule it every time step
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task* task = scinew Task("errorEstimate", this, &SerialMPM::errorEstimate);
  
  // if the finest level, compute flagged cells
  if (coarseLevel->getIndex() == coarseLevel->getGrid()->numLevels()-1) {
    task->requires(Task::NewDW, lb->pXLabel,     gac, 0);
  }
  else {
    task->requires(Task::NewDW, d_sharedState->get_refineFlag_label(),
                   0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0);
  }
  task->modifies(d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials());
  task->modifies(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials());
  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allMPMMaterials());

}

/// Schedule to mark initial flags for AMR regridding
void SerialMPM::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                             SchedulerP& sched)
{
  cout_doing << "SerialMPM::scheduleErrorEstimate on level " << coarseLevel->getIndex() << '\n';
  // Estimate error - this should probably be in it's own schedule,
  // and the simulation controller should not schedule it every time step
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task* task = scinew Task("errorEstimate", this, &SerialMPM::initialErrorEstimate);
  task->requires(Task::NewDW, lb->pXLabel,     gac, 0);

  task->modifies(d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials());
  task->modifies(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials());
  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allMPMMaterials());
}

void SerialMPM::printParticleCount(const ProcessorGroup* pg,
                                   const PatchSubset*,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  sumlong_vartype pcount;
  new_dw->get(pcount, lb->partCountLabel);
  
  if(pg->myrank() == 0){
    static bool printed=false;
    if(!printed){
      cerr << "Created " << (long) pcount << " total particles\n";
      printed=true;
    }
  }
}

void SerialMPM::computeAccStrainEnergy(const ProcessorGroup*,
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
void SerialMPM::countMaterialPointsPerLoadCurve(const ProcessorGroup*,
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
void SerialMPM::initializePressureBC(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* ,
                                     DataWarehouse* new_dw)
{
  // Get the current time
  double time = 0.0;
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
      cout_dbg << "    Load Curve = " << nofPressureBCs 
                 << " Num Particles = " << numPart << endl;

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
              pExternalForce[idx] = pbc->getForceVector(px[idx], forcePerPart);
            }
          }
        } // matl loop
      }  // patch loop
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
        //__________________________________
        //  hardwiring for Northrup Grumman nozzle   
        #define SerialMPM_1
        #include "../MPMICE/NGC_nozzle.i"
        #undef SerialMPM_1
      }

    }
  }

  if (flags->d_accStrainEnergy) {
    // Initialize the accumulated strain energy
    new_dw->put(max_vartype(0.0), lb->AccStrainEnergyLabel);
  }

  // Initialize the artificial damping ceofficient (alpha) to zero
  if (flags->d_artificialDampCoeff > 0.0) {
    double alpha = 0.0;    
    double alphaDot = 0.0;    
    new_dw->put(max_vartype(alpha), lb->pDampingCoeffLabel);
    new_dw->put(sum_vartype(alphaDot), lb->pDampingRateLabel);
  }

  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);
}


void SerialMPM::actuallyInitializeAddedMaterial(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* matls,
                                                DataWarehouse*,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing actuallyInitializeAddedMaterial on patch "
               << patch->getID() <<"\t\t\t MPM"<< endl;

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    cout << "num Matls = " << numMPMMatls << endl;
    CCVariable<short int> cellNAPID;
    int m=numMPMMatls-1;
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    particleIndex numParticles = mpm_matl->countParticles(patch);

    mpm_matl->createParticles(numParticles, cellNAPID, patch, new_dw);

    mpm_matl->getConstitutiveModel()->initializeCMData(patch,
                                                       mpm_matl,
                                                       new_dw);
  }
}


void SerialMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
                                              const PatchSubset*,
                                              const MaterialSubset*,
                                              DataWarehouse*,
                                              DataWarehouse*)
{
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing interpolateParticlesToGrid on patch " << patch->getID()
               <<"\t\t MPM"<< endl;

    int numMatls = d_sharedState->getNumMPMMatls();

    NCVariable<double> gmassglobal,gtempglobal;
    NCVariable<Vector> gvelglobal;
    new_dw->allocateAndPut(gmassglobal, lb->gMassLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gtempglobal, lb->gTemperatureLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gvelglobal, lb->gVelocityLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    gmassglobal.initialize(d_SMALL_NUM_MPM);
    gtempglobal.initialize(0.0);
    gvelglobal.initialize(Vector(0.0));

    Ghost::GhostType  gan = Ghost::AroundNodes;
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature, pSp_vol;
      constParticleVariable<Vector> pvelocity, pexternalforce,psize;
      constParticleVariable<double> pErosion;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pSp_vol,        lb->pSp_volLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);
      new_dw->get(pexternalforce, lb->pExtForceLabel_preReloc, pset);
      if(flags->d_8or27==27){
        old_dw->get(psize,        lb->pSizeLabel,          pset);
      }
      old_dw->get(pErosion,       lb->pErosionLabel,       pset);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> gexternalforce;
      NCVariable<double> gexternalheatrate;
      NCVariable<double> gTemperature;
      NCVariable<double> gSp_vol;
      NCVariable<double> gTemperatureNoBC;
      NCVariable<double> gnumnearparticles;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,       dwi,patch);
      new_dw->allocateAndPut(gSp_vol,          lb->gSp_volLabel,     dwi,patch);
      new_dw->allocateAndPut(gvolume,          lb->gVolumeLabel,     dwi,patch);
      new_dw->allocateAndPut(gvelocity,        lb->gVelocityLabel,   dwi,patch);
      new_dw->allocateAndPut(gTemperature,     lb->gTemperatureLabel,dwi,patch);
      new_dw->allocateAndPut(gTemperatureNoBC, lb->gTemperatureNoBCLabel,
                             dwi,patch);
      new_dw->allocateAndPut(gexternalforce,   lb->gExternalForceLabel,
                             dwi,patch);
      new_dw->allocateAndPut(gexternalheatrate,lb->gExternalHeatRateLabel,
                             dwi,patch);
      new_dw->allocateAndPut(gnumnearparticles,lb->gNumNearParticlesLabel,
                             dwi,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);
      gTemperatureNoBC.initialize(0);
      gexternalheatrate.initialize(0);
      gnumnearparticles.initialize(0.);
      gSp_vol.initialize(0.);

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity
      
      double totalmass = 0;
      Vector total_mom(0.0,0.0,0.0);

      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];
      Vector pmom;
      int n8or27=flags->d_8or27;

      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        if(n8or27==8){
          patch->findCellAndWeights(px[idx], ni, S);
        }
        else if(n8or27==27){
          patch->findCellAndWeights27(px[idx], ni, S, psize[idx]);
        }

        pmom = pvelocity[idx]*pmass[idx];
        total_mom += pvelocity[idx]*pmass[idx];

        // Add each particles contribution to the local mass & velocity 
        // Must use the node indices
        for(int k = 0; k < n8or27; k++) {
          if(patch->containsNode(ni[k])) {
            S[k] *= pErosion[idx];
            gmass[ni[k]]          += pmass[idx]                     * S[k];
            gvelocity[ni[k]]      += pmom                           * S[k];
            gvolume[ni[k]]        += pvolume[idx]                   * S[k];
            gexternalforce[ni[k]] += pexternalforce[idx]            * S[k];
            gTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
            //  gexternalheatrate[ni[k]] += pexternalheatrate[idx]      * S[k];
            gnumnearparticles[ni[k]] += 1.0;
            gSp_vol[ni[k]]        += pSp_vol[idx]  * pmass[idx]     *S[k];
          }
        }
      } // End of particle loop

      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter; 
        totalmass       += gmass[c];
        gmassglobal[c]  += gmass[c];
        gvelglobal[c]   += gvelocity[c];
        gvelocity[c]    /= gmass[c];
        gtempglobal[c]  += gTemperature[c];
        gTemperature[c] /= gmass[c];
        gTemperatureNoBC[c] = gTemperature[c];
        gSp_vol[c]      /= gmass[c];
      }

      // Apply grid boundary conditions to the velocity before storing the data

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity",   gvelocity,    n8or27);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gvelocity,    n8or27);
      bc.setBoundaryCondition(patch,dwi,"Temperature",gTemperature, n8or27);

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);

    }  // End loop over materials

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;
      gtempglobal[c] /= gmassglobal[c];
      gvelglobal[c] /= gmassglobal[c];
    }
  }  // End loop over patches
}

void SerialMPM::computeStressTensor(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{

  cout_doing <<"Doing computeStressTensor:MPM: \n" ;
  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    cout_dbg << " Patch = " << (patches->get(0))->getID();
    cout_dbg << " Mat = " << m;
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    cout_dbg << " MPM_Mat = " << mpm_matl;
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cout_dbg << " CM = " << cm;
    cm->setWorld(d_myworld);
    cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);
    cout_dbg << " Exit\n" ;
  }
}

void SerialMPM::updateErosionParameter(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* ,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    cout_doing << getpid() << "Doing updateErosionParameter on patch " 
               << patch->getID() << "\t MPM"<< endl;

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){

      cout_dbg << "updateErosionParameter:: material # = " << m << endl;

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      cout_dbg << "updateErosionParameter:: mpm_matl* = " << mpm_matl
                 << " dwi = " << dwi << " pset* = " << pset << endl;

      // Get the erosion data
      constParticleVariable<double> pErosion;
      ParticleVariable<double> pErosion_new;
      old_dw->get(pErosion, lb->pErosionLabel, pset);
      new_dw->allocateAndPut(pErosion_new, lb->pErosionLabel_preReloc, pset);
      cout_dbg << "updateErosionParameter:: Got Erosion data" << endl;

      // Get the localization info
      ParticleVariable<int> isLocalized;
      new_dw->allocateTemporary(isLocalized, pset);
      ParticleSubset::iterator iter = pset->begin(); 
      for (; iter != pset->end(); iter++) isLocalized[*iter] = 0;
      mpm_matl->getConstitutiveModel()->getDamageParameter(patch, isLocalized,
                                                           dwi, old_dw,new_dw);
      cout_dbg << "updateErosionParameter:: Got Damage Parameter" << endl;

      iter = pset->begin(); 
      for (; iter != pset->end(); iter++) {
        pErosion_new[*iter] = pErosion[*iter];
        if (isLocalized[*iter]) {
          if (flags->d_erosionAlgorithm == "RemoveMass") {
            pErosion_new[*iter] = 0.1*pErosion[*iter];
          } 
        } 
      }

      cout_dbg << "updateErosionParameter:: Updated Erosion " << endl;

    }
    cout_dbg <<"Done updateErosionParamter on patch " 
               << patch->getID() << "\t MPM"<< endl;
  }
}

void SerialMPM::computeArtificialViscosity(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  double C0 = flags->d_artificialViscCoeff1;
  double C1 = flags->d_artificialViscCoeff2;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing computeArtificialViscosity on patch " << patch->getID()
               <<"\t\t MPM"<< endl;

    // The following scheme for removing ringing behind a shock comes from:
    // VonNeumann, J.; Richtmyer, R. D. (1950): A method for the numerical
    // calculation of hydrodynamic shocks. J. Appl. Phys., vol. 21, pp. 232.

    Ghost::GhostType  gac   = Ghost::AroundCells;

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constNCVariable<Vector> gvelocity;
      ParticleVariable<double> p_q;
      constParticleVariable<Vector> psize;
      constParticleVariable<Point> px;
      constParticleVariable<double> pmass,pvol_def;
      new_dw->get(gvelocity, lb->gVelocityLabel, dwi,patch, gac, NGN);
      old_dw->get(px,        lb->pXLabel,                      pset);
      old_dw->get(pmass,     lb->pMassLabel,                   pset);
      new_dw->get(pvol_def,  lb->pVolumeDeformedLabel,         pset);
      new_dw->allocateAndPut(p_q,    lb->p_qLabel,             pset);
      if(flags->d_8or27==27){
        old_dw->get(psize,   lb->pSizeLabel,                   pset);
      }

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
        IntVector ni[MAX_BASIS];
        Vector d_S[MAX_BASIS];

        if(flags->d_8or27==8){
          patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
        }
        else if(flags->d_8or27==27){
          patch->findCellAndShapeDerivatives27(px[idx], ni, d_S,psize[idx]);
        }

        velGrad.set(0.0);
        for(int k = 0; k < flags->d_8or27; k++) {
          const Vector& gvel = gvelocity[ni[k]];
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
               c_dil = sqrt(K*pvol_def[idx]/pmass[idx]);
               p_q[idx] = (C0*fabs(c_dil*DTrace*dx_ave) +
                           C1*(DTrace*DTrace*dx_ave*dx_ave))*
                           (pmass[idx]/pvol_def[idx]);
             }

      }
    }
  }

}

void SerialMPM::computeInternalForce(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  // node based forces
  Vector bndyForce[6];
  double bndyArea[6];
  for(int iface=0;iface<6;iface++) {
      bndyForce[iface]  = Vector(0.);
      bndyArea [iface]  = 0.;
  }
  double partvoldef = 0.;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing computeInternalForce on patch " << patch->getID()
               <<"\t\t\t MPM"<< endl;

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    Matrix3 Id;
    Id.Identity();

    int numMPMMatls = d_sharedState->getNumMPMMatls();

    NCVariable<Matrix3>       gstressglobal;
    constNCVariable<double>   gmassglobal;
    new_dw->get(gmassglobal,  lb->gMassLabel,
                d_sharedState->getAllInOneMatl()->get(0), patch, Ghost::None,0);
    new_dw->allocateAndPut(gstressglobal, lb->gStressForSavingLabel, 
                           d_sharedState->getAllInOneMatl()->get(0), patch);

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
      constParticleVariable<Vector>  psize;
      constParticleVariable<double>  pErosion;
      NCVariable<Vector>             internalforce;
      constNCVariable<double>        gvolume;
      NCVariable<Matrix3>            gstress;
      constNCVariable<double>        gmass;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel,                      pset);
      old_dw->get(pmass,   lb->pMassLabel,                   pset);
      new_dw->get(pvol,    lb->pVolumeDeformedLabel,         pset);
      new_dw->get(pstress, lb->pStressLabel_preReloc,        pset);
      if(flags->d_8or27==27){
        old_dw->get(psize, lb->pSizeLabel,                   pset);
      }
      new_dw->get(gmass,   lb->gMassLabel, dwi, patch, Ghost::None, 0);
      new_dw->get(gvolume, lb->gVolumeLabel, dwi, patch, Ghost::None, 0);
      new_dw->get(pErosion,lb->pErosionLabel_preReloc,       pset);

      new_dw->allocateAndPut(gstress,      lb->gStressForSavingLabel,dwi,patch);
      new_dw->allocateAndPut(internalforce,lb->gInternalForceLabel,  dwi,patch);

      if(d_with_ice){
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
        new_dw->get(p_q,lb->p_qLabel, pset);
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
      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];
      Vector d_S[MAX_BASIS];
      Matrix3 stressmass;
      Matrix3 stresspress;
      int n8or27 = flags->d_8or27;

      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        if(n8or27==8){
          patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S,  d_S);
        }
        else if(n8or27==27){
          patch->findCellAndWeightsAndShapeDerivatives27(px[idx], ni, S,d_S,
                                                         psize[idx]);
        }

        stressmass  = pstress[idx]*pmass[idx];
        //stresspress = pstress[idx] + Id*p_pressure[idx];
        stresspress = pstress[idx] + Id*p_pressure[idx] - Id*p_q[idx];
        partvoldef += pvol[idx];

        for (int k = 0; k < n8or27; k++){
          if(patch->containsNode(ni[k])){
            Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                       d_S[k].z()*oodx[2]);
            div *= pErosion[idx];
            internalforce[ni[k]] -= (div * stresspress)  * pvol[idx];
            gstress[ni[k]]       += stressmass * S[k];
          }
        }
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        gstressglobal[c] += gstress[c];
        gstress[c] /= gmass[c];
      }

      // save boundary forces before apply symmetry boundary condition.
      for(list<Patch::FaceType>::const_iterator fit(d_bndy_traction_faces.begin()); 
          fit!=d_bndy_traction_faces.end();fit++) {       
        Patch::FaceType face = *fit;
        
        // Check if the face is on an external boundary
        if(patch->getBCType(face)==Patch::Neighbor)
           continue;
      
        // We are on the boundary, i.e. not on an interior patch
        // boundary, and also on the correct side, 
        // so do the traction accumulation . . .
        // loop nodes to find forces
        IntVector projlow, projhigh;
        patch->getFaceNodes(face, 0, projlow, projhigh);
        
        for (int i = projlow.x(); i<projhigh.x(); i++) {
          for (int j = projlow.y(); j<projhigh.y(); j++) {
            for (int k = projlow.z(); k<projhigh.z(); k++) {
              IntVector ijk(i,j,k);	
              // flip sign so that pushing on boundary gives positive force
              bndyForce[face] -= internalforce[ijk];
              
              double celldepth  = dx[face/2];
              bndyArea [face] += gvolume[ijk]/celldepth;
            }
          }
        }
        
      } // faces
      
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",internalforce,n8or27);

#ifdef KUMAR
      internalforce.initialize(Vector(0,0,0));
#endif
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      gstressglobal[c] /= gmassglobal[c];
    }
  }
  new_dw->put(sum_vartype(partvoldef), lb->TotalVolumeDeformedLabel);
   
  
  // be careful only to put the fields that we have built
  // that way if the user asks to output a field that has not been built
  // it will fail early rather than just giving zeros.
  for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
      ftit!=d_bndy_traction_faces.end();ftit++) {
    new_dw->put(sumvec_vartype(bndyForce[*ftit]),lb->BndyForceLabel[*ftit]);
    new_dw->put(sum_vartype(bndyArea[*ftit]),lb->BndyContactAreaLabel[*ftit]);
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

    cout_doing <<"Doing computeInternalHeatRate on patch " << patch->getID()
               <<"\t\t MPM"<< endl;

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    Ghost::GhostType  gac   = Ghost::AroundCells;
    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      double kappa = mpm_matl->getThermalConductivity();
      double Cv = mpm_matl->getSpecificHeat();
      
      constParticleVariable<Point>  px;
      constParticleVariable<double> pvol;
      constParticleVariable<double> pIntHeatRate;
      constParticleVariable<double> pMass;
      constParticleVariable<Vector> psize;
      constParticleVariable<double> pErosion;
      ParticleVariable<Vector>      pTemperatureGradient;
      constNCVariable<double>       gTemperature;
      constNCVariable<double>       gMass;
      NCVariable<double>            internalHeatRate;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       lb->pXLabel);

      old_dw->get(px,           lb->pXLabel,                         pset);
      new_dw->get(pvol,         lb->pVolumeDeformedLabel,            pset);
      new_dw->get(pIntHeatRate, lb->pInternalHeatRateLabel_preReloc, pset);

      old_dw->get(pMass,        lb->pMassLabel,                      pset);
      if(flags->d_8or27==27){
        old_dw->get(psize,      lb->pSizeLabel,           pset);
      }
      new_dw->get(pErosion,     lb->pErosionLabel_preReloc, pset);

      new_dw->get(gTemperature, lb->gTemperatureLabel,   dwi, patch, gac,2*NGN);
      new_dw->get(gMass,        lb->gMassLabel,          dwi, patch, gnone, 0);
      new_dw->allocateAndPut(internalHeatRate, lb->gInternalHeatRateLabel,
                             dwi, patch);
      new_dw->allocateTemporary(pTemperatureGradient, pset);
  
      internalHeatRate.initialize(0.);

      // Create a temporary variable to store the mass weighted grid node
      // internal heat rate that has been projected from the particles
      // to the grid
      NCVariable<double> gPIntHeatRate;
      new_dw->allocateTemporary(gPIntHeatRate, patch, gnone, 0);
      gPIntHeatRate.initialize(0.);

      // First compute the temperature gradient at each particle
      double S[MAX_BASIS];
      IntVector ni[MAX_BASIS];
      Vector d_S[MAX_BASIS];

      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        if(flags->d_8or27==8){
          patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S);
        }
        else if(flags->d_8or27==27){
          patch->findCellAndWeightsAndShapeDerivatives27(px[idx], ni, S, d_S,
                                                         psize[idx]);
        }

        // Weight the particle internal heat rate with the mass
        double pIntHeatRate_massWt = pIntHeatRate[idx]*pMass[idx];

        pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
        for (int k = 0; k < flags->d_8or27; k++){
          S[k] *= pErosion[idx];
          for (int j = 0; j<3; j++) {
            pTemperatureGradient[idx][j] += 
              gTemperature[ni[k]] * d_S[k][j] * oodx[j];
          }
          // Project the mass weighted particle internal heat rate to
          // the grid
          if(patch->containsNode(ni[k])){
             gPIntHeatRate[ni[k]] +=  (pIntHeatRate_massWt*S[k]);
          }
        }
      }


      // Get the internal heat rate due to particle deformation at the
      // grid nodes by dividing gPIntHeatRate by the grid mass
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        gPIntHeatRate[c] /= gMass[c];
        internalHeatRate[c] = gPIntHeatRate[c];
      }

      // Compute T,ii
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        if(flags->d_8or27==8){
          patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
        }
        else if(flags->d_8or27==27){
          patch->findCellAndShapeDerivatives27(px[idx], ni, d_S, psize[idx]);
        }

        // Calculate k/(rho*Cv)
        double alpha = (kappa*pvol[idx])/(pMass[idx]*Cv); 
        Vector T_i = pTemperatureGradient[idx];
        double T_ii = 0.0;
        IntVector node(0,0,0);
        for (int k = 0; k < flags->d_8or27; k++){
          node = ni[k];
          if(patch->containsNode(node)){
            Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                       d_S[k].z()*oodx[2]);
            // Question: Why decreasing internal heat rate ?
            T_ii = Dot(div, T_i)*alpha*flags->d_adiabaticHeating;
            internalHeatRate[node] -= T_ii;
          }
        }
      }
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

    cout_doing <<"Doing solveEquationsMotion on patch " << patch->getID()
               <<"\t\t\t MPM"<< endl;

    Vector gravity = d_sharedState->getGravity();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get required variables for this patch
      constNCVariable<Vector> internalforce;
      constNCVariable<Vector> externalforce;
      constNCVariable<Vector> gradPAccNC;  // for MPMICE
      constNCVariable<double> mass;
 
      new_dw->get(internalforce,lb->gInternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(externalforce,lb->gExternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(mass,         lb->gMassLabel,          dwi, patch, gnone, 0);
      if(d_with_ice){
        new_dw->get(gradPAccNC, lb->gradPAccNCLabel,     dwi, patch, gnone, 0);
      }
      else{
        NCVariable<Vector> gradPAccNC_create;
        new_dw->allocateTemporary(gradPAccNC_create,  patch);
        gradPAccNC_create.initialize(Vector(0.,0.,0.));
        gradPAccNC = gradPAccNC_create; // reference created data
      }

      //Uncomment to use damping
      //constNCVariable<Vector> velocity;
      //new_dw->get(velocity,  lb->gVelocityLabel,      dwi, patch, gnone, 0);
      //cout << "Damping is on" << endl;

      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocateAndPut(acceleration, lb->gAccelerationLabel, dwi, patch);
      acceleration.initialize(Vector(0.,0.,0.));

      for(NodeIterator iter = patch->getNodeIterator(flags->d_8or27);
                       !iter.done();iter++){
        IntVector c = *iter;
        Vector acc(0.0,0.0,0.0);
	//if (mass[c] > 1.0e-199)
	  acc = (internalforce[c] + externalforce[c])/mass[c] ;
        acceleration[c] = acc +  gravity + gradPAccNC[c];

//                 acceleration[c] =
//                    (internalforce[c] + externalforce[c]
//                    -5000.*velocity[c]*mass[c])/mass[c]
//                    + gravity + gradPAccNC[c];
      }
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

    cout_doing <<"Doing solveHeatEquations on patch " << patch->getID()
               <<"\t\t\t MPM"<< endl;

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      double Cv = mpm_matl->getSpecificHeat();
     
      // Get required variables for this patch
      constNCVariable<double> mass,externalHeatRate,gvolume;
      constNCVariable<double> thermalContactHeatExchangeRate;
      NCVariable<double> internalHeatRate;
            
      new_dw->get(mass,    lb->gMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(gvolume, lb->gVolumeLabel,    dwi, patch, Ghost::None, 0);
      new_dw->get(externalHeatRate,           lb->gExternalHeatRateLabel,
                  dwi, patch, Ghost::None, 0);
      new_dw->getModifiable(internalHeatRate, lb->gInternalHeatRateLabel,
                            dwi, patch);

      new_dw->get(thermalContactHeatExchangeRate,
                  lb->gThermalContactHeatExchangeRateLabel,
                  dwi, patch, Ghost::None, 0);

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Temperature",internalHeatRate,
                              gvolume,flags->d_8or27);

      // Create variables for the results
      NCVariable<double> tempRate;
      new_dw->allocateAndPut(tempRate, lb->gTemperatureRateLabel, dwi, patch);
      tempRate.initialize(0.0);
      int n8or27=flags->d_8or27;

      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter;
        tempRate[c] = internalHeatRate[c]*((mass[c]-1.e-200)/mass[c]) +  
	  (externalHeatRate[c])/(mass[c]*Cv)+thermalContactHeatExchangeRate[c];
      }
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

    cout_doing <<"Doing integrateAcceleration on patch " << patch->getID()
               <<"\t\t\t MPM"<< endl;

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      constNCVariable<Vector>  acceleration, velocity;
      delt_vartype delT;

      new_dw->get(acceleration,lb->gAccelerationLabel,dwi, patch,Ghost::None,0);
      new_dw->get(velocity,    lb->gVelocityLabel,    dwi, patch,Ghost::None,0);

      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

      // Create variables for the results
      NCVariable<Vector> velocity_star;
      new_dw->allocateAndPut(velocity_star, lb->gVelocityStarLabel, dwi, patch);
      velocity_star.initialize(Vector(0.0));

      for(NodeIterator iter = patch->getNodeIterator(flags->d_8or27);
                       !iter.done();iter++){
        IntVector c = *iter;
        velocity_star[c] = velocity[c] + acceleration[c] * delT;
      }
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

    cout_doing <<"Doing integrateTemperatureRate on patch " << patch->getID()
               << "\t\t MPM"<< endl;

    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<double> temp_old,temp_oldNoBC;
      NCVariable<double> temp_rate,tempStar;
      delt_vartype delT;
 
      new_dw->get(temp_old,    lb->gTemperatureLabel,     dwi,patch,gnone,0);
      new_dw->get(temp_oldNoBC,lb->gTemperatureNoBCLabel, dwi,patch,gnone,0);
      new_dw->getModifiable(temp_rate, lb->gTemperatureRateLabel,dwi,patch);

      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

      new_dw->allocateAndPut(tempStar, lb->gTemperatureStarLabel, dwi,patch);
      tempStar.initialize(0.0);
      int n8or27=flags->d_8or27;

      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter;
        tempStar[c] = temp_old[c] + temp_rate[c] * delT;
      }

      // Apply grid boundary conditions to the temperature 

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Temperature",tempStar,n8or27);

      // Now recompute temp_rate as the difference between the temperature
      // interpolated to the grid (no bcs applied) and the new tempStar
      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter;
        temp_rate[c] = (tempStar[c] - temp_oldNoBC[c]) / delT;
      }
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

    cout_doing <<"Doing setGridBoundaryConditions on patch " << patch->getID()
               <<"\t\t MPM"<< endl;

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    
    delt_vartype delT;            
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
                      
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;
      int n8or27=flags->d_8or27;

      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,  dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,  dwi,patch);
      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity",     gvelocity_star, n8or27);
      bc.setBoundaryCondition(patch,dwi,"Acceleration", gacceleration,  n8or27);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",    gvelocity_star, n8or27);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",    gacceleration,  n8or27);

    } // matl loop
  }  // patch loop
}

void SerialMPM::applyExternalLoads(const ProcessorGroup* ,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  // Get the current time
  double time = d_sharedState->getElapsedTime();
  cout_doing << "Current Time (applyExternalLoads) = " << time << endl;

  // Calculate the force vector at each particle for each pressure bc
  std::vector<double> forcePerPart;
  std::vector<PressureBC*> pbcP;
  std::vector<NormalForceBC*> nfbcP;
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
      if (bcs_type == "NormalForce") {
        NormalForceBC* nfbc =
         dynamic_cast<NormalForceBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);        nfbcP.push_back(nfbc);
                                                                                
        // Calculate the force per particle at current time
        forcePerPart.push_back(nfbc->getLoad(time));
      }
    }
  }

  // Loop thru patches to update external force vector
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

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
       bool do_NormalForceBCs=false;
       for (int ii = 0; 
             ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
        string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
         if (bcs_type == "Pressure") {
           do_PressureBCs=true;
         }
         if (bcs_type == "NormalForce") {
           do_NormalForceBCs=true;
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
        constParticleVariable<Vector> pExternalForce;
        ParticleVariable<Vector> pExternalForce_new;
        old_dw->get(pExternalForce, lb->pExternalForceLabel, pset);
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
            pExternalForce_new[idx] = pbc->getForceVector(px[idx], force);
          }
        }

        // Recycle the loadCurveIDs
        ParticleVariable<int> pLoadCurveID_new;
        new_dw->allocateAndPut(pLoadCurveID_new, 
                               lb->pLoadCurveIDLabel_preReloc, pset);
        pLoadCurveID_new.copyData(pLoadCurveID);
       }
       else if(do_NormalForceBCs){  // Scale the normal vector by a magnitude
        // Get the external force data and allocate new space for
        // external force
        constParticleVariable<Vector> pExternalForce;
        ParticleVariable<Vector> pExternalForce_new;
        old_dw->get(pExternalForce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pExternalForce_new,
                               lb->pExtForceLabel_preReloc,  pset);
                                                                                
        double mag = forcePerPart[0];
        // Iterate over the particles
        ParticleSubset::iterator iter = pset->begin();
        for(;iter != pset->end(); iter++){
          particleIndex idx = *iter;
          // For particles with an existing external force, apply the
          // new magnitude to the same direction.
          if(pExternalForce[idx].length() > 1.e-7){
            pExternalForce_new[idx] = mag*
                       (pExternalForce[idx]/pExternalForce[idx].length());
          } else{
            pExternalForce_new[idx] = Vector(0.,0.,0.);
          }
        }
        // Recycle the loadCurveIDs, not needed for this BC type yet
        ParticleVariable<int> pLoadCurveID_new;
        constParticleVariable<int> pLoadCurveID;
        old_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
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

void SerialMPM::calculateDampingRate(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  if (flags->d_artificialDampCoeff > 0.0) {
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      cout_doing <<"Doing calculateDampingRate on patch " 
                 << patch->getID() << "\t MPM"<< endl;

      double alphaDot = 0.0;
      int numMPMMatls=d_sharedState->getNumMPMMatls();
      for(int m = 0; m < numMPMMatls; m++){
        MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
        int dwi = mpm_matl->getDWIndex();

        // Get the arrays of particle values to be changed
        constParticleVariable<Point> px;
        constParticleVariable<Vector> psize;

        // Get the arrays of grid data on which the new part. values depend
        constNCVariable<Vector> gvelocity_star;

        ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
        old_dw->get(px, lb->pXLabel, pset);
        if(flags->d_8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);
        Ghost::GhostType  gac = Ghost::AroundCells;
        new_dw->get(gvelocity_star,   lb->gVelocityStarLabel,dwi,patch,gac,NGP);

        IntVector ni[MAX_BASIS];
        double S[MAX_BASIS];
        Vector d_S[MAX_BASIS];

        // Calculate artificial dampening rate based on the interpolated particle
        // velocities (ref. Ayton et al., 2002, Biophysical Journal, 1026-1038)
        // d(alpha)/dt = 1/Q Sum(vp*^2)
        ParticleSubset::iterator iter = pset->begin();
        for(;iter != pset->end(); iter++){
          particleIndex idx = *iter;
          if (flags->d_8or27 == 27) 
            patch->findCellAndWeightsAndShapeDerivatives27(px[idx],ni,S,d_S,
                                                           psize[idx]);
          else
            patch->findCellAndWeightsAndShapeDerivatives(  px[idx],ni,S,d_S);
          Vector vel(0.0,0.0,0.0);
          for (int k = 0; k < flags->d_8or27; k++) 
            vel += gvelocity_star[ni[k]]*S[k];
          alphaDot += Dot(vel,vel);
        }
        alphaDot /= flags->d_artificialDampCoeff;
      } 
      new_dw->put(sum_vartype(alphaDot), lb->pDampingRateLabel);
    }
  }
}

void SerialMPM::addNewParticles(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing addNewParticles on patch " 
               << patch->getID() << "\t MPM"<< endl;

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    // Find the mpm material that the void particles are going to change
    // into.
    MPMMaterial* null_matl = 0;
    int null_dwi = -1;
    for (int void_matl = 0; void_matl < numMPMMatls; void_matl++) {
      null_dwi = d_sharedState->getMPMMaterial(void_matl)->nullGeomObject();
      cout_dbg << "Null DWI = " << null_dwi << endl;
      if (null_dwi != -1) {
        null_matl = d_sharedState->getMPMMaterial(void_matl);
        null_dwi = void_matl;
        break;
      }
    }
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      if (dwi == null_dwi)
        continue;


      ParticleVariable<int> damage;
#if 0
      cout << "Current MPM Old_DW Labels in Add New Particles" << endl;
      vector<const VarLabel*> mpm_label = 
        mpm_matl->getParticleCreator()->returnParticleState();
      printParticleLabels(mpm_label,old_dw,dwi,patch);
#endif
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      new_dw->allocateTemporary(damage,pset);
      for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
           iter++) 
        damage[*iter] = 0;

      ParticleSubset* delset = scinew ParticleSubset(pset->getParticleSet(),
                                                     false,dwi,patch, 0);
      
      mpm_matl->getConstitutiveModel()->getDamageParameter(patch,damage,dwi,
                                                           old_dw,new_dw);

      for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
           iter++) {
        if (damage[*iter]) {
          cout_dbg << "damage[" << *iter << "]=" << damage[*iter] << endl;
          delset->addParticle(*iter);
        }
      }
      
      // Find the mpm material that corresponds to the void particles.
      // Will probably be the same type as the deleted ones, but have
      // different parameters.
      
      
      int numparticles = delset->numParticles();
      cout_dbg << "Num Failed Particles = " << numparticles << endl;
      if (numparticles != 0) {
        cout_dbg << "Deleted " << numparticles << " particles" << endl;
        ParticleCreator* particle_creator = null_matl->getParticleCreator();
        ParticleSet* set_add = scinew ParticleSet(numparticles);
        ParticleSubset* addset = scinew ParticleSubset(set_add,true,null_dwi,
                                                       patch,numparticles);

        //cout_dbg << "Address of delset = " << delset << endl;
        //cout_dbg << "Address of pset = " << pset << endl;
        //cout_dbg << "Address of set_add = " << set_add << endl;
        //cout_dbg << "Address of addset = " << addset << endl;
        
        map<const VarLabel*, ParticleVariableBase*>* newState
          = scinew map<const VarLabel*, ParticleVariableBase*>;

        //cout_dbg << "Address of newState = " << newState << endl;

        //cout_dbg << "Null Material" << endl;
        //vector<const VarLabel* > particle_labels = 
        //  particle_creator->returnParticleState();

        //printParticleLabels(particle_labels, old_dw, null_dwi,patch);

        //cout_dbg << "MPM Material" << endl;
        //vector<const VarLabel* > mpm_particle_labels = 
        //  mpm_matl->getParticleCreator()->returnParticleState();
        //printParticleLabels(mpm_particle_labels, old_dw, dwi,patch);

        particle_creator->allocateVariablesAdd(lb,new_dw,addset,newState,
                                               delset,old_dw);
        
        // Need to do the constitutive models particle variables;
        
        null_matl->getConstitutiveModel()->allocateCMDataAdd(new_dw,addset,
                                                             newState,delset,
                                                             old_dw);

        // Need to carry forward the cellNAPID for each time step;
        // Move the particle variable declarations in ParticleCreator.h to one
        // of the functions to save on memory;
        
        //cout_dbg << "addset num particles = " << addset->numParticles()
        //     << " for material " << addset->getMatlIndex() << endl;
        new_dw->addParticles(patch,null_dwi,newState);
        //cout_dbg << "Calling deleteParticles for material: " << dwi << endl;
        new_dw->deleteParticles(delset);
        
      } else
        delete delset;
    }
  }
  
}

/* Convert the localized particles of material "i" into particles of 
   material "i+1" */
void SerialMPM::convertLocalizedParticles(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  // This function is called only when the "createNewParticles" flag is on.
  // When this flag is on, every second material is a copy of the previous
  // material and is used the material into which particles of the previous
  // material are converted.
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing convertLocalizedParticles on patch " 
               << patch->getID() << "\t MPM"<< endl;

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m+=2){

      cout_dbg << "ConvertLocalizeParticles:: material # = " << m << endl;

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      cout_dbg << "ConvertLocalizeParticles:: mpm_matl* = " << mpm_matl
                 << " dwi = " << dwi << " pset* = " << pset << endl;

      ParticleVariable<int> isLocalized;

      //old_dw->allocateTemporary(isLocalized, pset);
      new_dw->allocateTemporary(isLocalized, pset);

      ParticleSubset::iterator iter = pset->begin(); 
      for (; iter != pset->end(); iter++) isLocalized[*iter] = 0;

      ParticleSubset* delset = scinew ParticleSubset(pset->getParticleSet(),
                                                     false, dwi, patch, 0);
      
      mpm_matl->getConstitutiveModel()->getDamageParameter(patch, isLocalized,
                                                           dwi, old_dw,new_dw);

      cout_dbg << "ConvertLocalizeParticles:: Got Damage Parameter" << endl;

      iter = pset->begin(); 
      for (; iter != pset->end(); iter++) {
        if (isLocalized[*iter]) {
          //cout << "damage[" << *iter << "]=" << isLocalized[*iter] << endl;
          delset->addParticle(*iter);
        }
      }

      cout_dbg << "ConvertLocalizeParticles:: Created Delset ";

      int numparticles = delset->numParticles();

      cout_dbg << "numparticles = " << numparticles << endl;

      if (numparticles != 0) {

        cout_dbg << "Converting " 
                   << numparticles << " particles of material " 
                   <<  m  << " into particles of material " << (m+1) 
                   << " in patch " << p << endl;

        MPMMaterial* conv_matl = d_sharedState->getMPMMaterial(m+1);
        int conv_dwi = conv_matl->getDWIndex();
      
        ParticleCreator* particle_creator = conv_matl->getParticleCreator();
        ParticleSet* set_add = scinew ParticleSet(numparticles);
        ParticleSubset* addset = scinew ParticleSubset(set_add, true,
                                                       conv_dwi, patch,
                                                       numparticles);
        
        map<const VarLabel*, ParticleVariableBase*>* newState
          = scinew map<const VarLabel*, ParticleVariableBase*>;

        //cout << "New Material" << endl;
        //vector<const VarLabel* > particle_labels = 
        //  particle_creator->returnParticleState();
        //printParticleLabels(particle_labels, old_dw, conv_dwi,patch);

        //cout << "MPM Material" << endl;
        //vector<const VarLabel* > mpm_particle_labels = 
        //  mpm_matl->getParticleCreator()->returnParticleState();
        //printParticleLabels(mpm_particle_labels, old_dw, dwi,patch);

        particle_creator->allocateVariablesAdd(lb, new_dw, addset, newState,
                                               delset, old_dw);
        
        conv_matl->getConstitutiveModel()->allocateCMDataAdd(new_dw, addset,
                                                             newState, delset,
                                                             old_dw);

        cout_dbg << "addset num particles = " << addset->numParticles()
                   << " for material " << addset->getMatlIndex() << endl;
        new_dw->addParticles(patch, conv_dwi, newState);
        new_dw->deleteParticles(delset);
        
        //delete addset;
      } 
      else delete delset;
    }
    cout_dbg <<"Done convertLocalizedParticles on patch " 
               << patch->getID() << "\t MPM"<< endl;
  }
  cout_dbg << "Completed convertLocalizedParticles " << endl;
  
}

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* ,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing interpolateToParticlesAndUpdate on patch " 
               << patch->getID() << "\t MPM"<< endl;

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
 
    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    Vector CMX(0.0,0.0,0.0);
    Vector CMV(0.0,0.0,0.0);
    double ke=0;
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
    bool combustion_problem=false;

    // Artificial Damping 
    double alphaDot = 0.0;
    double alpha = 0.0;
    if (flags->d_artificialDampCoeff > 0.0) {
      max_vartype dampingCoeff; 
      sum_vartype dampingRate;
      old_dw->get(dampingCoeff, lb->pDampingCoeffLabel);
      new_dw->get(dampingRate, lb->pDampingRateLabel);
      alpha = (double) dampingCoeff;
      alphaDot = (double) dampingRate;
      alpha += alphaDot*delT; // Calculate damping coefficient from damping rate
      new_dw->put(max_vartype(alpha), lb->pDampingCoeffLabel);
    }

    Material* reactant;
    int RMI = -99;
    reactant = d_sharedState->getMaterialByName("reactant");
    if(reactant != 0){
      RMI = reactant->getDWIndex();
      combustion_problem=true;
    }
    double move_particles=1.;
    if(!d_doGridReset){
      move_particles=0.;
    }
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew,pxx;
      constParticleVariable<Vector> pvelocity, psize;
      ParticleVariable<Vector> pvelocitynew, psizeNew;
      constParticleVariable<double> pmass, pvolume, pTemperature, pSp_vol;
      ParticleVariable<double> pmassNew,pvolumeNew,pTempNew, pSp_volNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;
      constParticleVariable<double> pErosion;

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<double> gTemperatureRate, gTemperature, gTemperatureNoBC;
      constNCVariable<double> dTdt, massBurnFrac, frictionTempRate;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pids,         lb->pParticleIDLabel,                pset);
      old_dw->get(pSp_vol,      lb->pSp_volLabel,                    pset);
      new_dw->get(pvolume,      lb->pVolumeDeformedLabel,            pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
      new_dw->get(pErosion,     lb->pErosionLabel_preReloc,          pset);

      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pxx,          lb->pXXLabel,                  pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pvolumeNew,   lb->pVolumeLabel_preReloc,     pset);
      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);
      new_dw->allocateAndPut(pSp_volNew,   lb->pSp_volLabel_preReloc,     pset);
     
      ParticleSubset* delset = scinew ParticleSubset(pset->getParticleSet(),
                                                     false,dwi,patch, 0);

      pids_new.copyData(pids);
      old_dw->get(psize,               lb->pSizeLabel,                 pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,        pset);
      psizeNew.copyData(psize);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      new_dw->get(gacceleration,   lb->gAccelerationLabel,   dwi,patch,gac,NGP);
      new_dw->get(gTemperatureRate,lb->gTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(gTemperature,    lb->gTemperatureLabel,    dwi,patch,gac,NGP);
      new_dw->get(gTemperatureNoBC,lb->gTemperatureNoBCLabel,dwi,patch,gac,NGP);
      new_dw->get(frictionTempRate,lb->frictionalWorkLabel,  dwi,patch,gac,NGP);
      if(d_with_ice){
        new_dw->get(dTdt,          lb->dTdt_NCLabel,         dwi,patch,gac,NGP);
        new_dw->get(massBurnFrac,  lb->massBurnFractionLabel,dwi,patch,gac,NGP);
      }
      else{
        NCVariable<double> dTdt_create,massBurnFrac_create;
        new_dw->allocateTemporary(dTdt_create,                   patch,gac,NGP);
        new_dw->allocateTemporary(massBurnFrac_create,           patch,gac,NGP);
        dTdt_create.initialize(0.);
        massBurnFrac_create.initialize(0.);
        dTdt = dTdt_create;                         // reference created data
        massBurnFrac = massBurnFrac_create;         // reference created data
      }

      double Cp=mpm_matl->getSpecificHeat();
      double rho_init=mpm_matl->getInitialDensity();

      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];
      Vector d_S[MAX_BASIS];

      double rho_frac_min = 0.;
      if(m == RMI){
        rho_frac_min = .1;
      }
      const Level* lvl = patch->getLevel();

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        if(flags->d_8or27==8){
          patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S);
        }
        else if(flags->d_8or27==27){
          patch->findCellAndWeightsAndShapeDerivatives27(px[idx], ni, S, d_S,
                                                         psize[idx]);
        }

        Vector vel(0.0,0.0,0.0);
        Vector acc(0.0,0.0,0.0);
        double tempRate = 0.0;
        double burnFraction = 0.0;

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          S[k] *= pErosion[idx];
          vel      += gvelocity_star[node]  * S[k];
          acc      += gacceleration[node]   * S[k];
          tempRate += (gTemperatureRate[node] + dTdt[node] +
                       frictionTempRate[node])   * S[k];
          burnFraction += massBurnFrac[node]     * S[k];
        }

        // Update the particle's position and velocity
        pxnew[idx]           = px[idx]    + vel*delT*move_particles;
        pdispnew[idx]        = pdisp[idx] + vel*delT;
        pvelocitynew[idx]    = pvelocity[idx]    + (acc - alpha*vel)*delT;
        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx]    + pdispnew[idx];
        pTempNew[idx]        = pTemperature[idx] + tempRate*delT ;
        pSp_volNew[idx]      = pSp_vol[idx];

        cout_heat << "MPM::Particle = " << idx 
                    << " T_old = " << pTemperature[idx]
                    << " Tdot = " << tempRate
                    << " dT = " << (tempRate*delT)
                    << " T_new = " << pTempNew[idx] << endl;

        double rho;
        if(pvolume[idx] > 0.){
          rho = max(pmass[idx]/pvolume[idx],rho_frac_min*rho_init);
        }
        else{
          rho = rho_init;
        }
        pmassNew[idx]        = Max(pmass[idx]*(1.    - burnFraction),0.);
        pvolumeNew[idx]      = pmassNew[idx]/rho;

        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
        ke += .5*pmass[idx]*pvelocitynew[idx].length2();
        CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
        CMV += pvelocitynew[idx]*pmass[idx];
      }

      // Delete particles that have left the domain
      // This is only needed if extra cells are being used.
      // Also delete particles whose mass is too small (due to combustion)
      // For particles whose new velocity exceeds a maximum set in the input
      // file, set their velocity back to the velocity that it came into
      // this step with
      for(ParticleSubset::iterator iter  = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        bool pointInReal = lvl->containsPointInRealCells(pxnew[idx]);
        bool pointInAny = lvl->containsPoint(pxnew[idx]);
        if((!pointInReal && pointInAny) || (pmassNew[idx] <= d_min_part_mass)){
          delset->addParticle(idx);
        }
        if(pvelocitynew[idx].length() > d_max_vel){
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

    if(combustion_problem){  // Adjust the min. part. mass if dt gets small
      if(delT < 5.e-9){
        if(delT < 1.e-10){
          d_min_part_mass = min(d_min_part_mass*2.0,5.e-9);
          if(d_myworld->myrank() == 0){
            cout << "New d_min_part_mass = " << d_min_part_mass << endl;
          }
        }
        else{
          d_min_part_mass = min(d_min_part_mass*2.0,5.e-12);
          if(d_myworld->myrank() == 0){
            cout << "New d_min_part_mass = " << d_min_part_mass << endl;
          }
        }
      }
      else if(delT > 2.e-8){
        d_min_part_mass = max(d_min_part_mass/2.0,3.e-15);
      }
    }

    // DON'T MOVE THESE!!!
    new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
    new_dw->put(sum_vartype(thermal_energy), lb->ThermalEnergyLabel);
    new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
    new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);

    // cout << "Solid mass lost this timestep = " << massLost << endl;
    // cout << "Solid momentum after advection = " << CMV << endl;

    // cout << "THERMAL ENERGY " << thermal_energy << endl;
  }
  
}

void 
SerialMPM::setParticleDefaultWithTemp(constParticleVariable<double>& pvar,
                                      ParticleSubset* pset,
                                      DataWarehouse* new_dw,
                                      double val)
{
  ParticleVariable<double>  temp;
  new_dw->allocateTemporary(temp,  pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){
    temp[*iter]=val;
  }
  pvar = temp; 
}

void 
SerialMPM::setParticleDefault(ParticleVariable<double>& pvar,
                              const VarLabel* label, 
                              ParticleSubset* pset,
                              DataWarehouse* new_dw,
                              double val)
{
  new_dw->allocateAndPut(pvar, label, pset);
  ParticleSubset::iterator iter = pset->begin();
  for (; iter != pset->end(); iter++) {
    pvar[*iter] = val;
  }
}

void 
SerialMPM::setParticleDefault(ParticleVariable<Vector>& pvar,
                              const VarLabel* label, 
                              ParticleSubset* pset,
                              DataWarehouse* new_dw,
                              const Vector& val)
{
  new_dw->allocateAndPut(pvar, label, pset);
  ParticleSubset::iterator iter = pset->begin();
  for (; iter != pset->end(); iter++) {
    pvar[*iter] = val;
  }
}

void 
SerialMPM::setParticleDefault(ParticleVariable<Matrix3>& pvar,
                              const VarLabel* label, 
                              ParticleSubset* pset,
                              DataWarehouse* new_dw,
                              const Matrix3& val)
{
  new_dw->allocateAndPut(pvar, label, pset);
  ParticleSubset::iterator iter = pset->begin();
  for (; iter != pset->end(); iter++) {
    pvar[*iter] = val;
  }
}

void SerialMPM::setSharedState(SimulationStateP& ssp)
{
  d_sharedState = ssp;
}

void SerialMPM::printParticleLabels(vector<const VarLabel*> labels,
                                    DataWarehouse* dw, int dwi, 
                                    const Patch* patch)
{
  for (vector<const VarLabel*>::const_iterator it = labels.begin(); 
       it != labels.end(); it++) {
    if (dw->exists(*it,dwi,patch))
      cout << (*it)->getName() << " does exists" << endl;
    else
      cout << (*it)->getName() << " does NOT exists" << endl;
  }
}

void
SerialMPM::scheduleParticleVelocityField(SchedulerP&,  const PatchSet*,
                                         const MaterialSet*)
{
}

void
SerialMPM::scheduleAdjustCrackContactInterpolated(SchedulerP&, 
                                                  const PatchSet*,
                                                  const MaterialSet*)
{
}

void
SerialMPM::scheduleAdjustCrackContactIntegrated(SchedulerP&, 
                                                const PatchSet*,
                                                const MaterialSet*)
{
}

void
SerialMPM::scheduleCalculateFractureParameters(SchedulerP&, 
                                               const PatchSet*,
                                               const MaterialSet*)
{
}

void
SerialMPM::scheduleDoCrackPropagation(SchedulerP& /*sched*/, 
                                      const PatchSet* /*patches*/, 
                                      const MaterialSet* /*matls*/)
{
}

void
SerialMPM::scheduleMoveCracks(SchedulerP& /*sched*/,const PatchSet* /*patches*/,
                              const MaterialSet* /*matls*/)
{
}

void
SerialMPM::scheduleUpdateCrackFront(SchedulerP& /*sched*/,
                                    const PatchSet* /*patches*/,
                                    const MaterialSet* /*matls*/)
{
}

void
SerialMPM::initialErrorEstimate(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* /*matls*/,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    amr_doing << "Doing SerialMPM::initialErrorEstimate on patch "<< patch->getID()<< endl;
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
SerialMPM::errorEstimate(const ProcessorGroup* group,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw)
{
  // coarsen the errorflag.
  cout_doing << "Doing Serial::errorEstimate" << '\n';
  const Level* level = getLevel(patches);
  if (level->getIndex() == level->getGrid()->numLevels()-1) {
    // on finest level, we do the same thing as initialErrorEstimate, so call it
    initialErrorEstimate(group, patches, matls, old_dw, new_dw);
  }
  else {
    const Level* fineLevel = level->getFinerLevel().get_rep();
  
    for(int p=0;p<patches->size();p++){  
      const Patch* coarsePatch = patches->get(p);
      amr_doing << "Doing SerialMPM::errorEstimate on patch " << coarsePatch->getID() << endl;
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
        new_dw->get(fineErrorFlag, d_sharedState->get_refineFlag_label(), 0, finePatch,
                    Ghost::None, 0);
        
        IntVector fl(finePatch->getCellLowIndex());
        IntVector fh(finePatch->getCellHighIndex());
        IntVector l(fineLevel->mapCellToCoarser(fl));
        IntVector h(fineLevel->mapCellToCoarser(fh));
        l = Max(l, coarsePatch->getCellLowIndex());
        h = Min(h, coarsePatch->getCellHighIndex());
        
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
SerialMPM::refine(const ProcessorGroup*,
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
      cout_doing <<"Doing refine on patch "
                 << patch->getID() << " material # = " << dwi << endl;
      if (!new_dw->haveParticleSubset(dwi, patch)) {
        ParticleSubset* pset = new_dw->createParticleSubset(0, dwi, patch);

        // Create arrays for the particle data
        ParticleVariable<Point>  px;
        ParticleVariable<double> pmass, pvolume, pTemperature, pSp_vol;
        ParticleVariable<Vector> pvelocity, pexternalforce, psize, pdisp;
        ParticleVariable<double> pErosion;
        ParticleVariable<int>    pLoadCurve;
        ParticleVariable<long64> pID;
        ParticleVariable<Matrix3> pdeform, pstress;
        
        new_dw->allocateAndPut(px,             lb->pXLabel,             pset);
        new_dw->allocateAndPut(pmass,          lb->pMassLabel,          pset);
        new_dw->allocateAndPut(pvolume,        lb->pVolumeLabel,        pset);
        new_dw->allocateAndPut(pSp_vol,        lb->pSp_volLabel,        pset);
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
        new_dw->allocateAndPut(pErosion,       lb->pErosionLabel,       pset);

      }
    }
  }

} // end refine()

void SerialMPM::scheduleCheckNeedAddMPMMaterial(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  cout_doing << "SerialMPM::scheduleCheckNeedAddMaterial" << endl;
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("MPM::checkNeedAddMPMMaterial",
                        this, &SerialMPM::checkNeedAddMPMMaterial);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->scheduleCheckNeedAddMPMMaterial(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, matls);
}

void SerialMPM::checkNeedAddMPMMaterial(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  cout_doing <<"Doing checkNeedAddMPMMaterial:MPM: \n" ;
  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->checkNeedAddMPMMaterial(patches, mpm_matl, old_dw, new_dw);
  }
}

void SerialMPM::scheduleSetNeedAddMaterialFlag(SchedulerP& sched,
                                               const LevelP& level,
                                               const MaterialSet* all_matls)
{
  cout_doing << "SerialMPM::scheduleSetNeedAddMaterialFlag" << endl;
  Task* t= scinew Task("SerialMPM::setNeedAddMaterialFlag",
               this, &SerialMPM::setNeedAddMaterialFlag);
  t->requires(Task::NewDW, lb->NeedAddMPMMaterialLabel);
  sched->addTask(t, level->eachPatch(), all_matls);
}

void SerialMPM::setNeedAddMaterialFlag(const ProcessorGroup*,
                                       const PatchSubset* ,
                                       const MaterialSubset* ,
                                       DataWarehouse* ,
                                       DataWarehouse* new_dw)
{
    sum_vartype need_add_flag;
    new_dw->get(need_add_flag, lb->NeedAddMPMMaterialLabel);

    if(need_add_flag>0.1){
      d_sharedState->setNeedAddMaterial(true);
      flags->d_canAddMPMMaterial=false;
    }
    else{
      d_sharedState->setNeedAddMaterial(false);
    }
}

bool SerialMPM::needRecompile(double , double , const GridP& )
{
  if(d_recompile){
    d_recompile = false;
    return true;
  }
  else{
    return false;
  }
}
