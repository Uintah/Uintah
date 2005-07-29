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
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>
#include <Packages/Uintah/CCA/Components/MPM/HeatConduction/HeatConduction.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

#undef KUMAR
//#define KUMAR

using namespace Uintah;
using namespace SCIRun;

using namespace std;

static DebugStream cout_doing("MPM", false);
static DebugStream cout_dbg("SerialMPM", false);
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
  heatConductionModel = 0;
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
  delete heatConductionModel;
  MPMPhysicalBCFactory::clean();

}

void SerialMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
                             SimulationStateP& sharedState)
{
  d_sharedState = sharedState;

  ProblemSpecP mpm_soln_ps = prob_spec->findBlock("MPM");

  if(mpm_soln_ps) {

    // Read all MPM flags (look in MPMFlags.cc)
    flags->readMPMFlags(mpm_soln_ps, grid);
    if (flags->d_integrator_type == "implicit")
      throw ProblemSetupException("Can't use implicit integration with -mpm", __FILE__, __LINE__);

    mpm_soln_ps->get("do_grid_reset", d_doGridReset);
    mpm_soln_ps->get("minimum_particle_mass",    d_min_part_mass);
    mpm_soln_ps->get("maximum_particle_velocity",d_max_vel);
    
    std::vector<std::string> bndy_face_txt_list;
    mpm_soln_ps->get("boundary_traction_faces", bndy_face_txt_list);
    
    // convert text representation of face into FaceType
    for(std::vector<std::string>::const_iterator ftit(bndy_face_txt_list.begin());
        ftit!=bndy_face_txt_list.end();ftit++) {
        Patch::FaceType face = Patch::invalidFace;
        for(Patch::FaceType ft=Patch::startFace;ft<=Patch::endFace;
            ft=Patch::nextFace(ft)) {
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
      throw ProblemSetupException("To use material addition, one must specify manual_add_material==true in the input file.",
                                  __FILE__, __LINE__);
    }
  }
  
  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(flags->d_8or27==27){
    NGP=2;
    NGN=2;
  }

  MPMPhysicalBCFactory::create(prob_spec);

  contactModel = ContactFactory::create(d_myworld, prob_spec,sharedState,lb,flags);
  thermalContactModel =
    ThermalContactFactory::create(prob_spec, sharedState, lb,flags);

  heatConductionModel = scinew HeatConduction(sharedState,lb,flags);

  ProblemSpecP p = prob_spec->findBlock("DataArchiver");
  if(!p->get("outputInterval", d_outputInterval))
    d_outputInterval = 1.0;

  materialProblemSetup(prob_spec, d_sharedState, lb, flags);

//  GridP grid;
//  addMaterial(prob_spec, grid ,sharedState);
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
    MPMMaterial *mat = scinew MPMMaterial(ps, lb, flags,sharedState);
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
    MPMMaterial *mat = scinew MPMMaterial(ps, lb, flags,sharedState);
    sharedState->registerMPMMaterial(mat);

    // If new particles are to be created, create a copy of each material
    // without the associated geometry
    if (flags->d_createNewParticles) {
      MPMMaterial *mat_copy = scinew MPMMaterial();
      mat_copy->copyWithoutGeom(mat, flags,sharedState);    
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
  t->computes(lb->pTempPreviousLabel); // for therma  stresm analysis
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
  if (cout_dbg.active())
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
  if (cout_doing.active())
    cout_doing << "Doing SerialMPM::scheduleInitializeAddedMaterial " << endl;

  Task* t = scinew Task("SerialMPM::actuallyInitializeAddedMaterial",
                  this, &SerialMPM::actuallyInitializeAddedMaterial);
                                                                                
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

  MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(numMPMMatls-1);
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
  if (!flags->doMPMOnLevel(level))
    return;

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  scheduleApplyExternalLoads(             sched, patches, matls);
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  scheduleComputeHeatExchange(            sched, patches, matls);
  scheduleExMomInterpolated(              sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);
  scheduleComputeContactArea(             sched, patches, matls);
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
                                    d_sharedState->d_particleState_preReloc,
                                    lb->pXLabel, 
                                    d_sharedState->d_particleState,
                                    lb->pParticleIDLabel, matls);
}

void SerialMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;
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
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

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
  t->requires(Task::OldDW, lb->pErosionLabel,          gan,NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,             gan,NGP);
    
  //t->requires(Task::OldDW, lb->pExternalHeatRateLabel, gan,NGP);

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
  t->computes(lb->gVelocityInterpLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureNoBCLabel);
  t->computes(lb->gExternalHeatRateLabel);
  t->computes(lb->gNumNearParticlesLabel);
  t->computes(lb->TotalMassLabel);
  
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeHeatExchange(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;
  /* computeHeatExchange
   *   in(G.MASS, G.TEMPERATURE, G.EXTERNAL_HEAT_RATE)
   *   operation(peform heat exchange which will cause each of
   *   velocity fields to exchange heat according to 
   *   the temperature differences)
   *   out(G.EXTERNAL_HEAT_RATE) */

  if (cout_doing.active())
    cout_doing << getpid() << " Doing MPM::ThermalContact::computeHeatExchange " << endl;

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
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;
  contactModel->addComputesAndRequiresInterpolated(sched, patches, matls);
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
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  // for thermal stress analysis
  scheduleComputeParticleTempFromGrid(sched, patches, matls); 

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

// Compute particle temperature by interpolating grid temperature
// for thermal stress analysis
void SerialMPM::scheduleComputeParticleTempFromGrid(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls)
{
  Ghost::GhostType gac = Ghost::AroundCells;
  Task* t = scinew Task("MPM::computeParticleTempFromGrid",
                        this, &SerialMPM::computeParticleTempFromGrid);
  t->requires(Task::NewDW, lb->gTemperatureLabel, gac, NGN);
  t->computes(lb->pTempCurrentLabel);
  sched->addTask(t, patches, matls);
}

// Compute the accumulated strain energy
void SerialMPM::scheduleUpdateErosionParameter(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

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
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

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
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  Task* t = scinew Task("MPM::computeArtificialViscosity",
                        this, &SerialMPM::computeArtificialViscosity);

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, lb->pXLabel,                 Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,              Ghost::None);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,    Ghost::None);
  t->requires(Task::OldDW,lb->pSizeLabel,             Ghost::None);
  t->requires(Task::NewDW,lb->gVelocityLabel, gac, NGN);
  t->computes(lb->p_qLabel);

  sched->addTask(t, patches, matls);
}


void SerialMPM::scheduleComputeContactArea(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  /*
   * computeContactArea */
  if(d_bndy_traction_faces.size()>0) {
    Task* t = scinew Task("MPM::computeContactArea",
                          this, &SerialMPM::computeContactArea);
    
    Ghost::GhostType  gnone = Ghost::None;
    t->requires(Task::NewDW, lb->gVolumeLabel, gnone);
    for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
        ftit!=d_bndy_traction_faces.end();ftit++) {
      int iface = (int)(*ftit);
      t->computes(lb->BndyContactCellAreaLabel[iface]);
    }
    sched->addTask(t, patches, matls);
  }
}

void SerialMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

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
  t->requires(Task::NewDW,lb->gVolumeLabel, gnone);
  t->requires(Task::NewDW,lb->gVolumeLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain, gnone);
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,      gan,NGP);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,       gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::OldDW,lb->pMassLabel,                 gan,NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,              gan,NGP);
    
  t->requires(Task::NewDW, lb->pErosionLabel_preReloc,    gan, NGP);

  if(d_with_ice){
    t->requires(Task::NewDW, lb->pPressureLabel,          gan,NGP);
  }

  if(flags->d_artificial_viscosity){
    t->requires(Task::NewDW, lb->p_qLabel,                gan,NGP);
  }

  t->computes(lb->gInternalForceLabel);
  t->computes(lb->TotalVolumeDeformedLabel);
  
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


void SerialMPM::scheduleComputeInternalHeatRate(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{  
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  heatConductionModel->scheduleComputeInternalHeatRate(sched,patches,matls);
}

void SerialMPM::scheduleSolveEquationsMotion(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

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
#if 0
  if(d_with_ice){
    t->requires(Task::NewDW, lb->gradPAccNCLabel,   Ghost::None);
  }
#endif
  t->computes(lb->gAccelerationLabel);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleSolveHeatEquations(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  heatConductionModel->scheduleSolveHeatEquations(sched,patches,matls);
}

void SerialMPM::scheduleIntegrateAcceleration(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

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
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  heatConductionModel->scheduleIntegrateTemperatureRate(sched,patches,matls);
}

void SerialMPM::scheduleExMomIntegrated(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  /* exMomIntegrated
   *   in(G.MASS, G.VELOCITY_STAR, G.ACCELERATION)
   *   operation(peform operations which will cause each of
   *              velocity fields to feel the influence of the
   *              the others according to specific rules)
   *   out(G.VELOCITY_STAR, G.ACCELERATION) */

  contactModel->addComputesAndRequiresIntegrated(sched, patches, matls);
}

void SerialMPM::scheduleSetGridBoundaryConditions(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  Task* t=scinew Task("MPM::setGridBoundaryConditions",
                      this, &SerialMPM::setGridBoundaryConditions);
                  
  const MaterialSubset* mss = matls->getUnion();
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::NewDW, lb->gVelocityInterpLabel,   Ghost::None);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleCalculateDampingRate(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

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
    t->requires(Task::OldDW, lb->pSizeLabel, Ghost::None);

    t->computes(lb->pDampingRateLabel);
    sched->addTask(t, patches, matls);
  }
}

void SerialMPM::scheduleAddNewParticles(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

//if  manual_new_material==false, DON't do this task OR
//if  create_new_particles==true, DON'T do this task
  if (!flags->d_addNewMaterial || flags->d_createNewParticles) return;

//if  manual__new_material==true, DO this task OR
//if  create_new_particles==false, DO this task
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
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

//if  create_new_particles==false, DON't do this task OR
//if  manual_create_new_matl==true, DON'T do this task
  if (!flags->d_createNewParticles || flags->d_addNewMaterial) return;

//if  create_new_particles==true, DO this task OR
//if  manual_create_new_matl==false, DO this task 
  Task* t=scinew Task("MPM::convertLocalizedParticles", this, 
                      &SerialMPM::convertLocalizedParticles);

  int numMatls = d_sharedState->getNumMPMMatls();


  if (cout_convert.active())
    cout_convert << "MPM:scheduleConvertLocalizedParticles : numMatls = " << numMatls << endl;

  for(int m = 0; m < numMatls; m+=2){

    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    if (cout_convert.active())
      cout_convert << " Material = " << m << " mpm_matl = " <<mpm_matl<< endl;

    mpm_matl->getParticleCreator()->allocateVariablesAddRequires(t, mpm_matl,
                                                                 patches, lb);
    if (cout_convert.active())
      cout_convert << "   Done ParticleCreator::allocateVariablesAddRequires\n";
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    if (cout_convert.active())
      cout_convert << "   cm = " << cm << endl;

    cm->allocateCMDataAddRequires(t,mpm_matl,patches,lb);

    if (cout_convert.active())
      cout_convert << "   Done cm->allocateCMDataAddRequires = " << endl;

    cm->addRequiresDamageParameter(t, mpm_matl, patches);

    if (cout_convert.active())
      cout_convert << "   Done cm->addRequiresDamageParameter = " << endl;

  }

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                        const PatchSet* patches,
                                                        const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

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
  t->requires(Task::OldDW, lb->pVelocityLabel,                  gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                      gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,            gnone);
  t->requires(Task::NewDW, lb->pErosionLabel_preReloc,          gnone);
  // for thermal stress analysis
  t->requires(Task::NewDW, lb->pTempCurrentLabel,               gnone);
    

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
  t->computes(lb->pTempPreviousLabel_preReloc); // for thermal stress 
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

void SerialMPM::scheduleRefine(const PatchSet* patches, 
                               SchedulerP& sched)
{
  Task* task = scinew Task("SerialMPM::refine", this, &SerialMPM::refine);
  sched->addTask(task, patches, d_sharedState->allMPMMaterials());
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

  if (cout_doing.active())
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
                   0, Task::FineLevel, d_sharedState->refineFlagMaterials(), 
                   Task::NormalDomain, Ghost::None, 0);
  }
  task->modifies(d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials());
  task->modifies(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials());
  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allMPMMaterials());

}

/// Schedule to mark initial flags for AMR regridding
void SerialMPM::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                             SchedulerP& sched)
{

  if (cout_doing.active())
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

void SerialMPM::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("switchTest",this, &SerialMPM::switchTest);

  task->requires(Task::OldDW, d_sharedState->get_delt_label() );
  task->computes(d_sharedState->get_switch_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(),d_sharedState->allMaterials());

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

    if (cout_doing.active())
      cout_doing <<"Doing actuallyInitialize on patch " << patch->getID() <<"\t\t\t MPM"<< endl;


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
                                                const MaterialSubset* /*matls*/,
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

    if (cout_doing.active())
      cout_doing <<"Doing interpolateParticlesToGrid on patch " << patch->getID() <<"\t\t MPM"<< endl;


    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());


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
      constParticleVariable<Vector> pvelocity, pexternalforce,psize;
      constParticleVariable<double> pErosion;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);
      old_dw->get(psize,          lb->pSizeLabel,          pset);
      old_dw->get(pErosion,       lb->pErosionLabel,       pset);
      new_dw->get(pexternalforce, lb->pExtForceLabel_preReloc, pset);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity,gvelocityInterp;
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
      new_dw->allocateAndPut(gvelocityInterp,  lb->gVelocityInterpLabel,
                                                                     dwi,patch);
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
      gvolume.initialize(d_SMALL_NUM_MPM);
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
      Vector pmom;
      int n8or27=flags->d_8or27;

      double pSp_vol = 1./mpm_matl->getInitialDensity();
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx]);

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
            gSp_vol[ni[k]]        += pSp_vol           * pmass[idx] * S[k];
            gnumnearparticles[ni[k]] += 1.0;
            //  gexternalheatrate[ni[k]] += pexternalheatrate[idx]      * S[k];
          }
        }
      } // End of particle loop

      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter; 
        totalmass       += gmass[c];
        gmassglobal[c]  += gmass[c];
        gvolumeglobal[c]  += gvolume[c];
        gvelglobal[c]   += gvelocity[c];
        gvelocity[c]    /= gmass[c];
        gtempglobal[c]  += gTemperature[c];
        gTemperature[c] /= gmass[c];
        gTemperatureNoBC[c] = gTemperature[c];
        gSp_vol[c]      /= gmass[c];
        gvelocityInterp[c]=gvelocity[c];
      }

      // Apply grid boundary conditions to the velocity before storing the data

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity",   gvelocity,       n8or27);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gvelocity,       n8or27);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gvelocityInterp, n8or27);
      bc.setBoundaryCondition(patch,dwi,"Temperature",gTemperature,    n8or27);

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);

    }  // End loop over materials

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;
      gtempglobal[c] /= gmassglobal[c];
      gvelglobal[c] /= gmassglobal[c];
    }
    delete interpolator;
  }  // End loop over patches
}

void SerialMPM::computeStressTensor(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{

  if (cout_doing.active())
    cout_doing <<"Doing computeStressTensor:MPM: \n" ;

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

    cm->setWorld(d_myworld);
    cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);

    if (cout_dbg.active())
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

    if (cout_doing.active())
      cout_doing << getpid() << "Doing updateErosionParameter on patch "  << patch->getID() << "\t MPM"<< endl;


    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){

      if (cout_dbg.active())
        cout_dbg << "updateErosionParameter:: material # = " << m << endl;


      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      if (cout_dbg.active())
        cout_dbg << "updateErosionParameter:: mpm_matl* = " << mpm_matl << " dwi = " << dwi << " pset* = " << pset << endl;


      // Get the erosion data
      constParticleVariable<double> pErosion;
      ParticleVariable<double> pErosion_new;
      old_dw->get(pErosion, lb->pErosionLabel, pset);
      new_dw->allocateAndPut(pErosion_new, lb->pErosionLabel_preReloc, pset);

      if (cout_dbg.active())
        cout_dbg << "updateErosionParameter:: Got Erosion data" << endl;


      // Get the localization info
      ParticleVariable<int> isLocalized;
      new_dw->allocateTemporary(isLocalized, pset);
      ParticleSubset::iterator iter = pset->begin(); 
      for (; iter != pset->end(); iter++) isLocalized[*iter] = 0;
      mpm_matl->getConstitutiveModel()->getDamageParameter(patch, isLocalized,
                                                           dwi, old_dw,new_dw);

      if (cout_dbg.active())
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

      if (cout_dbg.active())
        cout_dbg << "updateErosionParameter:: Updated Erosion " << endl;


    }

    if (cout_dbg.active())
      cout_dbg <<"Done updateErosionParamter on patch "  << patch->getID() << "\t MPM"<< endl;

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

    if (cout_doing.active())
      cout_doing <<"Doing computeArtificialViscosity on patch " << patch->getID() <<"\t\t MPM"<< endl;


    // The following scheme for removing ringing behind a shock comes from:
    // VonNeumann, J.; Richtmyer, R. D. (1950): A method for the numerical
    // calculation of hydrodynamic shocks. J. Appl. Phys., vol. 21, pp. 232.

    Ghost::GhostType  gac   = Ghost::AroundCells;

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());

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
      old_dw->get(psize,   lb->pSizeLabel,                   pset);

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

        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx]);

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
    delete interpolator;
  }

}

void SerialMPM::computeContactArea(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* ,
                                   DataWarehouse* /*old_dw*/,
                                   DataWarehouse* new_dw)
{
  // six indices for each of the faces
  double bndyCArea[6] = {0,0,0,0,0,0};
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing computeContactArea on patch " << patch->getID() <<"\t\t\t MPM"<< endl;

    
    Vector dx = patch->dCell();
    
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      constNCVariable<double> gvolume;
      
      new_dw->get(gvolume, lb->gVolumeLabel, dwi, patch, Ghost::None, 0);
     
      for(list<Patch::FaceType>::const_iterator
          fit(d_bndy_traction_faces.begin()); 
          fit!=d_bndy_traction_faces.end();fit++) {       
        Patch::FaceType face = *fit;
        int iface = (int)(face);
        
        // Check if the face is on an external boundary
        if(patch->getBCType(face)==Patch::Neighbor)
           continue;
        
        // We are on the boundary, i.e. not on an interior patch
        // boundary, and also on the correct side, 

        // loop over face nodes to find boundary areas
// Because this calculation uses gvolume, particle volumes interpolated to
// the nodes, it will give 1/2 the expected value because the particle values
// are distributed to all nodes, not just those on this face.  It would require
// particles on the other side of the face to "fill" the nodal volumes and give
// the correct area when divided by the face normal cell dimension (celldepth).
// To correct for this, nodearea incorporates a factor of two.

        IntVector projlow, projhigh;
        patch->getFaceNodes(face, 0, projlow, projhigh);
        const double celldepth  = dx[iface/2];
        
        for (int i = projlow.x(); i<projhigh.x(); i++) {
          for (int j = projlow.y(); j<projhigh.y(); j++) {
            for (int k = projlow.z(); k<projhigh.z(); k++) {
              IntVector ijk(i,j,k);
              double nodearea         = 2.0*gvolume[ijk]/celldepth; // node area
              bndyCArea[iface] += nodearea;

            }
          }
        }
      } // faces
    } // materials
  } // patches
  
  // be careful only to put the fields that we have built
  // that way if the user asks to output a field that has not been built
  // it will fail early rather than just giving zeros.
  for(std::list<Patch::FaceType>::const_iterator 
      ftit(d_bndy_traction_faces.begin());
      ftit!=d_bndy_traction_faces.end();ftit++) {
    int iface = (int)(*ftit);
    new_dw->put(sum_vartype(bndyCArea[iface]),
                lb->BndyContactCellAreaLabel[iface]);
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
  Vector bndyTraction[6];
  for(int iface=0;iface<6;iface++) {
    bndyForce   [iface]  = Vector(0.);
    bndyTraction[iface]  = Vector(0.);
  }
  double partvoldef = 0.;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing computeInternalForce on patch " << patch->getID() <<"\t\t\t MPM"<< endl;


    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    Matrix3 Id;
    Id.Identity();

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());


    int numMPMMatls = d_sharedState->getNumMPMMatls();

    NCVariable<Matrix3>       gstressglobal;
    constNCVariable<double>   gvolumeglobal;
    new_dw->get(gvolumeglobal,  lb->gVolumeLabel,
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
      NCVariable<Matrix3>            gstress;
      constNCVariable<double>        gvolume;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel,                      pset);
      old_dw->get(pmass,   lb->pMassLabel,                   pset);
      new_dw->get(pvol,    lb->pVolumeDeformedLabel,         pset);
      new_dw->get(pstress, lb->pStressLabel_preReloc,        pset);
      old_dw->get(psize,   lb->pSizeLabel,                   pset);
      new_dw->get(gvolume,   lb->gVolumeLabel, dwi, patch, Ghost::None, 0);
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

      Matrix3 stressvol;
      Matrix3 stresspress;
      int n8or27 = flags->d_8or27;

      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx]);

        stressvol  = pstress[idx]*pvol[idx];
        //stresspress = pstress[idx] + Id*p_pressure[idx];
        stresspress = pstress[idx] + Id*p_pressure[idx] - Id*p_q[idx];
        partvoldef += pvol[idx];

        for (int k = 0; k < n8or27; k++){
          if(patch->containsNode(ni[k])){
            Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                       d_S[k].z()*oodx[2]);
            div *= pErosion[idx];
            internalforce[ni[k]] -= (div * stresspress)  * pvol[idx];
            gstress[ni[k]]       += stressvol * S[k];
          }
        }
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        gstressglobal[c] += gstress[c];
        gstress[c] /= gvolume[c];
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

        IntVector projlow, projhigh;
        patch->getFaceNodes(face, 0, projlow, projhigh);
        Vector norm = face_norm(face);
	double celldepth  = dx[iface/2]; // length in direction perpendicular to boundary

	// loop over face nodes to find boundary forces, average stress (traction).
	// Note that nodearea incorporates a factor of two as described in the
	// bndyCellArea calculation in order to get node face areas.
        
        for (int i = projlow.x(); i<projhigh.x(); i++) {
          for (int j = projlow.y(); j<projhigh.y(); j++) {
            for (int k = projlow.z(); k<projhigh.z(); k++) {
              IntVector ijk(i,j,k);        
              
              // flip sign so that pushing on boundary gives positive force
              bndyForce[iface] -= internalforce[ijk];

	      double nodearea   = 2.0*gvolume[ijk]/celldepth; // node area
              for(int ic=0;ic<3;ic++) for(int jc=0;jc<3;jc++) {
		bndyTraction[iface][ic] += gstress[ijk](ic,jc)*norm[jc]*nodearea;
              }
              
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
      gstressglobal[c] /= gvolumeglobal[c];
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
    
    // Use the face force and traction calculations to provide a second estimate of the
    // contact area.
    double bndyContactArea_iface = bndyContactCellArea_iface;
    if(bndyTraction[iface][iface/2]*bndyTraction[iface][iface/2]>1.e-12)
      bndyContactArea_iface = bndyForce[iface][iface/2]/bndyTraction[iface][iface/2];

    new_dw->put(sum_vartype(bndyContactArea_iface), lb->BndyContactAreaLabel[iface]);
  }
}


void SerialMPM::solveEquationsMotion(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing solveEquationsMotion on patch " << patch->getID() <<"\t\t\t MPM"<< endl;

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
      //constNCVariable<Vector> gradPAccNC;  // for MPMICE
      constNCVariable<double> mass;
 
      new_dw->get(internalforce,lb->gInternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(externalforce,lb->gExternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(mass,         lb->gMassLabel,          dwi, patch, gnone, 0);
#if 0
      if(d_with_ice){
        new_dw->get(gradPAccNC, lb->gradPAccNCLabel,     dwi, patch, gnone, 0);
      }
      else{
        NCVariable<Vector> gradPAccNC_create;
        new_dw->allocateTemporary(gradPAccNC_create,  patch);
        gradPAccNC_create.initialize(Vector(0.,0.,0.));
        gradPAccNC = gradPAccNC_create; // reference created data
      }
#endif
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
          acceleration[c] = acc +  gravity;
#if 0
+ gradPAccNC[c];
#endif
//                 acceleration[c] =
//                    (internalforce[c] + externalforce[c]
//                    -5000.*velocity[c]*mass[c])/mass[c]
//                    + gravity + gradPAccNC[c];
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

    if (cout_doing.active())
      cout_doing <<"Doing integrateAcceleration on patch " << patch->getID() <<"\t\t\t MPM"<< endl;


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


void SerialMPM::setGridBoundaryConditions(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing setGridBoundaryConditions on patch " << patch->getID()<<"\t\t MPM"<< endl;


    int numMPMMatls=d_sharedState->getNumMPMMatls();
    
    delt_vartype delT;            
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
                      
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<Vector> gvelocityInterp;
      int n8or27=flags->d_8or27;

      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,  dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,  dwi,patch);
      new_dw->get(gvelocityInterp,         lb->gVelocityInterpLabel,dwi, patch,
                                                                Ghost::None,0);
      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity",     gvelocity_star, n8or27);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",    gvelocity_star, n8or27);
//    bc.setBoundaryCondition(patch,dwi,"Acceleration", gacceleration,  n8or27);

      // Now recompute acceleration as the difference between the velocity
      // interpolated to the grid (no bcs applied) and the new velocity_star
      for(NodeIterator iter = patch->getNodeIterator(n8or27); !iter.done();
                                                               iter++){
        IntVector c = *iter;
        gacceleration[c] = (gvelocity_star[c] - gvelocityInterp[c])/delT;
      }

      // Set symmetry BCs on acceleration if called for
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

  if (cout_doing.active())
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

    if (cout_doing.active())
      cout_doing <<"Doing applyExternalLoads on patch "  << patch->getID() << "\t MPM"<< endl;


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

      if (cout_doing.active())
        cout_doing <<"Doing calculateDampingRate on patch "   << patch->getID() << "\t MPM"<< endl;


      double alphaDot = 0.0;
      int numMPMMatls=d_sharedState->getNumMPMMatls();

      ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
      vector<IntVector> ni;
      ni.reserve(interpolator->size());
      vector<double> S;
      S.reserve(interpolator->size());
      vector<Vector> d_S;
      d_S.reserve(interpolator->size());

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
        old_dw->get(psize, lb->pSizeLabel, pset);
        Ghost::GhostType  gac = Ghost::AroundCells;
        new_dw->get(gvelocity_star,   lb->gVelocityStarLabel,dwi,patch,gac,NGP);
        // Calculate artificial dampening rate based on the interpolated particle
        // velocities (ref. Ayton et al., 2002, Biophysical Journal, 1026-1038)
        // d(alpha)/dt = 1/Q Sum(vp*^2)
        ParticleSubset::iterator iter = pset->begin();
        for(;iter != pset->end(); iter++){
          particleIndex idx = *iter;
          interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,
                                                              d_S,psize[idx]);
          Vector vel(0.0,0.0,0.0);
          for (int k = 0; k < flags->d_8or27; k++) 
            vel += gvelocity_star[ni[k]]*S[k];
          alphaDot += Dot(vel,vel);
        }
        alphaDot /= flags->d_artificialDampCoeff;
      } 
      new_dw->put(sum_vartype(alphaDot), lb->pDampingRateLabel);
      delete interpolator;
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

    if (cout_doing.active())
      cout_doing <<"Doing addNewParticles on patch "  << patch->getID() << "\t MPM"<< endl;

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    // Find the mpm material that the void particles are going to change
    // into.
    MPMMaterial* null_matl = 0;
    int null_dwi = -1;
    for (int void_matl = 0; void_matl < numMPMMatls; void_matl++) {
      null_dwi = d_sharedState->getMPMMaterial(void_matl)->nullGeomObject();

      if (cout_dbg.active())
        cout_dbg << "Null DWI = " << null_dwi << endl;

      if (null_dwi != -1) {
        null_matl = d_sharedState->getMPMMaterial(void_matl);
        null_dwi = null_matl->getDWIndex();
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

          if (cout_dbg.active())
            cout_dbg << "damage[" << *iter << "]=" << damage[*iter] << endl;

          delset->addParticle(*iter);
        }
      }
      
      // Find the mpm material that corresponds to the void particles.
      // Will probably be the same type as the deleted ones, but have
      // different parameters.
      
      
      int numparticles = delset->numParticles();

      if (cout_dbg.active())
        cout_dbg << "Num Failed Particles = " << numparticles << endl;

      if (numparticles != 0) {

        if (cout_dbg.active())
          cout_dbg << "Deleted " << numparticles << " particles" << endl;

        ParticleCreator* particle_creator = null_matl->getParticleCreator();
        ParticleSet* set_add = scinew ParticleSet(numparticles);
        ParticleSubset* addset = scinew ParticleSubset(set_add,true,null_dwi,
                                                       patch,numparticles);

        if (cout_dbg.active()) {
          cout_dbg << "Address of delset = " << delset << endl;
          cout_dbg << "Address of pset = " << pset << endl;
          cout_dbg << "Address of set_add = " << set_add << endl;
          cout_dbg << "Address of addset = " << addset << endl;
        }

        
        map<const VarLabel*, ParticleVariableBase*>* newState
          = scinew map<const VarLabel*, ParticleVariableBase*>;

        if (cout_dbg.active()) {
          cout_dbg << "Address of newState = " << newState << endl;
          cout_dbg << "Null Material" << endl;
        }

        //vector<const VarLabel* > particle_labels = 
        //  particle_creator->returnParticleState();

        //printParticleLabels(particle_labels, old_dw, null_dwi,patch);

        if (cout_dbg.active())
          cout_dbg << "MPM Material" << endl;

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

        if (cout_dbg.active()){
          cout_dbg << "addset num particles = " << addset->numParticles()
                   << " for material " << addset->getMatlIndex() << endl;
        }

        new_dw->addParticles(patch,null_dwi,newState);

        if (cout_dbg.active())
           cout_dbg << "Calling deleteParticles for material: " << dwi << endl;

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

    if (cout_doing.active())
      cout_doing <<"Doing convertLocalizedParticles on patch " << patch->getID() << "\t MPM"<< endl;


    int numMPMMatls=d_sharedState->getNumMPMMatls();

    if (cout_convert.active()) {
      cout_convert << "MPM::convertLocalizeParticles:: on patch"
           << patch->getID() << " numMPMMaterials = " << numMPMMatls
           << endl;
    }

    for(int m = 0; m < numMPMMatls; m+=2){

      if (cout_convert.active())
        cout_convert << " material # = " << m << endl;


      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      if (cout_convert.active()) {
        cout_convert << " mpm_matl* = " << mpm_matl
                     << " dwi = " << dwi << " pset* = " << pset << endl;
      }


      ParticleVariable<int> isLocalized;

      //old_dw->allocateTemporary(isLocalized, pset);
      new_dw->allocateTemporary(isLocalized, pset);

      ParticleSubset::iterator iter = pset->begin(); 
      for (; iter != pset->end(); iter++) isLocalized[*iter] = 0;

      ParticleSubset* delset = scinew ParticleSubset(pset->getParticleSet(),
                                                     false, dwi, patch, 0);
      
      mpm_matl->getConstitutiveModel()->getDamageParameter(patch, isLocalized,
                                                           dwi, old_dw,new_dw);

      if (cout_convert.active())
        cout_convert << " Got Damage Parameter" << endl;


      iter = pset->begin(); 
      for (; iter != pset->end(); iter++) {
        if (isLocalized[*iter]) {

          if (cout_convert.active())
            cout_convert << "damage[" << *iter << "]="
                         << isLocalized[*iter] << endl;
          delset->addParticle(*iter);
        }
      }

      if (cout_convert.active())
       cout_convert << " Created Delset ";


      int numparticles = delset->numParticles();

      if (cout_convert.active())
        cout_convert << " numparticles = " << numparticles << endl;


      if (numparticles != 0) {

        if (cout_convert.active()) {
          cout_convert << " Converting " 
                       << numparticles << " particles of material " 
                       <<  m  << " into particles of material " << (m+1) 
                       << " in patch " << p << endl;
        }


        MPMMaterial* conv_matl = d_sharedState->getMPMMaterial(m+1);
        int conv_dwi = conv_matl->getDWIndex();
      
        ParticleCreator* particle_creator = conv_matl->getParticleCreator();
        ParticleSet* set_add = scinew ParticleSet(numparticles);
        ParticleSubset* addset = scinew ParticleSubset(set_add, true,
                                                       conv_dwi, patch,
                                                       numparticles);
        
        map<const VarLabel*, ParticleVariableBase*>* newState
          = scinew map<const VarLabel*, ParticleVariableBase*>;

        if (cout_convert.active())
          cout_convert << "New Material" << endl;

        //vector<const VarLabel* > particle_labels = 
        //  particle_creator->returnParticleState();
        //printParticleLabels(particle_labels, old_dw, conv_dwi,patch);

        if (cout_convert.active())
          cout_convert << "MPM Material" << endl;

        //vector<const VarLabel* > mpm_particle_labels = 
        //  mpm_matl->getParticleCreator()->returnParticleState();
        //printParticleLabels(mpm_particle_labels, old_dw, dwi,patch);

        particle_creator->allocateVariablesAdd(lb, new_dw, addset, newState,
                                               delset, old_dw);
        
        conv_matl->getConstitutiveModel()->allocateCMDataAdd(new_dw, addset,
                                                             newState, delset,
                                                             old_dw);

        if (cout_convert.active()) {
          cout_convert << "addset num particles = " << addset->numParticles()
                       << " for material " << addset->getMatlIndex() << endl;
        }

        new_dw->addParticles(patch, conv_dwi, newState);
        new_dw->deleteParticles(delset);
        
        //delete addset;
      } 
      else delete delset;
    }

    if (cout_convert.active()) {
      cout_convert <<"Done convertLocalizedParticles on patch " 
                   << patch->getID() << "\t MPM"<< endl;
    }

  }

  if (cout_convert.active())
    cout_convert << "Completed convertLocalizedParticles " << endl;

  
}

// for thermal stress analysis
void SerialMPM::computeParticleTempFromGrid(const ProcessorGroup*,
                                            const PatchSubset* patches,
                                            const MaterialSubset*,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{                                           
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
	
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni; 
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();   

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constNCVariable<double> gTemperature;
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gTemperature, lb->gTemperatureLabel, dwi,patch, gac, NGP);

      constParticleVariable<Point> px;
      constParticleVariable<Vector> psize;
      old_dw->get(px,    lb->pXLabel,    pset);
      old_dw->get(psize, lb->pSizeLabel, pset);
      
      ParticleVariable<double> pTempCur;
      new_dw->allocateAndPut(pTempCur,lb->pTempCurrentLabel,pset);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++) {
        particleIndex idx = *iter;
        double pTemp=0.0;
	
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx]);
        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          pTemp += gTemperature[node] * S[k];
        }
        pTempCur[idx]=pTemp;
      } // End of loop over iter        
    } // End of loop over m
    delete interpolator;
  } // End of loop over p 
}

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
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
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());

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
      constParticleVariable<double> pmass, pvolume, pTemperature;
      ParticleVariable<double> pmassNew,pvolumeNew,pTempNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;
      constParticleVariable<double> pErosion;
      
      // for thermal stress analysis
      constParticleVariable<double> pTempCurrent; 
      ParticleVariable<double> pTempPreNew; 

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<double> gTemperatureRate, gTemperature, gTemperatureNoBC;
      constNCVariable<double> dTdt, massBurnFrac, frictionTempRate;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pids,         lb->pParticleIDLabel,                pset);
      new_dw->get(pvolume,      lb->pVolumeDeformedLabel,            pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
      new_dw->get(pErosion,     lb->pErosionLabel_preReloc,          pset);
      // for thermal stress analysis
      new_dw->get(pTempCurrent, lb->pTempCurrentLabel,               pset); 

      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pxx,          lb->pXXLabel,                  pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pvolumeNew,   lb->pVolumeLabel_preReloc,     pset);
      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);
      // for thermal stress analysis
      new_dw->allocateAndPut(pTempPreNew, lb->pTempPreviousLabel_preReloc,pset);

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
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx]);

        Vector vel(0.0,0.0,0.0);
        Vector acc(0.0,0.0,0.0);
        double fricTempRate = 0.0;
        double tempRate = 0.0;
        double burnFraction = 0.0;

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          S[k] *= pErosion[idx];
          vel      += gvelocity_star[node]  * S[k];
          acc      += gacceleration[node]   * S[k];

          fricTempRate = frictionTempRate[node]*flags->d_addFrictionWork;
          tempRate += (gTemperatureRate[node] + dTdt[node] +
                       fricTempRate)   * S[k];
          burnFraction += massBurnFrac[node]     * S[k];
        }

        // Update the particle's position and velocity
        pxnew[idx]           = px[idx]    + vel*delT*move_particles;
        pdispnew[idx]        = pdisp[idx] + vel*delT;
        pvelocitynew[idx]    = pvelocity[idx]    + (acc - alpha*vel)*delT;
        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx]    + pdispnew[idx];
        pTempNew[idx]        = pTemperature[idx] + tempRate*delT;
        pTempPreNew[idx]     = pTempCurrent[idx]; // for thermal stress

        if (cout_heat.active()) {
          cout_heat << "MPM::Particle = " << idx 
                    << " T_old = " << pTemperature[idx]
                    << " Tdot = " << tempRate
                    << " dT = " << (tempRate*delT)
                    << " T_new = " << pTempNew[idx] << endl;
        }


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
//	  cout << "Material = " << m << " Deleted Particle = " << idx 
//               << " xold = " << px[idx] << " xnew = " << pxnew[idx]
//	       << " vold = " << pvelocity[idx] << " vnew = "<< pvelocitynew[idx]
//	       << " massold = " << pmass[idx] << " massnew = " << pmassNew[idx]
//	       << " volnew = " << pvolumeNew[idx] << endl;
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

    // DON'T MOVE THESE!!!
    new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
    new_dw->put(sum_vartype(thermal_energy), lb->ThermalEnergyLabel);
    new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
    new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);

    // cout << "Solid mass lost this timestep = " << massLost << endl;
    // cout << "Solid momentum after advection = " << CMV << endl;

    // cout << "THERMAL ENERGY " << thermal_energy << endl;

    delete interpolator;
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

    if (amr_doing.active()) 
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

  if (cout_doing.active())
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

      if (amr_doing.active())
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
        ParticleVariable<double> pErosion;
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
        new_dw->allocateAndPut(pErosion,       lb->pErosionLabel,       pset);

      }
    }
  }

} // end refine()

void SerialMPM::scheduleCheckNeedAddMPMMaterial(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{

  if (cout_doing.active())
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

  if (cout_doing.active())
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

  if (cout_doing.active())
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

    if(need_add_flag < -0.1){
      d_sharedState->setNeedAddMaterial(-99);
      flags->d_canAddMPMMaterial=false;
      cout << "MPM setting NAM to -99" << endl;
    }
    else{
      d_sharedState->setNeedAddMaterial(0);
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


void SerialMPM::switchTest(const ProcessorGroup* group,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw)
{
  int time_step = d_sharedState->getCurrentTopLevelTimeStep();
  double sw = 0;
#if 1
  if (time_step == 6 )
    sw = 1;
  else
    sw = 0;
#endif

  max_vartype switch_condition(sw);
  new_dw->put(switch_condition,d_sharedState->get_switch_label(),getLevel(patches));
}
