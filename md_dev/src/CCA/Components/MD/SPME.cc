/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

#include <CCA/Components/MD/SPME.h>
#include <CCA/Components/MD/ShiftedCardinalBSpline.h>
#include <CCA/Components/MD/SPMEMapPoint.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/SimpleGrid.h>
#include <CCA/Components/MD/PatchMaterialKey.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <iomanip>
#include <cstring>

#include <sci_values.h>
#include <sci_defs/fftw_defs.h>

#ifdef DEBUG
#include <Core/Util/FancyAssert.h>
#endif
#define IV_ZERO IntVector(0,0,0)

using namespace Uintah;

extern SCIRun::Mutex cerrLock;

static DebugStream spme_cout("SPMECout", false);
static DebugStream spme_dbg("SPMEDBG", false);

SPME::SPME() :
    d_Qlock("node-local Q lock"), d_spmeLock("SPMEPatch data structures lock")
{

}

SPME::SPME(MDSystem* system,
           const double ewaldBeta,
           const bool isPolarizable,
           const double tolerance,
           const IntVector& kLimits,
           const int splineOrder,
           const int maxPolarizableIterations) :
      d_system(system),
      d_ewaldBeta(ewaldBeta),
      d_polarizable(isPolarizable),
      d_polarizationTolerance(tolerance),
      d_kLimits(kLimits),
      d_maxPolarizableIterations(maxPolarizableIterations),
      d_Qlock("node-local Q lock"),
      d_spmeLock("SPME shared data structure lock")
{
  d_interpolatingSpline = ShiftedCardinalBSpline(splineOrder);
  d_electrostaticMethod = Electrostatics::SPME;
}

SPME::~SPME()
{
  // cleanup the memory we have dynamically allocated
  std::map<int, SPMEPatch*>::iterator SPMEPatchIterator;

  for (SPMEPatchIterator = d_spmePatchMap.begin(); SPMEPatchIterator != d_spmePatchMap.end(); ++SPMEPatchIterator) {
    SPMEPatch* currSPMEPatch = SPMEPatchIterator->second;
    delete currSPMEPatch;
  }

  if (d_Q_nodeLocal) { delete d_Q_nodeLocal; }

  if (d_Q_nodeLocalScratch) { delete d_Q_nodeLocalScratch; }
}


//-----------------------------------------------------------------------------
// Interface implementations
void SPME::initialize(const ProcessorGroup* pg,
                      const PatchSubset* patches,
                      const MaterialSubset* materials,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  // We call SPME::initialize from MD::initialize, or if we've somehow maintained our object across a system change

  // Get useful information from global system descriptor to work with locally.
  d_unitCell        = d_system->getUnitCell();
  d_inverseUnitCell = d_system->getInverseCell();
  d_systemVolume    = d_system->getCellVolume();

  // now create SoleVariables to place in the DW; global Q and the FFTW plans (forward and backward)
  // FFTW plans for forward and backward 3D transform of global Q grid
  SoleVariable<fftw_plan> forwardTransformPlan;
  SoleVariable<fftw_plan> backwardTransformPlan;

  // now the local version of the global Q and Q_scratch arrays
  IntVector zero(0, 0, 0);
  d_Q_nodeLocal = scinew SimpleGrid<dblcomplex>(d_kLimits, zero, IV_ZERO, 0);
  d_Q_nodeLocalScratch = scinew SimpleGrid<dblcomplex>(d_kLimits, zero, IV_ZERO, 0);

  /*
   * Now do FFTW setup for the global FFTs...
   *
   * ptrdiff_t is a type able to represent the result of any valid pointer subtraction operation.
   * It is a standard C integer type which is (at least) 32 bits wide
   *   on a 32-bit machine and 64 bits wide on a 64-bit machine.
   */
  const ptrdiff_t xdim = d_kLimits(0);
  const ptrdiff_t ydim = d_kLimits(1);
  const ptrdiff_t zdim = d_kLimits(2);

  ptrdiff_t alloc_local, local_n, local_start;
  fftw_plan forwardPlan, backwardPlan;

  // Must initialize FFTW MPI before FFTW_MPI calls are made
  fftw_mpi_init();

  // This is the local portion of the global FFT array that will reside on each portion (slab decomposition)
  alloc_local = fftw_mpi_local_size_3d(xdim, ydim, zdim, pg->getComm(), &local_n, &local_start);
  d_localFFTData.complexData = fftw_alloc_complex(alloc_local);
  d_localFFTData.numElements = local_n;
  d_localFFTData.startAddress = local_start;

  // create the forward and reverse FFT MPI plans
  fftw_complex* complexData = d_localFFTData.complexData;
  forwardPlan = fftw_mpi_plan_dft_3d(xdim, ydim, zdim, complexData, complexData, pg->getComm(), FFTW_FORWARD, FFTW_MEASURE);
  backwardPlan = fftw_mpi_plan_dft_3d(xdim, ydim, zdim, complexData, complexData, pg->getComm(), FFTW_BACKWARD, FFTW_MEASURE);

  // set the SoleVariable data to be the forward and reverse FFTW plans
  forwardTransformPlan.setData(forwardPlan);
  backwardTransformPlan.setData(backwardPlan);

  // Initially register sole and reduction variables in the DW (initial timestep)
  new_dw->put(sum_vartype(0.0), d_lb->spmeFourierEnergyLabel);
  new_dw->put(matrix_sum(0.0), d_lb->spmeFourierStressLabel);

  SoleVariable<double> dependency;
  new_dw->put(dependency, d_lb->electrostaticsDependencyLabel);
  new_dw->put(forwardTransformPlan, d_lb->forwardTransformPlanLabel);
  new_dw->put(backwardTransformPlan, d_lb->backwardTransformPlanLabel);

// ------------------------------------------------------------------------
// Allocate and map the SPME patches
  Vector kReal = d_kLimits.asVector();

  size_t numPatches = patches->size();

  Vector systemCellExtent = (d_system->getCellExtent()).asVector();

  int splineSupport = d_interpolatingSpline.getSupport();
  IntVector plusGhostExtents(splineSupport, splineSupport, splineSupport);
  IntVector minusGhostExtents(0,0,0);

  for(size_t p = 0; p < numPatches; ++p) {

    const Patch* patch = patches->get(p);
    Vector patchLowIndex  = (patch->getCellLowIndex()).asVector();
    Vector patchHighIndex = (patch->getCellHighIndex()).asVector();
    Vector localCellExtent = patchHighIndex - patchLowIndex;

    double localCellVolumeFraction = (localCellExtent.x() * localCellExtent.y() * localCellExtent.z())
                                     / (systemCellExtent.x() * systemCellExtent.y() * systemCellExtent.z());

    IntVector patchKLow, patchKHigh;
    for (size_t index = 0; index < 3; ++index) {
      patchKLow[index] = floor(kReal[index] * (patchLowIndex[index] / systemCellExtent[index]));
      patchKHigh[index] = floor(kReal[index] * (patchHighIndex[index] / systemCellExtent[index]));
    }

    IntVector patchKGridExtents = (patchKHigh - patchKLow); // Number of K grid points in the local patch
    IntVector patchKGridOffset  = patchKLow;                // Starting indices for K grid in local patch

    // Instantiates an SPMEpatch which pre-allocates memory for all the local variables within an SPMEpatch
    SPMEPatch* spmePatch = new SPMEPatch(patchKGridExtents, patchKGridOffset, plusGhostExtents, minusGhostExtents,
                                         patch, localCellVolumeFraction, splineSupport, d_system);

    // Map the current spmePatch into the SPME object.
    d_spmeLock.writeLock();
    d_spmePatchMap.insert(SPMEPatchKey(patch->getID(),spmePatch));
    d_spmeLock.writeUnlock();
  }
}

// Note:  Must run SPME->setup() each time there is a new box/K grid mapping (e.g. every step for NPT)
//          This should be checked for in the system electrostatic driver
void SPME::setup(const ProcessorGroup* pg,
                 const PatchSubset*    patches,
                 const MaterialSubset* materials,
                 DataWarehouse*        old_dw,
                 DataWarehouse*        new_dw)
{
  size_t numPatches = patches->size();
  size_t splineSupport = d_interpolatingSpline.getSupport();

  IntVector  plusGhostExtents(splineSupport,splineSupport,splineSupport);
  IntVector minusGhostExtents(0,0,0);

  for(size_t p = 0; p < numPatches; ++p) {

    const Patch* patch = patches->get(p);
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;
    SimpleGrid<Matrix3>* stressPrefactor = currentSPMEPatch->getStressPrefactor();
    IntVector spmePatchExtents = currentSPMEPatch->getLocalExtents();
    IntVector spmePatchOffset  = currentSPMEPatch->getGlobalOffset();

    calculateStressPrefactor(stressPrefactor, spmePatchExtents, spmePatchOffset);
    SimpleGrid<double>* fTheta = currentSPMEPatch->getTheta();

    // No ghost cells; internal only
    SimpleGrid<double> fBGrid(spmePatchExtents, spmePatchOffset, IV_ZERO, 0);
    SimpleGrid<double> fCGrid(spmePatchExtents, spmePatchOffset, IV_ZERO, 0);

    // A SimpleGrid<double> of B(m1,m2,m3)=|b1(m1)|^2 * |b2(m2)|^2 * |b3(m3)|^2
    SPME::calculateBGrid(fBGrid, spmePatchExtents, spmePatchOffset);

    // A SimpleGrid<double> of C(m1,m2,m3)=(1/(PI*V))*exp(-PI^2*M^2/Beta^2)/M^2
    SPME::calculateCGrid(fCGrid, spmePatchExtents, spmePatchOffset);

    // Composite B*C into Theta
// Stubbing out interface to swap dimensions for FFT efficiency
//    int systemMaxKDimension = d_system->getMaxKDimensionIndex();
//    int systemMidKDimension = d_system->getMidDimensionIndex();
//    int systemMinKDimension = d_system->getMinDimensionIndex();
    int systemMaxKDimension = 0; // X for now
    int systemMidKDimension = 1; // Y for now
    int systemMinKDimension = 2; // Z for now
    size_t x_extent = spmePatchExtents[systemMaxKDimension];
    size_t y_extent = spmePatchExtents[systemMidKDimension];
    size_t z_extent = spmePatchExtents[systemMinKDimension];
    for (size_t xIndex = 0; xIndex < x_extent; ++xIndex) {
      for (size_t yIndex = 0; yIndex < y_extent; ++yIndex) {
        for (size_t zIndex = 0; zIndex < z_extent; ++zIndex) {
          (*fTheta)(xIndex, yIndex, zIndex) = fBGrid(xIndex, yIndex, zIndex) * fCGrid(xIndex, yIndex, zIndex);
        }
      }
    }
  }
  SoleVariable<double> dependency;
  old_dw->get(dependency, d_lb->electrostaticsDependencyLabel);
  new_dw->put(dependency, d_lb->electrostaticsDependencyLabel);
}

void SPME::calculate(const ProcessorGroup* pg,
                     const PatchSubset* perProcPatches,
                     const MaterialSubset* materials,
                     DataWarehouse* parentOldDW,
                     DataWarehouse* parentNewDW,
                     SchedulerP& subscheduler,
                     const LevelP& level,
                     SimulationStateP& sharedState)
{
  // this PatchSet is used for SPME::setup and also many of the subscheduled tasks
  const PatchSet* patches = level->eachPatch();

  // need the full material set
  const MaterialSet* allMaterials = sharedState->allMaterials();
  const MaterialSubset* allMaterialsUnion = allMaterials->getUnion();

  // temporarily turn off parentDW scrubbing
  DataWarehouse::ScrubMode parentOldDW_scrubmode = parentOldDW->setScrubbing(DataWarehouse::ScrubNone);
  DataWarehouse::ScrubMode parentNewDW_scrubmode = parentNewDW->setScrubbing(DataWarehouse::ScrubNone);

  GridP grid = level->getGrid();
  subscheduler->setParentDWs(parentOldDW, parentNewDW);
  subscheduler->advanceDataWarehouse(grid);

  DataWarehouse* subOldDW = subscheduler->get_dw(2);
  DataWarehouse* subNewDW = subscheduler->get_dw(3);

  // transfer data from parentOldDW to subDW
  subNewDW->transferFrom(parentOldDW, d_lb->pXLabel, perProcPatches, allMaterialsUnion);
  subNewDW->transferFrom(parentOldDW, d_lb->pChargeLabel, perProcPatches, allMaterialsUnion);
  subNewDW->transferFrom(parentOldDW, d_lb->pParticleIDLabel, perProcPatches, allMaterialsUnion);

  // explicitly forward SoleVariables to subDW (NOTE: transferFrom only works with Grid and Particle Variables)
  SoleVariable<fftw_plan> forwardTransformPlan;
  SoleVariable<fftw_plan> backwardTransformPlan;
  parentOldDW->get(forwardTransformPlan, d_lb->forwardTransformPlanLabel);
  parentOldDW->get(backwardTransformPlan, d_lb->backwardTransformPlanLabel);
  subNewDW->put(forwardTransformPlan, d_lb->forwardTransformPlanLabel);
  subNewDW->put(backwardTransformPlan, d_lb->backwardTransformPlanLabel);

  // reduction variables
  //  sum_vartype spmeFourierEnergy;
  //  matrix_sum spmeFourierStress;
  //  parentOldDW->get(spmeFourierEnergy, d_lb->spmeFourierEnergyLabel);
  //  parentOldDW->get(spmeFourierStress, d_lb->spmeFourierStressLabel);
  //  subNewDW->put(spmeFourierEnergy, d_lb->spmeFourierEnergyLabel);
  //  subNewDW->put(spmeFourierStress, d_lb->spmeFourierStressLabel);
    parentNewDW->put(sum_vartype(0.0), d_lb->spmeFourierEnergyLabel);
    parentNewDW->put(matrix_sum(0.0), d_lb->spmeFourierStressLabel);


  bool converged = false;
  int numIterations = 0;
  bool recompileSubscheduler = true;

  while (!converged && (numIterations < d_maxPolarizableIterations)) {

    // compile task graph (once)
    if (recompileSubscheduler) {

      subscheduler->initialize(3, 1);

      subscheduler->overrideVariableBehavior(d_lb->forwardTransformPlanLabel->getName(), false, false, false, true, true);
      subscheduler->overrideVariableBehavior(d_lb->backwardTransformPlanLabel->getName(), false, false, false, true, true);

      // prep for the forward FFT
      scheduleCalculatePreTransform(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);

      // Q grid reductions for forward FFT
      scheduleReduceNodeLocalQ(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);

      // Forward transform
      scheduleTransformRealToFourier(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW, level);

      // Do Fourier space calculations on transformed data
      scheduleCalculateInFourierSpace(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);

      // Reverse transform
      scheduleTransformFourierToReal(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW, level);

      // Redistribute force grid
      scheduleDistributeNodeLocalQ(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);

      // compile task graph
      subscheduler->compile();

      // make sure the above only happens once
      recompileSubscheduler = false;
    }

    // Need to re-map subNewDW
    subNewDW = subscheduler->get_dw(3);

    // move subNewDW to subOldDW
    subscheduler->advanceDataWarehouse(grid);
//    subOldDW->setScrubbing(DataWarehouse::ScrubComplete);
    subNewDW->setScrubbing(DataWarehouse::ScrubNone);

    //  execute the tasks
    subscheduler->execute();

    checkConvergence();
    numIterations++;
  }

  // Need to re-map subNewDW so we're "getting" from the right subDW when forwarding SoleVariables
  subNewDW = subscheduler->get_dw(3);

  /*
   * No ParticleVariable products from iterations (read only info from ParentOldDW) so don't need forward these
   * subNewDW --> parentNewDW, just need to forward the SoleVariables; parentOldDW --> parentNewDW, as the
   * computes statements specify in MD::scheduleElectrostaticsCalculate
   */

  // Push Sole Variables up to the parent DW
  SoleVariable<SimpleGrid<dblcomplex>*> QGridParent;
  SoleVariable<fftw_plan> forwardTransformPlanParent;
  SoleVariable<fftw_plan> backwardTransformPlanParent;
  subNewDW->get(forwardTransformPlanParent, d_lb->forwardTransformPlanLabel);
  subNewDW->get(backwardTransformPlanParent, d_lb->backwardTransformPlanLabel);
  parentNewDW->put(forwardTransformPlanParent, d_lb->forwardTransformPlanLabel);
  parentNewDW->put(backwardTransformPlanParent, d_lb->backwardTransformPlanLabel);

  // Push Reduction Variables up to the parent DW
  sum_vartype spmeFourierEnergyNew;
  matrix_sum spmeFourierStressNew;
  subNewDW->get(spmeFourierEnergyNew, d_lb->spmeFourierEnergyLabel);
  subNewDW->get(spmeFourierStressNew, d_lb->spmeFourierStressLabel);
  parentNewDW->put(spmeFourierEnergyNew, d_lb->spmeFourierEnergyLabel);
  parentNewDW->put(spmeFourierStressNew, d_lb->spmeFourierStressLabel);

  //  Turn scrubbing back on
  parentOldDW->setScrubbing(parentOldDW_scrubmode);
  parentNewDW->setScrubbing(parentNewDW_scrubmode);
}

void SPME::finalize(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* materials,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw)
{
  // Do force spreading
  SPME::calculatePostTransform(pg, patches, materials, old_dw, new_dw);
}

void SPME::scheduleCalculatePreTransform(SchedulerP& sched,
                                         const ProcessorGroup* pg,
                                         const PatchSet* patches,
                                         const MaterialSet* materials,
                                         DataWarehouse* subOldDW,
                                         DataWarehouse* subNewDW)
{
  printSchedule(patches, spme_cout, "SPME::scheduleCalculatePreTransform");

  Task* task = scinew Task("SPME::calculatePreTransform", this, &SPME::calculatePreTransform);

  int CUTOFF_RADIUS = d_system->getRequiredGhostCells();

  task->requires(Task::ParentNewDW, d_lb->pXLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_lb->pChargeLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_lb->pParticleIDLabel, Ghost::AroundNodes, CUTOFF_RADIUS);

  task->computes(d_lb->subSchedulerDependencyLabel);
  task->computes(d_lb->pXLabel);
  task->computes(d_lb->pChargeLabel);
  task->computes(d_lb->pParticleIDLabel);

  sched->addTask(task, patches, materials);
}

void SPME::scheduleReduceNodeLocalQ(SchedulerP& sched,
                                    const ProcessorGroup* pg,
                                    const PatchSet* patches,
                                    const MaterialSet* materials,
                                    DataWarehouse* subOldDW,
                                    DataWarehouse* subNewDW)
{
  printSchedule(patches, spme_cout, "SPME::scheduleReduceNodeLocalQ");

  Task* task = scinew Task("SPME::reduceNodeLocalQ", this, &SPME::reduceNodeLocalQ);

  task->requires(Task::NewDW, d_lb->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
  task->modifies(d_lb->subSchedulerDependencyLabel);

  sched->addTask(task, patches, materials);
}

void SPME::scheduleTransformRealToFourier(SchedulerP& sched,
                                          const ProcessorGroup* pg,
                                          const PatchSet* patches,
                                          const MaterialSet* materials,
                                          DataWarehouse* subOldDW,
                                          DataWarehouse* subNewDW,
                                          const LevelP& level)
{
  printSchedule(patches, spme_cout, "SPME::scheduleTransformRealToFourier");

  Task* task = scinew Task("SPME::transformRealToFourier", this, &SPME::transformRealToFourier);
  task->setType(Task::OncePerProc);
  task->usesMPI(true);

  task->requires(Task::NewDW, d_lb->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
  task->requires(Task::OldDW, d_lb->forwardTransformPlanLabel);
  task->requires(Task::OldDW, d_lb->backwardTransformPlanLabel);

  task->modifies(d_lb->subSchedulerDependencyLabel);
  task->computes(d_lb->forwardTransformPlanLabel);

  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perproc_patches = loadBal->getPerProcessorPatchSet(level);

  sched->addTask(task, perproc_patches, materials);
}

void SPME::scheduleCalculateInFourierSpace(SchedulerP& sched,
                                           const ProcessorGroup* pg,
                                           const PatchSet* patches,
                                           const MaterialSet* materials,
                                           DataWarehouse* subOldDW,
                                           DataWarehouse* subNewDW)
{
  printSchedule(patches, spme_cout, "SPME::scheduleCalculateInFourierSpace");

  Task* task = scinew Task("SPME::calculateInFourierSpace", this, &SPME::calculateInFourierSpace);

  task->requires(Task::NewDW, d_lb->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);

  task->modifies(d_lb->subSchedulerDependencyLabel);
  task->computes(d_lb->spmeFourierEnergyLabel);
  task->computes(d_lb->spmeFourierStressLabel);

  sched->addTask(task, patches, materials);
}

void SPME::scheduleTransformFourierToReal(SchedulerP& sched,
                                          const ProcessorGroup* pg,
                                          const PatchSet* patches,
                                          const MaterialSet* materials,
                                          DataWarehouse* subOldDW,
                                          DataWarehouse* subNewDW,
                                          const LevelP& level)
{
  printSchedule(patches, spme_cout, "SPME::scheduleTransformFourierToReal");

  Task* task = scinew Task("SPME::transformFourierToReal", this, &SPME::transformFourierToReal);
  task->setType(Task::OncePerProc);
  task->usesMPI(true);

  task->requires(Task::NewDW, d_lb->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
  task->requires(Task::OldDW, d_lb->forwardTransformPlanLabel);
  task->requires(Task::OldDW, d_lb->backwardTransformPlanLabel);

  task->modifies(d_lb->subSchedulerDependencyLabel);
  task->computes(d_lb->backwardTransformPlanLabel);

  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perproc_patches =  loadBal->getPerProcessorPatchSet(level);

  sched->addTask(task, perproc_patches, materials);
}

void SPME::scheduleDistributeNodeLocalQ(SchedulerP& sched,
                                        const ProcessorGroup* pg,
                                        const PatchSet* patches,
                                        const MaterialSet* materials,
                                        DataWarehouse* subOldDW,
                                        DataWarehouse* subNewDW)
{
  printSchedule(patches, spme_cout, "SPME::scheduleDistributeNodeLocalQ");

  Task* task = scinew Task("SPME::distributeNodeLocalQ-force", this, &SPME::distributeNodeLocalQ);

  task->requires(Task::NewDW, d_lb->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
  task->modifies(d_lb->subSchedulerDependencyLabel);

  sched->addTask(task, patches, materials);
}

void SPME::calculatePreTransform(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* materials,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  size_t numPatches = patches->size();
  size_t numLocalAtomTypes = materials->size();  // Right now we're storing atoms as a material

  for (size_t p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);
    // Extract current spmePatch
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

    // spmePatches map to patches 1:1; shouldn't need to lock for anything that's local to a patch
    SimpleGrid<dblcomplex>* Q_threadLocal = currentSPMEPatch->getQ();
    // Initialize once before mapping any type of atoms
    Q_threadLocal->initialize(0.0);
    for (size_t localAtomTypeIndex = 0; localAtomTypeIndex < numLocalAtomTypes; ++localAtomTypeIndex) {
      int globalAtomType = materials->get(localAtomTypeIndex);

      ParticleSubset* pset = old_dw->getParticleSubset(globalAtomType, patch);
      constParticleVariable<Point> px;
      old_dw->get(px, d_lb->pXLabel, pset);

      // When we have a material iterator in here, we should store/get charge by material.
      // Charge represents the static charge on a particle, which is set by particle type.
      // No need to store one for each particle. -- JBH
      // double globalAtomCharge = materials->getProperty(charge)  //???

      constParticleVariable<double> pcharge;
      constParticleVariable<long64> pids;
      CCVariable<int> dependency;
      old_dw->get(pcharge, d_lb->pChargeLabel, pset);
      old_dw->get(pids, d_lb->pParticleIDLabel, pset);
      new_dw->allocateAndPut(dependency, d_lb->subSchedulerDependencyLabel, globalAtomType, patch, Ghost::None, 0);


      // Verify the charge map can contain the necessary data and get it
      currentSPMEPatch->verifyChargeMapAllocation(pset->numParticles(),globalAtomType);
      std::vector<SPMEMapPoint>* gridMap = currentSPMEPatch->getChargeMap(globalAtomType);

      // and generate the charge map
      SPME::generateChargeMap(gridMap, pset, px, pids);
      SPME::mapChargeToGrid(currentSPMEPatch, gridMap, pset, pcharge);

    }  // end Atom Type Loop
  }  // end Patch loop

  // TODO keep an eye on this to make sure it works like we think it should
  if (Thread::self()->myid() == 0) {
    d_Q_nodeLocal->initialize(dblcomplex(0.0, 0.0));
  }

  // these need to be transfered forward each timestep so they can ultimately be passed back to the parent DW
  bool replace = true;
  new_dw->transferFrom(old_dw, d_lb->pXLabel, patches, materials, replace);
  new_dw->transferFrom(old_dw, d_lb->pChargeLabel, patches, materials, replace);
  new_dw->transferFrom(old_dw, d_lb->pParticleIDLabel, patches, materials, replace);
}

void SPME::reduceNodeLocalQ(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* materials,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{
  size_t numPatches = patches->size();

  for (size_t p = 0; p < numPatches; ++p) {

    const Patch* patch = patches->get(p);
    // Extract current spmePatch
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

    SimpleGrid<dblcomplex>* Q_patchLocal = currentSPMEPatch->getQ();
    IntVector localQOffset = Q_patchLocal->getOffset();  // Location of the local Q patches 0,0,0 origin
    IntVector localQExtent = Q_patchLocal->getExtentWithGhost(); // Size of the current local Q subset (with ghosts)

    IntVector globalBoundaries = d_Q_nodeLocal->getExtents();

    int xBase = localQOffset[0];
    int yBase = localQOffset[1];
    int zBase = localQOffset[2];

    int xExtent = localQExtent[0];
    int yExtent = localQExtent[1];
    int zExtent = localQExtent[2];

    int xMax = globalBoundaries[0];
    int yMax = globalBoundaries[1];
    int zMax = globalBoundaries[2];

    d_Qlock.lock();
    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int x_local = xmask;
      int x_global = xBase + xmask;
      if (x_global >= xMax) { x_global -= xMax; }
      if (x_global < 0)     { x_global += xMax; }
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_local = ymask;
        int y_global = yBase + ymask;
        if (y_global >= yMax) { y_global -= yMax; }
        if (y_global < 0)     { y_global += yMax; }
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_local = zmask;
          int z_global = zBase + zmask;
          if (z_global >= zMax) { z_global -= zMax; }
          if (z_global < 0)     { z_global += zMax; }

          // Recall d_Q_nodeLocal is a complete copy of the Q grid for reduction across MPI threads
          (*d_Q_nodeLocal)(x_global,y_global,z_global) += (*Q_patchLocal)(x_local,y_local,z_local);

        }
      }
    }
    d_Qlock.unlock();
  }
}

void SPME::transformRealToFourier(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* materials,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  // Do the homebrew MPI_Allreduce on our local Q-grid before we do the forward FFT
  int totalElements = d_kLimits(0) * d_kLimits(1) * d_kLimits(2);
  dblcomplex* sendbuf = d_Q_nodeLocal->getDataPtr();
  dblcomplex* recvbuf = d_Q_nodeLocalScratch->getDataPtr();
  MPI_Allreduce(sendbuf, recvbuf, totalElements, MPI_DOUBLE_COMPLEX, MPI_SUM, pg->getComm());

  SoleVariable<fftw_plan> forwardTransformPlan;
  old_dw->get(forwardTransformPlan, d_lb->forwardTransformPlanLabel);
  fftw_plan forwardPlan = forwardTransformPlan.get();

  // setup and copy data to-and-from this processor's portion of the global FFT array
  fftw_complex* localChunk = d_localFFTData.complexData;
  ptrdiff_t localN = d_localFFTData.numElements; // a (local_n * kLimits.y * kLimits.z) chunk of the global array
  ptrdiff_t localStart = d_localFFTData.startAddress;
  dblcomplex* nodeLocalData = recvbuf + localStart;
  size_t numElements = localN * d_kLimits[1] * d_kLimits[2];

  std::memcpy(localChunk, nodeLocalData, numElements * sizeof(dblcomplex));
  fftw_execute(forwardPlan);
  std::memcpy(nodeLocalData, localChunk, numElements * sizeof(dblcomplex));

  // carry forward plan for forward FFT
  SoleVariable<fftw_plan> global_forwardTransformPlan;
  old_dw->get(global_forwardTransformPlan, d_lb->forwardTransformPlanLabel);
  new_dw->put(global_forwardTransformPlan, d_lb->forwardTransformPlanLabel);
}

void SPME::calculateInFourierSpace(const ProcessorGroup* pg,
                                   const PatchSubset* patches,
                                   const MaterialSubset* materials,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  double spmeFourierEnergy = 0.0;
  Matrix3 spmeFourierStress(0.0);

  size_t numPatches = patches->size();
//  size_t numLocalAtomTypes = materials->size();

  for (size_t p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;
    IntVector localExtents      = currentSPMEPatch->getLocalExtents();

//    for (size_t localAtomTypeIndex = 0; localAtomTypeIndex < numLocalAtomTypes; ++localAtomTypeIndex) {
//      int globalAtomTypeIndex = materials->get(localAtomTypeIndex);

    SimpleGrid<double>*                      fTheta = currentSPMEPatch->getTheta();
    SimpleGrid<Matrix3>*            stressPrefactor = currentSPMEPatch->getStressPrefactor();

    // Multiply the transformed Q by B*C to get Theta
// Stubbing out interface to swap dimensions for FFT efficiency
//    int systemMaxKDimension = d_system->getMaxKDimensionIndex();
//    int systemMidKDimension = d_system->getMidDimensionIndex();
//    int systemMinKDimension = d_system->getMinDimensionIndex();
    int systemMaxKDimension = 0; // X for now
    int systemMidKDimension = 1; // Y for now
    int systemMinKDimension = 2; // Z for now

    size_t xMax = localExtents[systemMaxKDimension];
    size_t yMax = localExtents[systemMidKDimension];
    size_t zMax = localExtents[systemMinKDimension];

    // local-to-global Q coordinate translation. This eliminates the copy to-and-from local SPMEPatches
    IntVector localQOffset = currentSPMEPatch->getQ()->getOffset();  // Location of the local Q patches 0,0,0 origin
    int xBase = localQOffset[0];
    int yBase = localQOffset[1];
    int zBase = localQOffset[2];

    for (size_t kX = 0; kX < xMax; ++kX) {
      for (size_t kY = 0; kY < yMax; ++kY) {
        for (size_t kZ = 0; kZ < zMax; ++kZ) {
          int x_global = xBase + kX;
          int y_global = yBase + kY;
          int z_global = zBase + kZ;

          std::complex<double> gridValue = (*d_Q_nodeLocalScratch)(x_global, y_global, z_global);
          // Calculate (Q*Q^)*(B*C)
          (*d_Q_nodeLocalScratch)(x_global, y_global, z_global) *= (*fTheta)(kX, kY, kZ);
          spmeFourierEnergy += std::abs((*d_Q_nodeLocalScratch)(x_global, y_global, z_global) * conj(gridValue));
          spmeFourierStress += std::abs((*d_Q_nodeLocalScratch)(x_global, y_global, z_global) * conj(gridValue)) * (*stressPrefactor)(kX, kY, kZ);
        }
      }
    }
//    }  // end AtomType loop
  }  // end SPME Patch loop

  // put updated values for reduction variables into the DW
  new_dw->put(sum_vartype(0.5 * spmeFourierEnergy), d_lb->spmeFourierEnergyLabel);
  new_dw->put(matrix_sum(0.5 * spmeFourierStress), d_lb->spmeFourierStressLabel);

}

void SPME::transformFourierToReal(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* materials,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  // Do the homebrew MPI_Allreduce on our local Q-scratch-grid before we do the reverse FFT
  int totalElements = d_kLimits(0) * d_kLimits(1) * d_kLimits(2);
  dblcomplex* sendbuf = d_Q_nodeLocalScratch->getDataPtr();
  dblcomplex* recvbuf = d_Q_nodeLocal->getDataPtr();
  MPI_Allreduce(sendbuf, recvbuf, totalElements, MPI_DOUBLE_COMPLEX, MPI_SUM, pg->getComm());

  SoleVariable<fftw_plan> backwardTransformPlan;
  old_dw->get(backwardTransformPlan, d_lb->backwardTransformPlanLabel);
  fftw_plan backwardPlan = backwardTransformPlan.get();

  // setup and copy data to-and-from this processor's portion of the global FFT array
  fftw_complex* localChunk = d_localFFTData.complexData;
  ptrdiff_t localN = d_localFFTData.numElements;
  ptrdiff_t localStart = d_localFFTData.startAddress;
  dblcomplex* nodeLocalData = recvbuf + localStart;
  size_t numElements = localN * d_kLimits[1] * d_kLimits[2];

  std::memcpy(localChunk, nodeLocalData, numElements * sizeof(dblcomplex));
  fftw_execute(backwardPlan);
  std::memcpy(nodeLocalData, localChunk, numElements * sizeof(dblcomplex));

  // carry forward plan for backward FFT
  SoleVariable<fftw_plan> global_backwardTransformPlan;
  old_dw->get(global_backwardTransformPlan, d_lb->backwardTransformPlanLabel);
  new_dw->put(global_backwardTransformPlan, d_lb->backwardTransformPlanLabel);
}

//----------------------------------------------------------------------------
// Setup related routines

void SPME::calculateBGrid(SimpleGrid<double>& BGrid,
                          const IntVector& localExtents,
                          const IntVector& globalOffset) const
{
  size_t limit_Kx = d_kLimits.x();
  size_t limit_Ky = d_kLimits.y();
  size_t limit_Kz = d_kLimits.z();

  std::vector<double> mf1(limit_Kx);
  std::vector<double> mf2(limit_Ky);
  std::vector<double> mf3(limit_Kz);
  SPME::generateMFractionalVector(mf1, limit_Kx);
  SPME::generateMFractionalVector(mf2, limit_Ky);
  SPME::generateMFractionalVector(mf3, limit_Kz);

  size_t xExtents = localExtents.x();
  size_t yExtents = localExtents.y();
  size_t zExtents = localExtents.z();

  // localExtents is without ghost grid points
  std::vector<dblcomplex> b1(xExtents);
  std::vector<dblcomplex> b2(yExtents);
  std::vector<dblcomplex> b3(zExtents);
  //  generateBVector(b1, mf1, globalOffset.x(), localExtents.x());
  //  generateBVector(b2, mf2, globalOffset.y(), localExtents.y());
  //  generateBVector(b3, mf3, globalOffset.z(), localExtents.z());
  SPME::generateBVectorChunk(b1, globalOffset[0], localExtents[0], d_kLimits[0]);
  SPME::generateBVectorChunk(b2, globalOffset[1], localExtents[1], d_kLimits[1]);
  SPME::generateBVectorChunk(b3, globalOffset[2], localExtents[2], d_kLimits[2]);

  for (size_t kX = 0; kX < xExtents; ++kX) {
    for (size_t kY = 0; kY < yExtents; ++kY) {
      for (size_t kZ = 0; kZ < zExtents; ++kZ) {
        BGrid(kX, kY, kZ) = norm(b1[kX]) * norm(b2[kY]) * norm(b3[kZ]);
      }
    }
  }
}

void SPME::generateBVector(std::vector<dblcomplex>& bVector,
                           const std::vector<double>& mFractional,
                           const int initialIndex,
                           const int localGridExtent) const
{
  double PI = acos(-1.0);
  double twoPI = 2.0 * PI;
  int n = d_interpolatingSpline.getOrder();

  std::vector<double> zeroAlignedSpline = d_interpolatingSpline.evaluateGridAligned(0);
  size_t endingIndex = initialIndex + localGridExtent;

  // Formula 4.4 in Essman et al.: A smooth particle mesh Ewald method
  for (size_t BIndex = initialIndex; BIndex < endingIndex; ++BIndex) {
    double twoPi_m_over_K = twoPI * mFractional[BIndex];
    double numerator_term = static_cast<double>(n - 1) * twoPi_m_over_K;
    dblcomplex numerator = dblcomplex(cos(numerator_term), sin(numerator_term));
    dblcomplex denominator = 0.0;
    for (int denomIndex = 0; denomIndex <= n - 2; ++denomIndex) {
      double denom_term = static_cast<double>(denomIndex) * twoPi_m_over_K;
      denominator += zeroAlignedSpline[denomIndex + 1] * dblcomplex(cos(denom_term), sin(denom_term));
    }
    bVector[BIndex] = numerator / denominator;
  }
}

void SPME::generateBVectorChunk(std::vector<dblcomplex>& bVector,
                                const int m_initial,
                                const int localGridExtent,
                                const int K) const
{
  double PI = acos(-1.0);
  double twoPI = 2.0 * PI;
  int n = d_interpolatingSpline.getOrder();
  double KReal = static_cast<double>(K);
  double OrderMinus1 = static_cast<double>(n - 1);

  std::vector<double> zeroAlignedSpline = d_interpolatingSpline.evaluateGridAligned(0);

  // Formula 4.4 in Essman et. al.: A Smooth Particle Mesh Ewald Method
  for (int BIndex = 0; BIndex < localGridExtent; ++BIndex) {
    double m = static_cast<double>(m_initial + BIndex);
    double twoPI_m_over_K = twoPI * m / KReal;  // --> 2*pi*m/K
    double numerator_scalar = twoPI_m_over_K * OrderMinus1;  // --> (n-1)*2*pi*m/K
    dblcomplex numerator = dblcomplex(cos(numerator_scalar), sin(numerator_scalar));
    dblcomplex denominator(0.0, 0.0);
    for (int denomIndex = 0; denomIndex <= (n - 2); ++denomIndex) {
      double denom_term = static_cast<double>(denomIndex) * twoPI_m_over_K;
      denominator += zeroAlignedSpline[denomIndex + 1] * dblcomplex(cos(denom_term), sin(denom_term));
    }
    bVector[BIndex] = numerator / denominator;
  }
}

void SPME::calculateCGrid(SimpleGrid<double>& CGrid,
                          const IntVector& extents,
                          const IntVector& offset) const
{
  if (spme_dbg.active()) {
    // sanity check
    proc0thread0cout << "System Volume: " << d_systemVolume << endl;
  }

  std::vector<double> mp1(d_kLimits.x());
  std::vector<double> mp2(d_kLimits.y());
  std::vector<double> mp3(d_kLimits.z());

  SPME::generateMPrimeVector(mp1, d_kLimits.x());
  SPME::generateMPrimeVector(mp2, d_kLimits.y());
  SPME::generateMPrimeVector(mp3, d_kLimits.z());

  double PI = acos(-1.0);
  double PI2 = PI * PI;
  double invBeta2 = 1.0 / (d_ewaldBeta * d_ewaldBeta);
  double invVolFactor = 1.0 / (d_systemVolume * PI);

  int xOffset = offset.x();
  int yOffset = offset.y();
  int zOffset = offset.z();

  size_t xExtents = extents.x();
  size_t yExtents = extents.y();
  size_t zExtents = extents.z();

  for (size_t kX = 0; kX < xExtents; ++kX) {
    for (size_t kY = 0; kY < yExtents; ++kY) {
      for (size_t kZ = 0; kZ < zExtents; ++kZ) {
        if (kX != 0 || kY != 0 || kZ != 0) {
          Vector m(mp1[kX + xOffset], mp2[kY + yOffset], mp3[kZ + zOffset]);
          m = m * d_inverseUnitCell;
          double M2 = m.length2();
          double factor = PI2 * M2 * invBeta2;
          CGrid(kX, kY, kZ) = invVolFactor * exp(-factor) / M2;
        }
      }
    }
  }
  CGrid(0, 0, 0) = 0;
}

void SPME::calculateStressPrefactor(SimpleGrid<Matrix3>* stressPrefactor,
                                    const IntVector& extents,
                                    const IntVector& offset)
{
  std::vector<double> mp1(d_kLimits.x());
  std::vector<double> mp2(d_kLimits.y());
  std::vector<double> mp3(d_kLimits.z());

  SPME::generateMPrimeVector(mp1, d_kLimits.x());
  SPME::generateMPrimeVector(mp2, d_kLimits.y());
  SPME::generateMPrimeVector(mp3, d_kLimits.z());

  double PI = acos(-1.0);
  double PI2 = PI * PI;
  double invBeta2 = 1.0 / (d_ewaldBeta * d_ewaldBeta);

  int xOffset = offset.x();
  int yOffset = offset.y();
  int zOffset = offset.z();

  size_t xExtents = extents.x();
  size_t yExtents = extents.y();
  size_t zExtents = extents.z();

  for (size_t kX = 0; kX < xExtents; ++kX) {
    for (size_t kY = 0; kY < yExtents; ++kY) {
      for (size_t kZ = 0; kZ < zExtents; ++kZ) {
        if (kX != 0 || kY != 0 || kZ != 0) {
          Vector m(mp1[kX + xOffset], mp2[kY + yOffset], mp3[kZ + zOffset]);
          m = m * d_inverseUnitCell;
          double M2 = m.length2();
          Matrix3 localStressContribution(-2.0 * (1.0 + PI2 * M2 * invBeta2) / M2);

          // Multiply by fourier vectorial contribution
          /*
           * for (size_t s1 = 0; s1 < 3; ++s1)
           *   for (size_t s2 = 0; s2 < 3; ++s2)
           *     localStressContribution(s1, s2) *= (m[s1] * m[s2]);
           */
          localStressContribution(0, 0) *= (m[0] * m[0]);
          localStressContribution(0, 1) *= (m[0] * m[1]);
          localStressContribution(0, 2) *= (m[0] * m[2]);
          localStressContribution(1, 0) *= (m[1] * m[0]);
          localStressContribution(1, 1) *= (m[1] * m[1]);
          localStressContribution(1, 2) *= (m[1] * m[2]);
          localStressContribution(2, 0) *= (m[2] * m[0]);
          localStressContribution(2, 1) *= (m[2] * m[1]);
          localStressContribution(2, 2) *= (m[2] * m[2]);

          // Account for delta function
          /*
           * for (size_t delta = 0; delta < 3; ++delta)
           *   localStressContribution(delta, delta) += 1.0;
           */
          localStressContribution(0, 0) += 1.0;
          localStressContribution(1, 1) += 1.0;
          localStressContribution(2, 2) += 1.0;


          (*stressPrefactor)(kX, kY, kZ) = localStressContribution;
        }
      }
    }
  }
  (*stressPrefactor)(0, 0, 0) = Matrix3(0.0);
}

void SPME::generateChargeMap(std::vector<SPMEMapPoint>* chargeMap,
                             ParticleSubset* pset,
                             constParticleVariable<Point>& particlePositions,
                             constParticleVariable<long64>& particleIDs)
{
  // Loop through particles
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter) {
    particleIndex pidx = *iter;

    SCIRun::Point position = particlePositions[pidx];
    particleId pid = particleIDs[pidx];
    SCIRun::Vector particleGridCoordinates;

    //Calculate reduced coordinates of point to recast into charge grid
    particleGridCoordinates = (position.asVector()) * d_inverseUnitCell;
    // ** NOTE: JBH --> We may want to do this with a bit more thought eventually, since multiplying by the InverseUnitCell
    //                  is expensive if the system is orthorhombic, however it's not clear it's more expensive than dropping
    //                  to call MDSystem->isOrthorhombic() and then branching the if statement appropriately.

    SCIRun::Vector kReal = d_kLimits.asVector();
    particleGridCoordinates *= kReal;
    SCIRun::IntVector particleGridOffset(particleGridCoordinates.asPoint());
    SCIRun::Vector splineValues = particleGridOffset.asVector() - particleGridCoordinates;

    std::vector<double> xSplineArray = d_interpolatingSpline.evaluateGridAligned(splineValues.x());
    std::vector<double> ySplineArray = d_interpolatingSpline.evaluateGridAligned(splineValues.y());
    std::vector<double> zSplineArray = d_interpolatingSpline.evaluateGridAligned(splineValues.z());

    std::vector<double> xSplineDeriv = d_interpolatingSpline.derivativeGridAligned(splineValues.x());
    std::vector<double> ySplineDeriv = d_interpolatingSpline.derivativeGridAligned(splineValues.y());
    std::vector<double> zSplineDeriv = d_interpolatingSpline.derivativeGridAligned(splineValues.z());

    SCIRun::IntVector extents(xSplineArray.size(), ySplineArray.size(), zSplineArray.size());

    SimpleGrid<double> chargeGrid(extents, particleGridOffset, IV_ZERO, 0);
    SimpleGrid<SCIRun::Vector> forceGrid(extents, particleGridOffset, IV_ZERO, 0);

    size_t XExtent = xSplineArray.size();
    size_t YExtent = ySplineArray.size();
    size_t ZExtent = zSplineArray.size();

    for (size_t xidx = 0; xidx < XExtent; ++xidx) {
      double dampX = xSplineArray[xidx];
      for (size_t yidx = 0; yidx < YExtent; ++yidx) {
        double dampY = ySplineArray[yidx];
        double dampXY = dampX * dampY;
        for (size_t zidx = 0; zidx < ZExtent; ++zidx) {
          double dampZ = zSplineArray[zidx];
          double dampYZ = dampY * dampZ;
          double dampXZ = dampX * dampZ;

          chargeGrid(xidx, yidx, zidx) = dampXY * dampZ;
          forceGrid(xidx, yidx, zidx) = Vector(dampYZ * xSplineDeriv[xidx] * kReal.x(), dampXZ * ySplineDeriv[yidx] * kReal.y(),
                                               dampXY * zSplineDeriv[zidx] * kReal.z());

        }
      }
    }

    // TODO -> Look at building these in place in the chargeMap to save time.
    SPMEMapPoint currentMapPoint(pid, particleGridOffset, chargeGrid, forceGrid);
    chargeMap->push_back(currentMapPoint);
  }
}

void SPME::mapChargeToGrid(SPMEPatch* spmePatch,
                           const std::vector<SPMEMapPoint>* gridMap,
                           ParticleSubset* pset,
                           constParticleVariable<double>& charges)
{
  // grab local Q grid
  SimpleGrid<dblcomplex>* Q_patchLocal = spmePatch->getQ();
  IntVector patchOffset = spmePatch->getGlobalOffset();
//  IntVector patchExtent = spmePatch->getLocalExtents();
  IntVector patchExtent = Q_patchLocal->getExtentWithGhost();
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter) {
    particleIndex pidx = *iter;

    const SimpleGrid<double> chargeMap = (*gridMap)[pidx].getChargeGrid();
    double charge = charges[pidx];

    IntVector       QAnchor = chargeMap.getOffset();  // Location of the 0,0,0 origin for the charge map grid
    IntVector supportExtent = chargeMap.getExtents();  // Extents of the charge map grid

    IntVector Base=QAnchor-patchOffset;

    int x_Base = Base[0];
    int y_Base = Base[1];
    int z_Base = Base[2];
    int xExtent = supportExtent[0];
    int yExtent = supportExtent[1];
    int zExtent = supportExtent[2];


    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int x_anchor = x_Base + xmask;
      if (x_anchor > patchExtent.x()) {
        std::cerr << " Error:  x_anchor exceeds patch Extent in mapChargeToGrid"
                  << " xBase: " << x_Base << " xMask: " << xmask << " xAnchor: " << x_anchor
                  << " xPatchExtent: " << patchExtent.x() << endl;
      }
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_anchor = y_Base + ymask;
        if (y_anchor > patchExtent.y()) {
          std::cerr << " Error:  y_anchor exceeds patch Extent in mapChargeToGrid"
                    << " yBase: " << y_Base << " yMask: " << ymask << " yAnchor: " << y_anchor
                    << " yPatchExtent: " << patchExtent.y() << endl;
        }
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_anchor = z_Base + zmask;
          if (z_anchor > patchExtent.z()) {
            std::cerr << " Error:  z_anchor exceeds patch Extent in mapChargeToGrid"
                      << " zBase: " << z_Base << " zMask: " << zmask << " zAnchor: " << z_anchor
                      << " zPatchExtent: " << patchExtent.z() << endl;
          }
          // Local patch has no wrapping, we have ghost cells to write into
          dblcomplex val = charge * chargeMap(xmask, ymask, zmask);
          (*Q_patchLocal)(x_anchor, y_anchor, z_anchor) += val;
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
// Post transform calculation related routines

void SPME::calculatePostTransform(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* materials,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  size_t numPatches = patches->size();
  size_t numLocalAtomTypes = materials->size();
  for (size_t p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

    for (size_t localAtomTypeIndex = 0; localAtomTypeIndex < numLocalAtomTypes; ++localAtomTypeIndex) {
      int globalAtomType = materials->get(localAtomTypeIndex);

      ParticleSubset* pset = old_dw->getParticleSubset(globalAtomType, patch);

      constParticleVariable<double> pcharge;
      ParticleVariable<Vector> pforcenew;
      old_dw->get(pcharge, d_lb->pChargeLabel, pset);
      new_dw->allocateAndPut(pforcenew, d_lb->pElectrostaticsForceLabel_preReloc, pset);

      std::vector<SPMEMapPoint>* gridMap = currentSPMEPatch->getChargeMap(globalAtomType);

      // Calculate electrostatic contribution to f_ij(r)
      SPME::mapForceFromGrid(currentSPMEPatch, gridMap, pset, pcharge, pforcenew);

      ParticleVariable<double> pchargenew;
      new_dw->allocateAndPut(pchargenew, d_lb->pChargeLabel_preReloc, pset);
      // carry these values over for now
      pchargenew.copyData(pcharge);
    }
  }
}

void SPME::mapForceFromGrid(SPMEPatch* spmePatch,
                            const std::vector<SPMEMapPoint>* gridMap,
                            ParticleSubset* pset,
                            constParticleVariable<double>& charges,
                            ParticleVariable<Vector>& pforcenew)
{
  SimpleGrid<std::complex<double> >* Q_patchLocal = spmePatch->getQ();
  IntVector patchOffset = spmePatch->getGlobalOffset();
//  IntVector patchExtent = spmePatch->getLocalExtents();
  IntVector patchExtent = Q_patchLocal->getExtentWithGhost();
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter) {
    particleIndex pidx = *iter;

    SimpleGrid<SCIRun::Vector> forceMap = (*gridMap)[pidx].getForceGrid();
    double charge = charges[pidx];

    SCIRun::Vector newForce = Vector(0, 0, 0);
    IntVector       QAnchor = forceMap.getOffset();  // Location of the 0,0,0 origin for the force map grid
    IntVector supportExtent = forceMap.getExtents();  // Extents of the force map grid

    IntVector Base=QAnchor-patchOffset;

    int x_Base = Base[0];
    int y_Base = Base[1];
    int z_Base = Base[2];

    int xExtent = supportExtent[0];
    int yExtent = supportExtent[1];
    int zExtent = supportExtent[2];

    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int x_anchor = x_Base + xmask;
      if (x_anchor > patchExtent.x()) {
        std::cerr << " Error:  x_anchor exceeds patch Extent in mapForceFromGrid"
                  << " xBase: " << x_Base << " xMask: " << xmask << " xAnchor: " << x_anchor
                  << " xPatchExtent: " << patchExtent.x();
      }
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_anchor = y_Base + ymask;
        if (y_anchor > patchExtent.y()) {
          std::cerr << " Error:  y_anchor exceeds patch Extent in mapForceFromGrid"
                    << " yBase: " << y_Base << " yMask: " << ymask << " yAnchor: " << y_anchor
                    << " yPatchExtent: " << patchExtent.y();
        }
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_anchor = z_Base + zmask;
          if (z_anchor > patchExtent.z()) {
            std::cerr << " Error:  z_anchor exceeds patch Extent in mapForceFromGrid"
                      << " zBase: " << z_Base << " zMask: " << zmask << " zAnchor: " << z_anchor
                      << " zPatchExtent: " << patchExtent.z();
          }
        // Local grid should have appropriate ghost cells, so no wrapping necessary.
          double QReal = std::real((*Q_patchLocal)(x_anchor, y_anchor, z_anchor));
          newForce += forceMap(xmask, ymask, zmask) * QReal * charge * d_inverseUnitCell;
        }
      }
    }
    // sanity check
    if (spme_dbg.active()) {
      if (pidx < 5) {
        cerrLock.lock();
        std::cerr << " Force Check (" << pidx << "): " << newForce << endl;
        pforcenew[pidx] = newForce;
        cerrLock.unlock();
      }
    }
  }
}

//-----------------------------------------------------------------------------
// Routines to be used with polarizable implementation

void SPME::copyToNodeLocalQ(const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* materials,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  size_t numPatches = patches->size();

  for (size_t p=0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);
    //Extract current spmePatch
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

    SimpleGrid<dblcomplex>* Q_patchLocal = currentSPMEPatch->getQ();
    IntVector localQOffset = Q_patchLocal->getOffset(); // Location of the local Q patch's 0,0,0 origin
    IntVector localQExtent = Q_patchLocal->getExtents(); // No ghost cells here, because we only want what's internal to the patch to transfer

    //IntVector globalBoundaries = d_Q_nodeLocal->getExtents();

    int xBase = localQOffset[0];
    int yBase = localQOffset[1];
    int zBase = localQOffset[2];

    int xExtent = localQExtent[0];
    int yExtent = localQExtent[1];
    int zExtent = localQExtent[2];

    //int xMax = globalBoundaries[0];
    //int yMax = globalBoundaries[1];
    //int zMax = globalBoundaries[2];

    // We SHOULDN'T need to lock because we should never hit the same memory location with any two threads..
    // Wrapping shouldn't be needed since the patch spatial decomposition ensures no internals to a patch cross boundary conditions.
    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int x_local = xmask;
      int x_global = xBase + xmask;
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_local = ymask;
        int y_global = yBase + ymask;
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_local = zmask;
          int z_global = zBase + zmask;
          (*d_Q_nodeLocal)(x_global, y_global, z_global) = (*Q_patchLocal)(x_local, y_local, z_local);
        }
      }
    }
  }
}

void SPME::distributeNodeLocalQ(const ProcessorGroup* pg,
                                const PatchSubset* patches,
                                const MaterialSubset* materials,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  size_t numPatches = patches->size();

  for (size_t p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);

    // Extract current spmePatch
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

    SimpleGrid<dblcomplex>* Q_patchLocal = currentSPMEPatch->getQ();
    IntVector localQOffset = Q_patchLocal->getOffset();  // Location of the local Q patches 0,0,0 origin
    IntVector localQExtent = Q_patchLocal->getExtentWithGhost(); // Size of the current local Q subset (with ghosts)

    IntVector globalBoundaries = d_Q_nodeLocal->getExtents();

    int xBase = localQOffset[0];
    int yBase = localQOffset[1];
    int zBase = localQOffset[2];

    int xExtent = localQExtent[0];
    int yExtent = localQExtent[1];
    int zExtent = localQExtent[2];

    int xMax = globalBoundaries[0];
    int yMax = globalBoundaries[1];
    int zMax = globalBoundaries[2];

    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int x_local = xmask;
      int x_global = xBase + xmask;
      if (x_global >= xMax) { x_global -= xMax; }
      if (x_global < 0)     { x_global += xMax; }
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_local = ymask;
        int y_global = yBase + ymask;
        if (y_global >= yMax) { y_global -= yMax; }
        if (y_global < 0)     { y_global += yMax; }
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_local = zmask;
          int z_global = zBase + zmask;
          if (z_global >= zMax) { z_global -= zMax; }
          if (z_global < 0)     { z_global += zMax; }
          // Recall d_Q_nodeLocal is a complete copy of the Q grid for reduction across MPI threads
          (*Q_patchLocal)(x_local,y_local,z_local) = (*d_Q_nodeLocal)(x_global,y_global,z_global);
        }
      }
    }
  }
}

bool SPME::checkConvergence() const
{
  // Subroutine determines if polarizable component has converged
  if (!d_polarizable) {
    return true;
  } else {
    // throw an exception for now, but eventually will check convergence here.
    throw InternalError("Error: Polarizable force field not yet implemented!", __FILE__, __LINE__);
  }

  // TODO keep an eye on this to make sure it works like we think it should
  if (Thread::self()->myid() == 0) {
    d_Q_nodeLocal->initialize(dblcomplex(0.0, 0.0));
  }
}

