/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include <CCA/Components/MD/Electrostatics/SPME-busted/ShiftedCardinalBSpline.h>
#include <CCA/Components/MD/Electrostatics/SPME-busted/SPME.h>
#include <CCA/Components/MD/Electrostatics/SPME-busted/SPMEMapPoint.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Parallel/Parallel.h>

#include <Core/Thread/Thread.h>

#include <Core/Grid/Box.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>

#include <Core/Math/MiscMath.h>

#include <Core/Util/DebugStream.h>

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

#include <sci_values.h>
#include <sci_defs/fftw_defs.h>

#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/SimpleGrid.h>


#ifdef DEBUG
#include <Core/Util/FancyAssert.h>
#endif

#define IV_ZERO IntVector(0,0,0)

using namespace OldSPME;

// This file implements the core of the SPME implementation of the electrostatics factory
//  Constructors, destructors, the four basic interface operation
//
// This file also contains the scheduling logic for the subscheduler in the calculate interface

const double SPME::d_dipoleMixRatio = 0.2;

SPME::SPME() :
    d_Qlock("node-local Q lock"), d_spmeLock("SPMEPatch data structures lock")
{

}

SPME::SPME(MDSystem* system,
           const double ewaldBeta,
           const double cutoffRadius,
           const bool isPolarizable,
           const double tolerance,
           const IntVector& kLimits,
           const int splineOrder,
           const int maxPolarizableIterations) :
      d_system(system),
      d_ewaldBeta(ewaldBeta),
      d_electrostaticRadius(cutoffRadius),
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

#ifdef HAVE_FFTW

  fftw_cleanup_threads();
  fftw_mpi_cleanup();
  fftw_cleanup();

#endif
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

  // Must initialize FFTW MPI and threads before FFTW_MPI calls are made
  fftw_init_threads();
  fftw_mpi_init();

  // This is the local portion of the global FFT array that will reside on each portion (slab decomposition)
  ptrdiff_t local_n, local_start;
  ptrdiff_t alloc_local = fftw_mpi_local_size_3d(xdim, ydim, zdim, pg->getComm(), &local_n, &local_start);
  d_localFFTData.complexData = fftw_alloc_complex(alloc_local);
  d_localFFTData.numElements = local_n;
  d_localFFTData.startAddress = local_start;

  // create the forward and reverse FFT MPI plans
  fftw_complex* complexData = d_localFFTData.complexData;
  fftw_plan_with_nthreads(Parallel::getNumThreads());
  d_forwardPlan = fftw_mpi_plan_dft_3d(xdim, ydim, zdim, complexData, complexData, pg->getComm(), FFTW_FORWARD, FFTW_MEASURE);
  d_backwardPlan = fftw_mpi_plan_dft_3d(xdim, ydim, zdim, complexData, complexData, pg->getComm(), FFTW_BACKWARD, FFTW_MEASURE);

  // Initially register sole and reduction variables in the DW (initial timestep)
  new_dw->put(sum_vartype(0.0), d_label->electrostatic->rElectrostaticInverseEnergy);
  new_dw->put(matrix_sum(0.0), d_label->electrostatic->rElectrostaticInverseStress);
//  new_dw->put(sum_vartype(0.0), d_label->electrostaticReciprocalEnergyLabel);
//  new_dw->put(matrix_sum(0.0), d_label->electrostaticReciprocalStressLabel);
  //!FIXME JBH
  // new_dw->put(vector_sum(0.0), d_label->spmeDipoleVectorLabel);

  SoleVariable<double> dependency;
  new_dw->put(dependency, d_label->electrostatic->dElectrostaticDependency);
//  new_dw->put(dependency, d_lb->electrostaticsDependencyLabel);

  // ------------------------------------------------------------------------
  // Allocate and map the SPME patches
  Vector kReal = d_kLimits.asVector();
  Vector systemCellExtent = (d_system->getCellExtent()).asVector();

  int splineSupport = d_interpolatingSpline.getSupport();
  IntVector plusGhostExtents(splineSupport, splineSupport, splineSupport);
  IntVector minusGhostExtents(0,0,0);

  size_t numPatches = patches->size();
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

    // A SimpleGrid<double> of C(m1,m2,m3)=(1/(PI*V))*exp(-PI^2*M^2/Beta^2)/M^2 !* C(0,0,0)==0
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
  old_dw->get(dependency, d_label->electrostatic->dElectrostaticDependency);
  new_dw->put(dependency, d_label->electrostatic->dElectrostaticDependency);
//  old_dw->get(dependency, d_label->electrostaticsDependencyLabel);
//  new_dw->put(dependency, d_label->electrostaticsDependencyLabel);
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
  subNewDW->transferFrom(parentOldDW, d_label->global->pX, perProcPatches, allMaterialsUnion);
//  subNewDW->transferFrom(parentOldDW, d_label->pXLabel, perProcPatches, allMaterialsUnion);
//  subNewDW->transferFrom(parentOldDW, d_lb->pChargeLabel, perProcPatches, allMaterialsUnion);
  subNewDW->transferFrom(parentOldDW, d_label->global->pID, perProcPatches, allMaterialsUnion);
//  subNewDW->transferFrom(parentOldDW, d_label->pParticleIDLabel, perProcPatches, allMaterialsUnion);

  // reduction variables
  //  sum_vartype spmeFourierEnergy;
  //  matrix_sum spmeFourierStress;
  //  parentOldDW->get(spmeFourierEnergy, d_lb->spmeFourierEnergyLabel);
  //  parentOldDW->get(spmeFourierStress, d_lb->spmeFourierStressLabel);
  //  subNewDW->put(spmeFourierEnergy, d_lb->spmeFourierEnergyLabel);
  //  subNewDW->put(spmeFourierStress, d_label->spmeFourierStressLabel);
  parentNewDW->put(sum_vartype(0.0), d_label->electrostatic->rElectrostaticInverseEnergy);
  parentNewDW->put(matrix_sum(0.0), d_label->electrostatic->rElectrostaticInverseStress);
//    parentNewDW->put(sum_vartype(0.0), d_label->electrostaticReciprocalEnergyLabel);
//    parentNewDW->put(matrix_sum(0.0), d_label->electrostaticReciprocalStressLabel);


  bool converged = false;
  int numIterations = 0;
//  bool recompileSubscheduler = true;

  //!FIXME Shouldn't the subscheduler taskgraph compilation go here?
  subscheduler->initialize(3,1);

  // Realspace portion of SPME
  scheduleCalculateRealspace(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);

  // Calc local Q grid chunks
  scheduleCalculatePreTransform(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);

  // Q grid aggregation
  scheduleReduceNodeLocalQ(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);

  //Forward Xform
  scheduleTransformRealToFourier(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW, level);

  // Energy, stress(0), Force prep
  scheduleCalculateInFourierSpace(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);

  //Reverse Xform
  scheduleTransformFourierToReal(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW, level);

  // Redistribute Q grid
  scheduleDistributeNodeLocalQ(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);

  if (d_polarizable) {
	  // Update field for new dipole prediction and stress tensor
	  scheduleUpdateFieldandStress(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);

	  // Update local dipole prediction
	  scheduleCalculateNewDipoles(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);
  }
  subscheduler->compile();

  // Begin iterative charge/dipole calculation loop
  while (!converged && (numIterations < d_maxPolarizableIterations)) {

//    // compile task graph (once)
//    if (recompileSubscheduler) {
//
//      subscheduler->initialize(3, 1);
//
//      // prep for the forward FFT
//      scheduleCalculatePreTransform(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);
//
//      // Q grid reductions for forward FFT
//      scheduleReduceNodeLocalQ(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);
//
//      // Forward transform
//      scheduleTransformRealToFourier(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW, level);
//
//      // Do Fourier space calculations on transformed data
//      scheduleCalculateInFourierSpace(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);
//
//      // Reverse transform
//      scheduleTransformFourierToReal(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW, level);
//
//      // Redistribute force grid
//      scheduleDistributeNodeLocalQ(subscheduler, pg, patches, allMaterials, subOldDW, subNewDW);
//
//      // compile task graph
//      subscheduler->compile();
//
//      // make sure the above only happens once
//      recompileSubscheduler = false;
//    }

    // Need to re-map subNewDW
    subNewDW = subscheduler->get_dw(3);

    // move subNewDW to subOldDW
    subscheduler->advanceDataWarehouse(grid);
//    subOldDW->setScrubbing(DataWarehouse::ScrubComplete);
    subNewDW->setScrubbing(DataWarehouse::ScrubNone);

    //  execute the tasks
    subscheduler->execute();

    converged = checkConvergence();
    numIterations++;
  }

  // Need to re-map subNewDW so we're "getting" from the right subDW when forwarding SoleVariables
  subNewDW = subscheduler->get_dw(3);

  /*
   * No ParticleVariable products from iterations (read only info from ParentOldDW) so don't need forward these
   * subNewDW --> parentNewDW, just need to forward the SoleVariables; parentOldDW --> parentNewDW, as the
   * computes statements specify in MD::scheduleElectrostaticsCalculate
   */

  // For polar simulations we actually do need to push the dipoles up to the parent, as they should be particle variables
  // !FIXME

  // Push Reduction Variables up to the parent DW
  sum_vartype spmeFourierEnergyNew;
  matrix_sum spmeFourierStressNew;
  subNewDW->get(spmeFourierEnergyNew, d_label->electrostatic->rElectrostaticInverseEnergy);
  subNewDW->get(spmeFourierStressNew, d_label->electrostatic->rElectrostaticInverseStress);
//  subNewDW->get(spmeFourierEnergyNew, d_label->electrostaticReciprocalEnergyLabel);
//  subNewDW->get(spmeFourierStressNew, d_label->electrostaticReciprocalStressLabel);
  parentNewDW->put(spmeFourierEnergyNew, d_label->electrostatic->rElectrostaticInverseEnergy);
  parentNewDW->put(spmeFourierStressNew, d_label->electrostatic->rElectrostaticInverseStress);
//  parentNewDW->put(spmeFourierEnergyNew, d_label->electrostaticReciprocalEnergyLabel);
//  parentNewDW->put(spmeFourierStressNew, d_label->electrostaticReciprocalStressLabel);

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

void SPME::scheduleUpdateFieldandStress(SchedulerP& sched,
                                        const ProcessorGroup* pg,
                                        const PatchSet* patches,
                                        const MaterialSet* materials,
                                        DataWarehouse* subOldDW,
                                        DataWarehouse* subNewDW)
{
  printSchedule(patches, spme_cout, "SPME::scheduleUpdateFieldandStress");

  Task* task = scinew Task("SPME::updateFieldAndStress", this, &SPME::dipoleUpdateFieldAndStress);

  // Requires the dipoles from the last iteration
  task->requires(Task::OldDW, d_label->electrostatic->pMu);
  task->requires(Task::OldDW, d_label->electrostatic->pE_electroInverse_preReloc);

  // Calculates the new inverse space field prediction and updates the inverse space stress tensor
  task->computes(d_label->electrostatic->pE_electroInverse_preReloc);
  task->modifies(d_label->electrostatic->rElectrostaticInverseStress);

}

void SPME::scheduleCalculateNewDipoles(SchedulerP& sched,
                                       const ProcessorGroup* pg,
                                       const PatchSet* patches,
                                       const MaterialSet* materials,
                                       DataWarehouse* subOldDW,
                                       DataWarehouse* subNewDW)
{
	printSchedule(patches, spme_cout, "SPME::scheduleCalculateNewDipoles");

	Task* task = scinew Task("SPME::calculateNewDipoles", this, &SPME::calculateNewDipoles);

	// Requires the updated field from both the realspace and reciprocal calculation
	// Also may want the dipoles from the previous iteration
	task->requires(Task::OldDW, d_label->electrostatic->pMu, Ghost::None, 0);
	task->requires(Task::NewDW, d_label->electrostatic->pE_electroReal_preReloc, Ghost::None, 0);
	task->requires(Task::NewDW, d_label->electrostatic->pE_electroInverse_preReloc, Ghost::None, 0);

	// Overwrites each dipole array at iteration n with the full estimate of dipole array at iteration n+1
	task->computes(d_label->electrostatic->pMu_preReloc);

	sched->addTask(task, patches, materials);


}

void SPME::scheduleCalculateRealspace(SchedulerP& sched,
                                      const ProcessorGroup* pg,
                                      const PatchSet* patches,
                                      const MaterialSet* materials,
                                      DataWarehouse* subOldDW,
                                      DataWarehouse* subNewDW)
{
	printSchedule(patches, spme_cout, "SPME::scheduleCalculateRealspace");

	Task* task = scinew Task("SPME::calculateRealspace", this, &SPME::calculateRealspace);

	int CUTOFF_RADIUS = d_system->getElectrostaticGhostCells();
	// Requires the location and ID of all atom positions, which don't change for the polarizability iteration
	task->requires(Task::ParentOldDW, d_label->global->pX, Ghost::AroundNodes, CUTOFF_RADIUS);
	task->requires(Task::ParentOldDW, d_label->global->pID, Ghost::AroundNodes, CUTOFF_RADIUS);
	// Also requires the last iteration's dipole guess, which does change for the polarizability iteration
    task->requires(Task::OldDW, d_label->electrostatic->pMu, Ghost::AroundNodes, CUTOFF_RADIUS);

    // Computes the realspace contribution to the electrostatic field, force, and stress tensor
    task->computes(d_label->electrostatic->pE_electroReal_preReloc);
    task->computes(d_label->electrostatic->rElectrostaticRealEnergy);
    task->computes(d_label->electrostatic->rElectrostaticRealStress);

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

  int CUTOFF_RADIUS = d_system->getElectrostaticGhostCells();

  // Setup requires the position and ID arrays from the parent process
  task->requires(Task::ParentOldDW, d_label->global->pX, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::ParentOldDW, d_label->global->pID, Ghost::AroundNodes, CUTOFF_RADIUS);

//  task->requires(Task::ParentNewDW, d_label->pXLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_lb->pChargeLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_label->pParticleIDLabel, Ghost::AroundNodes, CUTOFF_RADIUS);

  // Computes (copies from parent to local) position and ID, sets dependency flag
  task->computes(d_label->electrostatic->dSubschedulerDependency);
  task->computes(d_label->global->pX);
  task->computes(d_label->global->pID);

  // May also want to initialize things like field, force, etc.. here since it's outside the polarization loop
  // !FIXME

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

  // FIXME!  Is this redundant with the following "modifies"?
  task->requires(Task::NewDW, d_label->electrostatic->dSubschedulerDependency, Ghost::None, 0);
//  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
  task->modifies(d_label->electrostatic->dSubschedulerDependency);
//  task->modifies(d_label->subSchedulerDependencyLabel);

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

  task->requires(Task::NewDW, d_label->electrostatic->dSubschedulerDependency, Ghost::None, 0);

//  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);

  task->modifies(d_label->electrostatic->dSubschedulerDependency);
//  task->modifies(d_label->subSchedulerDependencyLabel);

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

  task->requires(Task::NewDW, d_label->electrostatic->dSubschedulerDependency, Ghost:: Ghost::None, 0);

//  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
  task->modifies(d_label->electrostatic->dSubschedulerDependency);
  task->computes(d_label->electrostatic->rElectrostaticInverseEnergy);
  task->computes(d_label->electrostatic->rElectrostaticInverseStress);
//  task->modifies(d_label->subSchedulerDependencyLabel);
//  task->computes(d_label->electrostaticReciprocalEnergyLabel);
//  task->computes(d_label->electrostaticReciprocalStressLabel);

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

  task->requires(Task::NewDW, d_label->electrostatic->dSubschedulerDependency, Ghost::None, 0);
//  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
  task->modifies(d_label->electrostatic->dSubschedulerDependency);
//  task->modifies(d_label->subSchedulerDependencyLabel);

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

  task->requires(Task::NewDW, d_label->electrostatic->dSubschedulerDependency, Ghost::None, 0);

  task->modifies(d_label->electrostatic->dSubschedulerDependency);
  //  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
//  task->modifies(d_label->subSchedulerDependencyLabel);

  sched->addTask(task, patches, materials);
}

void SPME::registerRequiredParticleStates(std::vector<const VarLabel*>& particleState,
                                          std::vector<const VarLabel*>& particleState_preReloc,
                                          MDLabel* d_label) const {

  // We absolutely need per-particle information to implement polarizable SPME
  if (d_polarizable) {
    particleState.push_back(d_label->electrostatic->pMu);
    particleState_preReloc.push_back(d_label->electrostatic->pMu_preReloc);
    particleState.push_back(d_label->electrostatic->pE_electroReal);
    particleState.push_back(d_label->electrostatic->pE_electroInverse);
    particleState_preReloc.push_back(d_label->electrostatic->pE_electroReal_preReloc);
    particleState_preReloc.push_back(d_label->electrostatic->pE_electroInverse_preReloc);
  }

  // We -probably- don't need relocatable Force information, however it may be the easiest way to
  //   implement the required per-particle Force information.
  particleState.push_back(d_label->electrostatic->pF_electroInverse);
  particleState.push_back(d_label->electrostatic->pF_electroReal);
  particleState_preReloc.push_back(d_label->electrostatic->pF_electroInverse_preReloc);
  particleState_preReloc.push_back(d_label->electrostatic->pF_electroReal_preReloc);

  // Note:  Per particle charges may be required in some FF implementations (i.e. ReaxFF), however we will let
  //        the FF themselves register these variables if these are present and needed.

}

