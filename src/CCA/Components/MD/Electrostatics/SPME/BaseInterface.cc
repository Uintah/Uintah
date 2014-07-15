/*
 *
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
 *
 * ----------------------------------------------------------
 * SPME_BaseInterface.cc
 *
 *  Created on: May 15, 2014
 *      Author: jbhooper
 */

#include <CCA/Ports/Scheduler.h>

#include <Core/Grid/Variables/SoleVariable.h>

#include <CCA/Components/MD/MDUtil.h>

#include <CCA/Components/MD/Electrostatics/SPME/SPME.h>


#ifdef DEBUG
#include <Core/Util/FancyAssert.h>
#endif

using namespace Uintah;
//-------1_________2---------3_________4---------5________6---------7_________8


const double SPME::d_dipoleMixRatio = 0.1;

SPME::SPME(const double             ewaldBeta,
           const double             cutoffRadius,
           const int                ghostCells,
           const SCIRun::IntVector  kLimits,
           const int                splineOrder,
           const bool               polarizable,
           const int                maxPolarizableIterations,
           const double             polarizationTolerance)
          :d_ewaldBeta(ewaldBeta),
           d_electrostaticRadius(cutoffRadius),
           d_electrostaticGhostCells(ghostCells),
           d_kLimits(kLimits),
           f_polarizable(polarizable),
           d_maxPolarizableIterations(maxPolarizableIterations),
           d_polarizationTolerance(polarizationTolerance),
           d_Qlock("node-local Q lock"),
           d_spmeLock("SPME shared data structure lock")
{
  d_interpolatingSpline = ShiftedCardinalBSpline(splineOrder);
  d_electrostaticMethod = Electrostatics::SPME;
}

SPME::~SPME() {

  // Clean up the dynamically allocated patch memory
  std::map<int, SPMEPatch*>::iterator SPMEPatchIterator;
  std::map<int, SPMEPatch*>::iterator start = d_spmePatchMap.begin();
  std::map<int, SPMEPatch*>::iterator end   = d_spmePatchMap.end();

  for (SPMEPatchIterator = start; SPMEPatchIterator != end; ++SPMEPatchIterator)
  {
    SPMEPatch* currSPMEPatch = SPMEPatchIterator->second;
    delete currSPMEPatch;
  }

  if (d_Q_nodeLocal) {
    delete d_Q_nodeLocal;
  }

  if (d_Q_nodeLocalScratch) {
    delete d_Q_nodeLocalScratch;
  }

  // This used to be wrapped in #ifdef HAVE_FFTW; however we can't do SPME if we don't anyway
  fftw_cleanup_threads();
  fftw_mpi_cleanup();
  fftw_cleanup();

}

// Interface implementations
void SPME::initialize(const ProcessorGroup*   pg,
                      const PatchSubset*      patches,
                      const MaterialSubset*   materials,
                      DataWarehouse*        /*oldDW*/,
                      DataWarehouse*          newDW,
                      const SimulationStateP* simState,
                      MDSystem*               systemInfo,
                      const MDLabel*          label,
                      CoordinateSystem*       coordSys) {
  // SPME::initialize is called from MD::initialize

  /* Initialize the local version of the global Q and Q_scratch arrays
   *   Rather than forcing reductions on processor for each patch, we have a
   *   single per-processor pool, where we handle thread access manually with
   *    locks/unlocks as appropriate
   */
  d_Q_nodeLocal         = scinew SimpleGrid<dblcomplex>(d_kLimits,
                                                        MDConstants::IV_ZERO,
                                                        MDConstants::IV_ZERO,
                                                        0);
  d_Q_nodeLocalScratch  = scinew SimpleGrid<dblcomplex>(d_kLimits,
                                                        MDConstants::IV_ZERO,
                                                        MDConstants::IV_ZERO,
                                                        0);

//---->>>> Setup FFTW related quantities
  /*
   * ptrdiff_t is a type able to represent the result of any valid pointer
   * subtraction operations.
   *
   * It is a standard C integer type which is (at least) 32 bits wide on a
   * 32-bit machine, and 64 bits wide on a 64-bit machine.
   */
  const ptrdiff_t xdim = d_kLimits(0);
  const ptrdiff_t ydim = d_kLimits(1);
  const ptrdiff_t zdim = d_kLimits(2);
  MPI_Comm SPMEComm = pg->getComm();

  // Initialize FFTW MPI and threads before making any FFTW_MPI calls
  fftw_init_threads();
  fftw_mpi_init();

  /*
   * Map and allocate the local portion of the global FFT array that will
   * reside on each processor (slab decomposition)
   */
  // Determine the size of the local memory to be allocated
  ptrdiff_t local_n, local_start;
  ptrdiff_t alloc_local = fftw_mpi_local_size_3d(xdim, ydim, zdim, SPMEComm,
                                                 &local_n, &local_start);

  // Allocate the local memory via FFTW and fill our fftw data structure
  d_localFFTData.complexData    = fftw_alloc_complex(alloc_local);
  d_localFFTData.numElements    = local_n;
  d_localFFTData.startAddress   = local_start;

  // Create the fftw forward and reverse transformation plans for threaded MPI
  fftw_complex* complexData = d_localFFTData.complexData;
  fftw_plan_with_nthreads(Parallel::getNumThreads());
  d_forwardPlan     = fftw_mpi_plan_dft_3d(xdim, ydim, zdim, complexData,
                                           complexData, SPMEComm,
                                           FFTW_FORWARD, FFTW_MEASURE);
  d_backwardPlan    = fftw_mpi_plan_dft_3d(xdim, ydim, zdim, complexData,
                                           complexData, SPMEComm,
                                           FFTW_BACKWARD, FFTW_MEASURE);
//  SoleVariable<double> dependency;
//  newDW->put(dependency, label->electrostatic->dElectrostaticDependency);

//---->>>> Allocate and map the SPME patches
  SCIRun::Vector    kReal               = d_kLimits.asVector();
  SCIRun::IntVector systemCellExtent    = coordSys->getCellExtent();
  double            totalCellInverse    = 1.0/ ( systemCellExtent.x() *
                                                 systemCellExtent.y() *
                                                 systemCellExtent.z() );
  int               splinePoints        = d_interpolatingSpline.getSupport();

  SCIRun::IntVector plusGhostExtents(splinePoints, splinePoints, splinePoints);
  SCIRun::IntVector minusGhostExtents(0, 0, 0);

  // Loop through the patches and set them up
  size_t            numPatches      =   patches->size();
  size_t            numAtomTypes    =   materials->size();



  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch*    patch           = patches->get(patchIndex);

    // Initialize reduction variables for this patch:
    newDW->put(sum_vartype(0.0),
               label->electrostatic->rElectrostaticInverseEnergy);
    newDW->put(sum_vartype(0.0),
               label->electrostatic->rElectrostaticRealEnergy);

    // We calculate stress even if we don't need it later; in most simulations
    // we'll want to track it and it's small additional expense to calculate.
    // We can add in a flag to gate calculation later if we really care.
    newDW->put(matrix_sum(MDConstants::M3_0),
               label->electrostatic->rElectrostaticInverseStress);
    newDW->put(matrix_sum(MDConstants::M3_0),
               label->electrostatic->rElectrostaticRealStress);

    if (f_polarizable) {
      newDW->put(matrix_sum(MDConstants::M3_0),
                 label->electrostatic->rElectrostaticInverseStressDipole);
    }

    // Quick material loop to initialize per-particle related variables
    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex) {
      int atomType = materials->get(typeIndex);
      ParticleSubset* atomSubset = newDW->getParticleSubset(atomType, patch);

      ParticleVariable<Vector> pF_real, pF_inverse;
      newDW->allocateAndPut(pF_real,
                            label->electrostatic->pF_electroReal,
                            atomSubset);
      newDW->allocateAndPut(pF_inverse,
                            label->electrostatic->pF_electroInverse,
                            atomSubset);

      particleIndex numAtoms = atomSubset->numParticles();
      for (particleIndex atom = 0; atom < numAtoms; ++atom)
      {
        pF_real[atom]       =   MDConstants::V_ZERO;
        pF_inverse[atom]    =   MDConstants::V_ZERO;
      }

      if (f_polarizable) {
        ParticleVariable<Vector>   pE_real, pE_inverse, pMu;
        newDW->allocateAndPut(pE_real,
                              label->electrostatic->pE_electroReal,
                              atomSubset);
        newDW->allocateAndPut(pE_inverse,
                              label->electrostatic->pE_electroInverse,
                              atomSubset);
        newDW->allocateAndPut(pMu,
                              label->electrostatic->pMu,
                              atomSubset);
        particleIndex numAtoms = atomSubset->numParticles();
        for (particleIndex atom = 0; atom < numAtoms; ++atom)
        {
          pE_real[atom]     =   MDConstants::V_ZERO;
          pE_inverse[atom]  =   MDConstants::V_ZERO;
          pMu[atom]         =   MDConstants::V_ZERO;
        }
      }
    }

    SCIRun::Vector  patchLowIndex   = (patch->getCellLowIndex()).asVector();
    SCIRun::Vector  patchHighIndex  = (patch->getCellHighIndex()).asVector();
    SCIRun::Vector  localCellExtent = patchHighIndex - patchLowIndex;

    // Determine the fraction of the overall system volume patch represents
    double  localCellVolumeFraction = localCellExtent.x() *
                                      localCellExtent.y() *
                                      localCellExtent.z() *
                                      totalCellInverse;

    // Determine the K-grid low and high boundaries which this patch maps into
    IntVector patchKLow, patchKHigh;
    for (size_t unit = 0; unit < 3; ++unit) {
      patchKLow[unit]   = floor(kReal[unit]*
                                (patchLowIndex[unit]/systemCellExtent[unit]));
      patchKHigh[unit]  = floor(kReal[unit]*
                                (patchHighIndex[unit]/systemCellExtent[unit]));
    }

    // Number of K grid points onto which the local patch maps
    SCIRun::IntVector patchKGridExtents = (patchKHigh - patchKLow);
    // Starting index for K grid of local patch
    SCIRun::IntVector patchKGridOffset  = patchKLow;

    // Create an SPMEPatch
    SPMEPatch* spmePatch = new SPMEPatch(patchKGridExtents, patchKGridOffset,
                                         plusGhostExtents, minusGhostExtents,
                                         patch, localCellVolumeFraction,
                                         splinePoints, systemInfo);

    // Write current SPMEPatch to the processor's patch map.
    d_spmeLock.writeLock();
    d_spmePatchMap.insert(SPMEPatchKey(patch->getID(),spmePatch));
    d_spmeLock.writeUnlock();
  }
}

void SPME::setup(       const ProcessorGroup*   pg,
                        const PatchSubset*      patches,
                        const MaterialSubset*   materials,
                        DataWarehouse*          oldDW,
                        DataWarehouse*          newDW,
                        const SimulationStateP*       /*simState*/,
                        MDSystem*               systemInfo,
                        const MDLabel*          label,
                        CoordinateSystem*       coordSys) {

  //Uintah::Matrix3 inverseUnitCell = coordSys->getInverseCell();

  if (coordSys->queryCellChanged()) { // Update SPME phase factors
    /*
     * Basis vectors/angles change, we need to recalculate all of the SPME
     * phase factor information.
     */
    size_t numPatches       = patches->size();
    size_t splineSupport    = d_interpolatingSpline.getSupport();

    IntVector   plusGhostExtents(splineSupport, splineSupport, splineSupport);
    IntVector   minusGhostExtents(0, 0, 0);

    for( size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
      const Patch* patch = patches->get(patchIndex);
      // Grab our current patch map
      SPMEPatch*            spmePatch       =
                                d_spmePatchMap.find(patch->getID())->second;

      // Grab the stress prefactor from our SPMEPatch
      SimpleGrid<Matrix3>*  stressPrefactor = spmePatch->getStressPrefactor();

      SCIRun::IntVector spmePatchExtents    = spmePatch->getLocalExtents();
      SCIRun::IntVector spmePatchOffset     = spmePatch->getGlobalOffset();

      SPME::calculateStressPrefactor(stressPrefactor, coordSys);

      // Theta is the composite phase factor
      SimpleGrid<double>*   fTheta          = spmePatch->getTheta();

      // No ghost cells; internal only
      SimpleGrid<double>    fBGrid(spmePatchExtents,
                                   spmePatchOffset,
                                   MDConstants::IV_ZERO,
                                   0);
      SimpleGrid<double>    fCGrid(spmePatchExtents,
                                   spmePatchOffset,
                                   MDConstants::IV_ZERO,
                                   0);
      // B(m1,m2,m3) = |b1(m1)|^2*|b2(m2)|^2*|b3(m3)|^2
      SPME::calculateBGrid(fBGrid);

      // C(m1,m2,m3) = (1/(Pi*V))*exp(-Pi^2*M^2/Beta^2)/M^2; C(0,0,0) == 0
      SPME::calculateCGrid(fCGrid, coordSys);

      // Composite B*C into Theta
      int systemMaxKDimension = 0; // X for now
      int systemMidKDimension = 1; // Y for now
      int systemMinKDimension = 2; // Z for now
      size_t x_extent = spmePatchExtents[systemMaxKDimension];
      size_t y_extent = spmePatchExtents[systemMidKDimension];
      size_t z_extent = spmePatchExtents[systemMinKDimension];
      for (size_t xIndex = 0; xIndex < x_extent; ++xIndex) {
        for (size_t yIndex = 0; yIndex < y_extent; ++yIndex) {
          for (size_t zIndex = 0; zIndex < z_extent; ++zIndex) {
            (*fTheta)(xIndex, yIndex, zIndex) = fBGrid(xIndex, yIndex, zIndex)*
                                                fCGrid(xIndex, yIndex, zIndex);
          } // zIndex
        } // yIndex
      } // xIndex
    } // patchIndex
  } // Update SPME phase factors

  SoleVariable<double> dependency;
  newDW->put(dependency, label->electrostatic->dElectrostaticDependency);

}

void SPME::calculate(   const ProcessorGroup*   pg,
                        const PatchSubset*      perProcPatches,
                        const MaterialSubset*   materials,
                        DataWarehouse*          parentOldDW,
                        DataWarehouse*          parentNewDW,
                        const SimulationStateP* simState,
                        MDSystem*               systemInfo,
                        const MDLabel*          label,
                        CoordinateSystem*       coordSys,
                        SchedulerP&             subscheduler,
                        const LevelP&           level)
{

  // Generate the spline coefficients for the loop calculation.  If we
  // incorporate the charge seperately we can reduce this step to only once
  // regardless of whether or not we are calculating induced dipoles.
  if (f_polarizable) { 	// generate dipole chargemap
    generateChargeMapDipole(pg, perProcPatches, materials,
                    parentOldDW, parentNewDW,
                    label, coordSys);
  }
  else                  // generate non-dipole chargemap
  {
    generateChargeMap(pg, perProcPatches, materials,
                      parentOldDW, parentNewDW,
                      label, coordSys);
  }


  //  Get the full material set
  const MaterialSet*    allMaterials        =   (*simState)->allMaterials();
  const MaterialSubset* allMaterialsUnion   =   allMaterials->getUnion();

  // Most of the calculate loop falls under the control of the subscheduler
  //  Temporarily turn off parentDW scrubbing
  DataWarehouse::ScrubMode parentOldDW_scrubmode =
                           parentOldDW->setScrubbing(DataWarehouse::ScrubNone);

  DataWarehouse::ScrubMode parentNewDW_scrubmode =
                           parentNewDW->setScrubbing(DataWarehouse::ScrubNone);

  GridP grid    =   level->getGrid();
  subscheduler->setParentDWs(parentOldDW, parentNewDW);
  subscheduler->advanceDataWarehouse(grid); // Generates the first subNewDW
  subscheduler->setInitTimestep(true);      // Necessary to populate the subNewDW

  DataWarehouse*        subOldDW            =   subscheduler->get_dw(2);
  DataWarehouse*        subNewDW            =   subscheduler->get_dw(3);

  if (f_polarizable) {  // Transfer last timestep's dipole to subDW
    subNewDW->transferFrom(parentOldDW,
                           label->electrostatic->pMu,
                           perProcPatches,
                           allMaterialsUnion);
  }

//  subNewDW->transferFrom(parentOldDW,
//                         label->global->pX,
//                         perProcPatches,
//                         allMaterialsUnion);
//
//  subNewDW->transferFrom(parentOldDW,
//                         label->global->pID,
//                         perProcPatches,
//                         allMaterialsUnion);

  subscheduler->setInitTimestep(false);

//  // Initialize new parent DW for reduction variables.
//  //   Note:  We should probably skip this and just initialize them with the
//  //          converged value after the polarization loop is done instead.
//  parentNewDW->put(sum_vartype(0.0),
//                   label->electrostatic->rElectrostaticInverseEnergy);
//  parentNewDW->put(matrix_sum(0.0),
//                   label->electrostatic->rElectrostaticInverseStress);
//  parentNewDW->put(sum_vartype(0.0),
//                   label->electrostatic->rElectrostaticRealEnergy);
//  parentNewDW->put(matrix_sum(0.0),
//                   label->electrostatic->rElectrostaticRealStress);

  // Prime variables for the loop
  bool                  converged           =   false;
  int                   numIterations       =   0;
  const PatchSet*       individualPatches   =   level->eachPatch();

  // Populate and compile the subscheduler
  // FIXME:  Should this be here?  Won't we end up compiling every calculation
  //         loop?  Could we compile once in initialize?

  subscheduler->initialize(3,1);

  scheduleInitializeLocalStorage(pg, individualPatches, allMaterials,
                                 subOldDW, subNewDW,
                                 label,
                                 level,
                                 subscheduler);

  scheduleCalculateRealspace(pg, individualPatches, allMaterials,
                             subOldDW, subNewDW,
                             simState, label, coordSys,
                             subscheduler,
                             parentOldDW);

  scheduleCalculatePretransform(pg, individualPatches, allMaterials,
                                subOldDW, subNewDW,
                                simState, label, coordSys,
                                subscheduler);

  scheduleReduceNodeLocalQ(pg, individualPatches, allMaterials,
                           subOldDW, subNewDW,
                           label,
                           subscheduler);

  scheduleTransformRealToFourier(pg, individualPatches, allMaterials,
                                 subOldDW, subNewDW,
                                 label,
                                 level,
                                 subscheduler);

  scheduleCalculateInFourierSpace(pg, individualPatches, allMaterials,
                                  subOldDW, subNewDW,
                                  label,
                                  subscheduler);

  scheduleTransformFourierToReal(pg, individualPatches, allMaterials,
                                 subOldDW, subNewDW,
                                 label,
                                 level,
                                 subscheduler);

  scheduleDistributeNodeLocalQ(pg, individualPatches, allMaterials,
                               subOldDW, subNewDW,
                               label,
                               subscheduler );

//  scheduleCalculatePostTransform(pg, individualPatches, allMaterials,
//                                 subOldDW, subNewDW,
//                                 simState, label, coordSys,
//                                 subscheduler);

  if (f_polarizable) {
    scheduleUpdateFieldAndStress(pg, individualPatches, allMaterials,
                                 subOldDW, subNewDW,
                                 label, coordSys,
                                 subscheduler);
    scheduleCalculateNewDipoles(pg, individualPatches, allMaterials,
                                subOldDW, subNewDW,
                                label,
                                subscheduler);
    scheduleCheckConvergence(pg, individualPatches, allMaterials,
                             subOldDW, subNewDW,
                             label,
                             subscheduler);
  }
  subscheduler->compile();

  // transfer data from parentOldDW to subDW
//  subNewDW->transferFrom(parentOldDW, label->electrostatic->pMu, perProcPatches,
//                         allMaterialsUnion);
//  subNewDW->transferFrom(parentOldDW, label->global->pX, perProcPatches,
//                         allMaterialsUnion);
//  subNewDW->transferFrom(parentOldDW, label->global->pID, perProcPatches,
//                         allMaterialsUnion);

//  DataWarehouse* origOld = subOldDW;
//  DataWarehouse* origNew = subNewDW;
  while (!converged && (numIterations < d_maxPolarizableIterations)) {
    converged = true;
    // Map subNewDW
//    subNewDW    =   subscheduler->get_dw(3);
    // Cycle data warehouses
    subscheduler->advanceDataWarehouse(grid);
//    subNewDW->setScrubbing(DataWarehouse::ScrubNone);
    subscheduler->get_dw(2)->setScrubbing(DataWarehouse::ScrubNone); // Functionally equivalent;
    // Sets current subOldDW to scrubmode ScrubNone

//    DataWarehouse* newOld = subscheduler->get_dw(2);
//    DataWarehouse* newNew = subscheduler->get_dw(3);
    // Run our compiled calculation taskgraph
    subscheduler->execute();

//    DataWarehouse* updatedOld = subscheduler->get_dw(2);
//    DataWarehouse* updatedNew = subscheduler->get_dw(3);

    // Extract the reduced deviation value from the subNewDW
    sum_vartype polarizationDeviation;
    subscheduler->get_dw(3)->get(polarizationDeviation,
                                 label->electrostatic->rPolarizationDeviation);

//    subNewDW->get(polarizationDeviation,
//                  label->electrostatic->rPolarizationDeviation);

    double deviationValue = polarizationDeviation;
    if (f_polarizable && (deviationValue > d_polarizationTolerance)) {
      converged = false;
    }
  //TODO FIXME Force MPI reduction here across converged variable
    numIterations++;
  }
  subNewDW  =   subscheduler->get_dw(3);

  std::cout << "Polarization loop completed with " << numIterations << " iterations." << std::endl;
  // Push energies up to parent DW
  // Should write a transfer function for this
  sum_vartype spmeEnergyTemp;
  subNewDW->get(spmeEnergyTemp,
                label->electrostatic->rElectrostaticRealEnergy);
  parentNewDW->put(spmeEnergyTemp,
                   label->electrostatic->rElectrostaticRealEnergy);
  subNewDW->get(spmeEnergyTemp,
                label->electrostatic->rElectrostaticInverseEnergy);
  parentNewDW->put(spmeEnergyTemp,
                   label->electrostatic->rElectrostaticInverseEnergy);

  // Push stresses up to parent DW
  matrix_sum  spmeStressTemp;
  subNewDW->get(spmeStressTemp,
                label->electrostatic->rElectrostaticRealStress);
  parentNewDW->put(spmeStressTemp,
                   label->electrostatic->rElectrostaticRealStress);
  subNewDW->get(spmeStressTemp,
                label->electrostatic->rElectrostaticInverseStress);
  parentNewDW->put(spmeStressTemp,
                   label->electrostatic->rElectrostaticInverseStress);

  if (f_polarizable) { // Transfer polarizable stress contribution
    subNewDW->get(spmeStressTemp,
                  label->electrostatic->rElectrostaticInverseStressDipole);
    parentNewDW->put(spmeStressTemp,
                     label->electrostatic->rElectrostaticInverseStressDipole);
  }

  parentOldDW->setScrubbing(parentOldDW_scrubmode);
  parentNewDW->setScrubbing(parentNewDW_scrubmode);


  // FIXME APH  Work around for below error (patch/material loop)
  size_t numPatches = perProcPatches->size();
  size_t numAtomTypes = materials->size();

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = perProcPatches->get(patchIndex);
    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex) {
      int atomType = materials->get(typeIndex);
      ParticleSubset* atomSet = subscheduler->get_dw(2)->getParticleSubset(atomType, patch);


      constParticleVariable<Vector> pMuPull, pERealPull, pEInversePull;
      ParticleVariable<Vector>      pMuPush, pERealPush, pEInversePush;
      if (f_polarizable) { // Transfer dipole and field
        subNewDW->get(pMuPull,
                      label->electrostatic->pMu_preReloc,
                      atomSet);
        subNewDW->get(pERealPull,
                      label->electrostatic->pE_electroReal_preReloc,
                      atomSet);
        subNewDW->get(pEInversePull,
                      label->electrostatic->pE_electroInverse_preReloc,
                      atomSet);

        parentNewDW->allocateAndPut(pMuPush,
                                    label->electrostatic->pMu_preReloc,
                                    atomSet);
        parentNewDW->allocateAndPut(pERealPush,
                                    label->electrostatic->pE_electroReal_preReloc,
                                    atomSet);
        parentNewDW->allocateAndPut(pEInversePush,
                                    label->electrostatic->pE_electroInverse_preReloc,
                                    atomSet);

      }

      constParticleVariable<Vector> pFRealPull, pFInversePull;
      ParticleVariable<Vector>      pFRealPush, pFInversePush;
      subNewDW->get(pFRealPull,
                    label->electrostatic->pF_electroReal_preReloc,
                    atomSet);
//      subNewDW->get(pFInversePull,
//                    label->electrostatic->pF_electroInverse_preReloc,
//                    atomSet);
      parentNewDW->allocateAndPut(pFRealPush,
                                  label->electrostatic->pF_electroReal_preReloc,
                                  atomSet);
//      parentNewDW->allocateAndPut(pFInversePush,
//                                  label->electrostatic->pF_electroInverse_preReloc,
//                                  atomSet);

      size_t numAtoms = atomSet->numParticles();
      for (size_t atom = 0; atom < numAtoms; ++atom) {
        if (f_polarizable) {
          pMuPush[atom]         =   pMuPull[atom];
          pERealPush[atom]      =   pERealPull[atom];
          pEInversePush[atom]   =   pEInversePull[atom];
        }
        pFRealPush[atom]    =   pFRealPull[atom];
//        pFInversePush[atom] =   pFInversePull[atom];
      } // Loop over atoms
    } // Loop over atom types
  } // Loop over patches

  // FIXME APH I would like to use this instead of the above, however this complains:
  // 0 Caught exception: An UnknownVariable exception was thrown.
  // /home/jbhooper/arlDev/UintahARL/src/CCA/Components/Schedulers/OnDemandDataWarehouse.cc:1131
  // Unknown variable: ParticleSubset, (low: [int 0, 0, 0], high: [int 50, 50, 25] DWID 1) requested from DW 1, Level 0, patch 0([ [0.00, 0.00, 0.00] [135.18, 135.18, 67.59] ]), material index: 0 (Cannot find particle set on patch)
//  if (f_polarizable) {
//    parentNewDW->transferFrom(subNewDW,
//                              label->electrostatic->pMu_preReloc,
//                              perProcPatches,
//                              allMaterialsUnion);
//    parentNewDW->transferFrom(subNewDW,
//                              label->electrostatic->pE_electroInverse_preReloc,
//                              perProcPatches,
//                              allMaterialsUnion);
//    parentNewDW->transferFrom(subNewDW,
//                              label->electrostatic->pE_electroReal_preReloc,
//                              perProcPatches,
//                              allMaterialsUnion);
//  }
////  parentNewDW->transferFrom(subNewDW,
////                            label->electrostatic->pF_electroInverse_preReloc,
////                            perProcPatches,
////                            allMaterialsUnion);
//  parentNewDW->transferFrom(subNewDW,
//                            label->electrostatic->pF_electroReal_preReloc,
//                            perProcPatches,
//                            allMaterialsUnion);

  // Dipoles have converged, calculate forces
  if (f_polarizable) {
    calculatePostTransformDipole(pg, perProcPatches, materials,
                                 parentOldDW, parentNewDW,
                                 simState, label, coordSys);
  }
  else {
    calculatePostTransform(pg, perProcPatches, materials,
                           parentOldDW, parentNewDW,
                           simState, label, coordSys);
  }

}

void SPME::finalize(    const ProcessorGroup*   pg,
                        const PatchSubset*      patches,
                        const MaterialSubset*   materials,
                        DataWarehouse*          oldDW,
                        DataWarehouse*          newDW,
                        const SimulationStateP* simState,
                        MDSystem*               systemInfo,
                        const MDLabel*          label,
                        CoordinateSystem*       coordSys)
{
  // Do something here?
  /*
   * Eventually we may want to pre-cache memory for, e.g. polarizable loop
   * acceleration in the real-space calculation.  This would be created
   * afer/outside the constructor, so wouldn't automatically be cleaned up.
   *
   * In such a case, the clean-up should go here.
   */
}
