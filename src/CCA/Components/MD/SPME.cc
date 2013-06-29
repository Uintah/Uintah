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
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Box.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <iomanip>

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
    d_Qlock("node-local Q lock"), d_spmePatchLock("SPMEPatch data structures lock")
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
      d_spmePatchLock("SPME shared data structure lock")
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
  SoleVariable<SimpleGrid<dblcomplex>*> Q_global;

  IntVector zero(0, 0, 0);
  SimpleGrid<dblcomplex>* Q = scinew SimpleGrid<dblcomplex>(d_kLimits, zero, IV_ZERO, 0);
  //Q->initialize(dblcomplex(0.0, 0.0));  // Initialization should be done at appropriate place in calculate loop
  Q_global.setData(Q); // TODO May be superfluous

  /*
   * ptrdiff_t is a standard C integer type which is (at least) 32 bits wide
   * on a 32-bit machine and 64 bits wide on a 64-bit machine.
   */
  const ptrdiff_t xdim = d_kLimits(0);
  const ptrdiff_t ydim = d_kLimits(1);
  const ptrdiff_t zdim = d_kLimits(2);
  ptrdiff_t alloc_local, local_n, local_start;
  fftw_plan forwardPlan, backwardPlan;

  fftw_mpi_init();
  // We shouldn't need the local information if FFTW will handle data distribution
  alloc_local = fftw_mpi_local_size_3d(xdim, ydim, zdim, MPI_COMM_WORLD, &local_n, &local_start);
  d_localFFTData = fftw_alloc_complex(alloc_local);

  forwardPlan = fftw_mpi_plan_dft_3d(xdim, ydim, zdim, d_localFFTData, d_localFFTData, pg->getComm(), FFTW_FORWARD, FFTW_MEASURE);
  backwardPlan = fftw_mpi_plan_dft_3d(xdim, ydim, zdim, d_localFFTData, d_localFFTData, pg->getComm(), FFTW_BACKWARD, FFTW_MEASURE);
  //

  // Allocate FFT plans for global Q data
  //ptrdiff_t Q_global_data;
  //forwardPlan = fftw_mpi_plan_dft_3d(d_kLimits.x(),d_kLimits.y(),d_kLimits.z());

  forwardTransformPlan.setData(forwardPlan);
  backwardTransformPlan.setData(backwardPlan);

  // Initially register our basic reduction variables in the DW (initial timestep)
  new_dw->put(sum_vartype(0.0), d_lb->spmeFourierEnergyLabel);
  new_dw->put(matrix_sum(0.0), d_lb->spmeFourierStressLabel);
  new_dw->put(forwardTransformPlan, d_lb->forwardTransformPlanLabel);
  new_dw->put(backwardTransformPlan, d_lb->backwardTransformPlanLabel);
  new_dw->put(Q_global, d_lb->globalQLabel);

  // now the local version of the global Q and Q_scratch arrays
  d_Q_nodeLocal = scinew SimpleGrid<dblcomplex>(d_kLimits, zero, IV_ZERO, 0);
  d_Q_nodeLocalScratch = scinew SimpleGrid<dblcomplex>(d_kLimits, zero, IV_ZERO, 0);

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

    double localCellVolumeFraction = (localCellExtent.x()*localCellExtent.y()*localCellExtent.z())/
                                     (systemCellExtent.x()*systemCellExtent.y()*systemCellExtent.z());

    IntVector patchKLow, patchKHigh;
    for (size_t index = 0; index < 3; ++index) {
      patchKLow[index] = ceil(kReal[index] * (patchLowIndex[index] / systemCellExtent[index]));
      patchKHigh[index] = floor(kReal[index] * (patchHighIndex[index] / systemCellExtent[index]));
    }
    IntVector patchKGridExtents = (patchKHigh - patchKLow); // Number of K grid points in the local patch
    IntVector patchKGridOffset  = patchKLow;                // Starting indices for K grid in local patch

    // Instantiates an SPMEpatch which pre-allocates memory for all the local variables within an SPMEpatch
    SPMEPatch* spmePatch = new SPMEPatch(patchKGridExtents, patchKGridOffset, plusGhostExtents, minusGhostExtents,
                                         patch, localCellVolumeFraction, splineSupport, d_system);

    // Map the current spmePatch into the SPME object.
    d_spmePatchLock.writeLock();
    d_spmePatchMap.insert(SPMEPatchKey(patch->getID(),spmePatch));
    d_spmePatchLock.writeUnlock();
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

    SimpleGrid<double> fBGrid = calculateBGrid(spmePatchExtents, spmePatchOffset);
    SimpleGrid<double> fCGrid = calculateCGrid(spmePatchExtents, spmePatchOffset);

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
  return;
}

void SPME::calculate(const ProcessorGroup* pg,
                     const PatchSubset* patches,
                     const MaterialSubset* materials,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw,
                     SchedulerP& subscheduler,
                     const LevelP& level)
{
  if (d_system->queryBoxChanged()) {
    setup(pg, patches, materials, old_dw, new_dw);
  }

  bool converged = false;
  int numIterations = 0;
  while (!converged && (numIterations < d_maxPolarizableIterations)) {

    // Do calculation steps until the Real->Fourier space transform
    calculatePreTransform(pg, patches, materials, old_dw, new_dw);
    // We have a oncePerPatch threaded, locked initialization of d_Q_nodeLocal in calculatePreTransform
    //   This is inefficient and should be fixed; we should need to do this once per proc, rather than once per patch

    // Q grid reductions
    reduceNodeLocalQ(pg, patches, materials, old_dw, new_dw);

    // Forward transform
    transformRealToFourier(pg, patches, materials, old_dw, new_dw);

    // Redistribute charge grid
    distributeNodeLocalQ(pg, patches, materials, old_dw, new_dw);

    // Do Fourier space calculations on transformed data
    calculateInFourierSpace(pg, patches, materials, old_dw, new_dw);
    // We have a oncePerPatch threaded, locked initialization of d_Q_nodeLocal in calculateInFourierSpace
    //   This is inefficient and should be fixed; we should need to do this once per proc, rather than once per patch

    // Q grid composition (!Not reduction)
    copyToNodeLocalQ(pg, patches, materials, old_dw, new_dw);

    // Reverse transform
    transformFourierToReal(pg, patches, materials, old_dw, new_dw);

    // Redistribute force grid
    distributeNodeLocalQ(pg, patches, materials, old_dw, new_dw);

    checkConvergence();
    numIterations++;
  }

  // Do force spreading and clean up calculations -- or does this go in finalize?
  SPME::calculatePostTransform(pg, patches, materials, old_dw, new_dw);
}

void SPME::finalize(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* materials,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw)
{
  // carry FFTW related SoleVariables forward
  SoleVariable<fftw_plan> forwardTransformPlan;
  SoleVariable<fftw_plan> backwardTransformPlan;
  SoleVariable<SimpleGrid<dblcomplex>*> QGrid;
  old_dw->get(forwardTransformPlan, d_lb->forwardTransformPlanLabel);
  old_dw->get(backwardTransformPlan, d_lb->backwardTransformPlanLabel);
  old_dw->get(QGrid, d_lb->globalQLabel);

  new_dw->put(forwardTransformPlan, d_lb->forwardTransformPlanLabel);
  new_dw->put(backwardTransformPlan, d_lb->backwardTransformPlanLabel);
  new_dw->put(QGrid, d_lb->globalQLabel);

  //d_system->setBoxChanged(false); // No, we set a new box based on the results of the integrator
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
      old_dw->get(pcharge, d_lb->pChargeLabel, pset);
      old_dw->get(pids, d_lb->pParticleIDLabel, pset);


      // Verify the charge map can contain the necessary data and get it
      currentSPMEPatch->verifyChargeMapAllocation(pset->numParticles(),globalAtomType);
      std::vector<SPMEMapPoint>* gridMap = currentSPMEPatch->getChargeMap(globalAtomType);

      // and generate the charge map
      generateChargeMap(gridMap, pset, px, pids);
      mapChargeToGrid(currentSPMEPatch, gridMap, pset, pcharge);

    }  // end Atom Type Loop
  }  // end Patch loop
  d_Qlock.lock();
  d_Q_nodeLocal->initialize(dblcomplex(0.0,0.0));
  d_Qlock.unlock();
  // This initializes things once for each patch, but it's probably the easiest way to do it nonetheless.
  // Later we can set a state variable to check if it's initialized and if so not re-initialize it.
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

    SimpleGrid<std::complex<double> >* Q_patchLocal = currentSPMEPatch->getQ();
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

    for (size_t kX = 0; kX < xMax; ++kX) {
      for (size_t kY = 0; kY < yMax; ++kY) {
        for (size_t kZ = 0; kZ < zMax; ++kZ) {

          std::complex<double> gridValue = (*Q_patchLocal)(kX, kY, kZ);
          // Calculate (Q*Q^)*(B*C)
          (*Q_patchLocal)(kX, kY, kZ) *= (*fTheta)(kX, kY, kZ);
          spmeFourierEnergy += std::abs((*Q_patchLocal)(kX, kY, kZ) * conj(gridValue));
          spmeFourierStress += std::abs((*Q_patchLocal)(kX, kY, kZ) * conj(gridValue)) * (*stressPrefactor)(kX, kY, kZ);
        }
      }
    }
//    }  // end AtomType loop
  }  // end SPME Patch loop

  // put updated values for reduction variables into the DW
  new_dw->put(sum_vartype(0.5 * spmeFourierEnergy), d_lb->spmeFourierEnergyLabel);
  new_dw->put(matrix_sum(0.5 * spmeFourierStress), d_lb->spmeFourierStressLabel);
}

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
      mapForceFromGrid(currentSPMEPatch, gridMap, pset, pcharge, pforcenew);

      ParticleVariable<double> pchargenew;
      new_dw->allocateAndPut(pchargenew, d_lb->pChargeLabel_preReloc, pset);
      // carry these values over for now
      pchargenew.copyData(pcharge);
    }
  }
}

void SPME::transformRealToFourier(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* materials,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
//  SoleVariable<fftw_plan> forwardTransformPlan;
//  old_dw->get(forwardTransformPlan, d_lb->forwardTransformPlanLabel);
//  fftw_execute(forwardTransformPlan.get());

  int xdim = d_kLimits(0);
  int ydim = d_kLimits(0);
  int zdim = d_kLimits(0);

  fftw_complex* array_fft = reinterpret_cast<fftw_complex*>(d_Q_nodeLocal->getDataPtr());
  fftw_plan forwardTransformPlan = fftw_plan_dft_3d(xdim, ydim, zdim, array_fft, array_fft, FFTW_FORWARD, FFTW_MEASURE);
  fftw_execute(forwardTransformPlan);
}

void SPME::transformFourierToReal(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* materials,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
//  SoleVariable<fftw_plan> backwardTransformPlan;
//  old_dw->get(backwardTransformPlan, d_lb->backwardTransformPlanLabel);
//  fftw_execute(backwardTransformPlan.get());

  int xdim = d_kLimits.x();
  int ydim = d_kLimits.y();
  int zdim = d_kLimits.z();

  fftw_complex* array_fft = reinterpret_cast<fftw_complex*>(d_Q_nodeLocal->getDataPtr());
  fftw_plan backwardTransformPlan = fftw_plan_dft_3d(xdim, ydim, zdim, array_fft, array_fft, FFTW_BACKWARD, FFTW_MEASURE);
  fftw_execute(backwardTransformPlan);
}

//----------------------------------------------------------------------------
// Setup related routines
std::vector<dblcomplex> SPME::generateBVector(const std::vector<double>& mFractional,
                                              const int initialIndex,
                                              const int localGridExtent) const
{
  double PI = acos(-1.0);
  double twoPI = 2.0 * PI;
  int n = d_interpolatingSpline.getOrder();

  std::vector<dblcomplex> B(localGridExtent);
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
    B[BIndex] = numerator / denominator;
  }
  return B;
}

SimpleGrid<double> SPME::calculateBGrid(const IntVector& localExtents,
                                        const IntVector& globalOffset) const
{
  size_t limit_Kx = d_kLimits.x();
  size_t limit_Ky = d_kLimits.y();
  size_t limit_Kz = d_kLimits.z();

  std::vector<double> mf1 = SPME::generateMFractionalVector(limit_Kx);
  std::vector<double> mf2 = SPME::generateMFractionalVector(limit_Ky);
  std::vector<double> mf3 = SPME::generateMFractionalVector(limit_Kz);

  // localExtents is without ghost grid points
  std::vector<dblcomplex> b1 = generateBVector(mf1, globalOffset.x(), localExtents.x());
  std::vector<dblcomplex> b2 = generateBVector(mf2, globalOffset.y(), localExtents.y());
  std::vector<dblcomplex> b3 = generateBVector(mf3, globalOffset.z(), localExtents.z());

  SimpleGrid<double> BGrid(localExtents, globalOffset, IV_ZERO, 0);  // No ghost cells; internal only

  size_t xExtents = localExtents.x();
  size_t yExtents = localExtents.y();
  size_t zExtents = localExtents.z();

  int xOffset = globalOffset.x();
  int yOffset = globalOffset.y();
  int zOffset = globalOffset.z();

  for (size_t kX = 0; kX < xExtents; ++kX) {
    for (size_t kY = 0; kY < yExtents; ++kY) {
      for (size_t kZ = 0; kZ < zExtents; ++kZ) {
        BGrid(kX, kY, kZ) = norm(b1[kX + xOffset]) * norm(b2[kY + yOffset]) * norm(b3[kZ + zOffset]);
      }
    }
  }
  return BGrid;
}

SimpleGrid<double> SPME::calculateCGrid(const IntVector& extents,
                                        const IntVector& offset) const
{
  // sanity check
  std::cerr << "System Volume: " << d_systemVolume << endl;

  std::vector<double> mp1 = SPME::generateMPrimeVector(d_kLimits.x());
  std::vector<double> mp2 = SPME::generateMPrimeVector(d_kLimits.y());
  std::vector<double> mp3 = SPME::generateMPrimeVector(d_kLimits.z());

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

  SimpleGrid<double> CGrid(extents, offset, IV_ZERO, 0);  // No ghost cells; internal only
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
  return CGrid;
}

void SPME::calculateStressPrefactor(SimpleGrid<Matrix3>* stressPrefactor,
                                    const IntVector& extents,
                                    const IntVector& offset)
{
  std::vector<double> mp1 = SPME::generateMPrimeVector(d_kLimits.x());
  std::vector<double> mp2 = SPME::generateMPrimeVector(d_kLimits.y());
  std::vector<double> mp3 = SPME::generateMPrimeVector(d_kLimits.z());

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
          for (size_t s1 = 0; s1 < 3; ++s1) {
            for (size_t s2 = 0; s2 < 3; ++s2) {
              localStressContribution(s1, s2) *= (m[s1] * m[s2]);
            }
          }

          // Account for delta function
          for (size_t delta = 0; delta < 3; ++delta) {
            localStressContribution(delta, delta) += 1.0;
          }

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

    Point position = particlePositions[pidx];
    particleId pid = particleIDs[pidx];
    Vector particleGridCoordinates;

    //Calculate reduced coordinates of point to recast into charge grid
    particleGridCoordinates = (position.asVector()) * d_inverseUnitCell;
    // ** NOTE: JBH --> We may want to do this with a bit more thought eventually, since multiplying by the InverseUnitCell
    //                  is expensive if the system is orthorhombic, however it's not clear it's more expensive than dropping
    //                  to call MDSystem->isOrthorhombic() and then branching the if statement appropriately.

    Vector kReal = d_kLimits.asVector();
    particleGridCoordinates *= kReal;
    IntVector particleGridOffset(particleGridCoordinates.asPoint());
    Vector splineValues = particleGridOffset.asVector() - particleGridCoordinates;

    vector<double> xSplineArray = d_interpolatingSpline.evaluateGridAligned(splineValues.x());
    vector<double> ySplineArray = d_interpolatingSpline.evaluateGridAligned(splineValues.y());
    vector<double> zSplineArray = d_interpolatingSpline.evaluateGridAligned(splineValues.z());

    vector<double> xSplineDeriv = d_interpolatingSpline.derivativeGridAligned(splineValues.x());
    vector<double> ySplineDeriv = d_interpolatingSpline.derivativeGridAligned(splineValues.y());
    vector<double> zSplineDeriv = d_interpolatingSpline.derivativeGridAligned(splineValues.z());

    IntVector extents(xSplineArray.size(), ySplineArray.size(), zSplineArray.size());

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
    SPMEMapPoint currentMapPoint(pid, particleGridOffset, chargeGrid, forceGrid);
    chargeMap->push_back(currentMapPoint);
    // TODO -> Look at building these in place in the chargeMap to save time.
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
// PostTransform calculation related routines
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
    if (pidx < 5) {
      std::cerr << " Force Check (" << pidx << "): " << newForce << endl;
      pforcenew[pidx] = newForce;
    }
  }
}

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
      int x_index = xBase + xmask;
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_index = yBase + ymask;
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_index = zBase + zmask;
          (*d_Q_nodeLocal)(x_index, y_index, z_index) = (*Q_patchLocal)(x_index, y_index, z_index);
        }
      }
    }
  }
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
      int x_local = xBase + xmask;
      int x_global = x_local;
      if (x_global >= xMax) { x_global -= xMax; }
      if (x_global < 0)     { x_global += xMax; }
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_local = yBase + ymask;
        int y_global = y_local;
        if (y_global >= yMax) { y_global -= yMax; }
        if (y_global < 0)     { y_global += yMax; }
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_local = zBase + zmask;
          int z_global = z_local;
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
      int x_local = xBase + xmask;
      int x_global = x_local;
      if (x_global >= xMax) { x_global -= xMax; }
      if (x_global < 0)     { x_global += xMax; }
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_local = yBase + ymask;
        int y_global = y_local;
        if (y_global >= yMax) { y_global -= yMax; }
        if (y_global < 0)     { y_global += yMax; }
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_local = zBase + zmask;
          int z_global = z_local;
          if (z_global >= zMax) { z_global -= zMax; }
          if (z_global < 0)     { z_global += zMax; }
          // Recall d_Q_nodeLocal is a complete copy of the Q grid for reduction across MPI threads
          (*Q_patchLocal)(x_local,y_local,z_local) = (*d_Q_nodeLocal)(x_global,y_global,z_global);
        }
      }
    }
    d_Qlock.lock();
    d_Q_nodeLocal->initialize(dblcomplex(0.0,0.0));
    d_Qlock.unlock();
  }
}

bool SPME::checkConvergence() const
{
  // Subroutine determines if polarizable component has converged
  bool polarizable = getPolarizableCalculation();
  if (!polarizable) {
    return true;
  } else {
    // throw an exception for now, but eventually will check convergence here.
    throw InternalError("Error: Polarizable force field not yet implemented!", __FILE__, __LINE__);
  }
}

