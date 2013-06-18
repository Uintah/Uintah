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

using namespace Uintah;

extern SCIRun::Mutex cerrLock;

static DebugStream spme_dbg("SPMEDBG", false);

SPME::SPME()
{

}

SPME::~SPME()
{
  // cleanup the memory we have dynamically allocated
  std::vector<SPMEPatch*>::iterator PatchIterator;
  for (PatchIterator = d_spmePatches.begin(); PatchIterator != d_spmePatches.end(); ++PatchIterator) {
    SPMEPatch* spmePatch = *PatchIterator;

    SimpleGrid<std::complex<double> >* q = spmePatch->getQ();
    if (q) {
      delete q;
    }

    SimpleGrid<double>* theta = spmePatch->getTheta();
    if (theta) {
      delete theta;
    }

//    SimpleGrid<Matrix3>* stressPrefactor = spmePatch->getStressPrefactor();
//    if (stressPrefactor) {
//      delete stressPrefactor;
//    }

    delete spmePatch;
  }

  if (d_Q) {
    delete d_Q;
  }

  std::map<PatchMaterialKey, std::vector<SPMEMapPoint>*>::iterator iter;
  for (iter = d_gridMap.begin(); iter != d_gridMap.end(); iter++) {
    std::vector<SPMEMapPoint>* gridmap = iter->second;
    delete gridmap;
  }

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
      d_maxPolarizableIterations(maxPolarizableIterations)
{
  d_interpolatingSpline = ShiftedCardinalBSpline(splineOrder);
  d_electrostaticMethod = Electrostatics::SPME;
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

  // Initially register our basic reduction variables in the DW
  new_dw->put(sum_vartype(0.0), d_lb->spmeFourierEnergyLabel);
  new_dw->put(matrix_sum(0.0), d_lb->spmeFourierStressLabel);

  // now create SoleVariables to place in the DW; global Q and the FFTW plans (forward and backward)
  // FFTW plans for forward and backward 3D transform of global Q grid
  SoleVariable<SimpleGrid<dblcomplex>*> QGrid;
  SoleVariable<fftw_plan> forwardTransformPlan;
  SoleVariable<fftw_plan> backwardTransformPlan;

  IntVector zero(0, 0, 0);
  SimpleGrid<dblcomplex>* Q = scinew SimpleGrid<dblcomplex>(d_kLimits, zero, 0);
  Q->initialize(dblcomplex(0.0, 0.0));
  QGrid.setData(Q);

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
  alloc_local = fftw_mpi_local_size_3d(xdim, ydim, zdim, MPI_COMM_WORLD, &local_n, &local_start);
  d_localFFTData = fftw_alloc_complex(alloc_local);

  forwardPlan = fftw_mpi_plan_dft_3d(xdim, ydim, zdim, d_localFFTData, d_localFFTData, pg->getComm(), FFTW_FORWARD, FFTW_MEASURE);
  backwardPlan = fftw_mpi_plan_dft_3d(xdim, ydim, zdim, d_localFFTData, d_localFFTData, pg->getComm(), FFTW_BACKWARD, FFTW_MEASURE);

  forwardTransformPlan.setData(forwardPlan);
  backwardTransformPlan.setData(backwardPlan);

  new_dw->put(forwardTransformPlan, d_lb->forwardTransformPlanLabel);
  new_dw->put(backwardTransformPlan, d_lb->backwardTransformPlanLabel);
  new_dw->put(QGrid, d_lb->globalQLabel);

  // now the local version of the global Q array
  d_Q = scinew SimpleGrid<dblcomplex>(d_kLimits, zero, 0);
  d_Q->initialize(dblcomplex(0.0, 0.0));

  // Get useful information from global system descriptor to work with locally.
  d_unitCell = d_system->getUnitCell();
  d_inverseUnitCell = d_system->getInverseCell();
  d_systemVolume = d_system->getCellVolume();

  int numLocalSites;
  IntVector patchLowIndex, patchHighIndex, patchExtents;
  int xRatio, yRatio, zRatio;
  int spaceAdjustment = 2;

  // Create charge-maps for each patch/material set
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);
    unsigned int numMatls = materials->size();
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = materials->get(m);

      // (x/X * y/Y * z/Z) * N_i * spaceAdjustment
      patchLowIndex = patch->getCellLowIndex();
      patchHighIndex = patch->getCellHighIndex();
      patchExtents = patchHighIndex - patchLowIndex;
      xRatio = patchExtents.x() / xdim;
      yRatio = patchExtents.y() / ydim;
      zRatio = patchExtents.z() / zdim;
      numLocalSites = (xRatio * yRatio * zRatio) * d_system->getNumAtoms() * spaceAdjustment;
      std::vector<SPMEMapPoint>* gridmap = new std::vector<SPMEMapPoint>();
      gridmap->reserve(numLocalSites);
      d_gridMap.insert(pair<PatchMaterialKey, std::vector<SPMEMapPoint>*>(PatchMaterialKey(patch, matl), gridmap));
    }
  }
}

// Note:  Must run SPME->setup() each time there is a new box/K grid mapping (e.g. every step for NPT)
//          This should be checked for in the system electrostatic driver
void SPME::setup(const ProcessorGroup* pg,
                 const PatchSubset* patches,
                 const MaterialSubset* materials,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw)
{
  size_t numPatches = patches->size();
  d_spmePatches.reserve(numPatches);

  for (size_t p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);

    Vector totalCellExtent = (d_system->getCellExtent()).asVector();
    Vector patchLowIndex = (patch->getCellLowIndex()).asVector();
    Vector patchHighIndex = (patch->getCellHighIndex()).asVector();

    SCIRun::IntVector patchKLow, patchKHigh;
    for (size_t idx = 0; idx < 3; ++idx) {
      int KComponent = d_kLimits(idx);
      patchKLow[idx] = ceil(static_cast<double>(KComponent) * (patchLowIndex[idx] / totalCellExtent[idx]));
      patchKHigh[idx] = floor(static_cast<double>(KComponent) * (patchHighIndex[idx] / totalCellExtent[idx]));
    }

    IntVector patchKGridExtents = (patchKHigh - patchKLow);  // Number of K Grid points for this patch
    IntVector patchKGridOffset = patchKLow;                  // Lowest K Grid point vector
    int splineSupport = d_interpolatingSpline.getSupport();
    IntVector plusGhostExtents = IntVector(splineSupport, splineSupport, splineSupport);
    IntVector minusGhostExtents = IntVector(0, 0, 0);  // All ghosts are in the positive direction for shifted splines

    SPMEPatch* spmePatch = new SPMEPatch(patchKGridExtents, patchKGridOffset, plusGhostExtents, minusGhostExtents, patch);

    // Check to make sure plusGhostExtents+minusGhostExtents is right way to enter number of ghost cells (i.e. total, not per offset)
    SimpleGrid<dblcomplex>* q = scinew SimpleGrid<dblcomplex>(patchKGridExtents, patchKGridOffset, splineSupport);
    q->initialize(std::complex<double>(0.0, 0.0));

    // No ghost cells; internal only
    SimpleGrid<Matrix3>* stressPrefactor = scinew SimpleGrid<Matrix3>(patchKGridExtents, patchKGridOffset, 0);
    calculateStressPrefactor(stressPrefactor, patchKGridExtents, patchKGridOffset);

    // No ghost cells; internal only
    SimpleGrid<double>* fTheta = scinew SimpleGrid<double>(patchKGridExtents, patchKGridOffset, 0);
    fTheta->initialize(0.0);

    // Calculate B and C - we should only have to do this if KLimits or the inverse cell changes
    SimpleGrid<double> fBGrid = calculateBGrid(patchKGridExtents, patchKGridOffset);
    SimpleGrid<double> fCGrid = calculateCGrid(patchKGridExtents, patchKGridOffset);

    // Composite B and C into Theta
    size_t xExtent = patchKGridExtents.x();
    size_t yExtent = patchKGridExtents.y();
    size_t zExtent = patchKGridExtents.z();
    for (size_t xidx = 0; xidx < xExtent; ++xidx) {
      for (size_t yidx = 0; yidx < yExtent; ++yidx) {
        for (size_t zidx = 0; zidx < zExtent; ++zidx) {

          // Composite B and C into Theta
          (*fTheta)(xidx, yidx, zidx) = fBGrid(xidx, yidx, zidx) * fCGrid(xidx, yidx, zidx);

          if (spme_dbg.active()) {
            cerrLock.lock();
            // looking for "nan" values interspersed in B and C
            if (std::isnan(fBGrid(xidx, yidx, zidx))) {
              std::cout << "B: " << xidx << " " << yidx << " " << zidx << std::endl;
              std::cin.get();
            }
            if (std::isnan(fCGrid(xidx, yidx, zidx))) {
              std::cout << "C: " << xidx << " " << yidx << " " << zidx << std::endl;
              std::cin.get();
            }
            cerrLock.unlock();
          }
        }
      }
    }
    spmePatch->setTheta(fTheta);
    spmePatch->setStressPrefactor(stressPrefactor);
    spmePatch->setQ(q);
    d_spmePatches.push_back(spmePatch);
  }
}

void SPME::calculate(const ProcessorGroup* pg,
                     const PatchSubset* patches,
                     const MaterialSubset* materials,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw)
{
  bool converged = false;
  int numIterations = 0;
  while (!converged && (numIterations < d_maxPolarizableIterations)) {

    // Do calculation steps until the Real->Fourier space transform
    calculatePreTransform(pg, patches, materials, old_dw, new_dw);

    // Reduce Q, forward transform, and redistribute charge grid
    transformRealToFourier(pg, patches, materials, old_dw, new_dw);

    // Do Fourier space calculations on transformed data
    calculateInFourierSpace(pg, patches, materials, old_dw, new_dw);

    // Reduce Q, reverse transform, and redistribute force grid
    transformFourierToReal(pg, patches, materials, old_dw, new_dw);

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
}

void SPME::calculatePreTransform(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* materials,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  std::vector<SPMEPatch*>::iterator PatchIterator;
  for (PatchIterator = d_spmePatches.begin(); PatchIterator != d_spmePatches.end(); ++PatchIterator) {
    SPMEPatch* spmePatch = *PatchIterator;
    unsigned int numMatls = materials->size();
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = materials->get(m);

      const Patch* patch = spmePatch->getPatch();
      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);

      constParticleVariable<Point> px;
      constParticleVariable<double> pcharge;
      constParticleVariable<long64> pids;
      old_dw->get(px, d_lb->pXLabel, pset);
      old_dw->get(pcharge, d_lb->pChargeLabel, pset);
      old_dw->get(pids, d_lb->pParticleIDLabel, pset);

      // When we have a material iterator in here, we should store/get charge by material.
      // Charge represents the static charge on a particle, which is set by particle type.
      // No need to store one for each particle. -- JBH

      // Generate the data that maps the charges in the patch onto the grid
      PatchMaterialKey key(spmePatch->getPatch(), matl);
      std::vector<SPMEMapPoint>* gridMap = d_gridMap.find(key)->second;
      generateChargeMap(gridMap, pset, px, pids);

      // Calculate Q(r) for each local patch (if more than one per proc)
      mapChargeToGrid(spmePatch, gridMap, pset, pcharge);
    }
  }

  // local reduction for Q grids belonging to SPMEPatches (if more than one patch per proc)
  reduceLocalQGrids();

  // put local Q grid in for reduction via infrastructure
//  new_dw->put(q_kgrid_sum(*(d_Q->getDataArray())), d_lb->QLabel);
}

void SPME::calculateInFourierSpace(const ProcessorGroup* pg,
                                   const PatchSubset* patches,
                                   const MaterialSubset* materials,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  double spmeFourierEnergy = 0.0;
  Matrix3 spmeFourierStress(0.0);

  std::vector<SPMEPatch*>::iterator PatchIterator;
  for (PatchIterator = d_spmePatches.begin(); PatchIterator != d_spmePatches.end(); PatchIterator++) {
    SPMEPatch* spmePatch = *PatchIterator;
    SimpleGrid<std::complex<double> >* Q = spmePatch->getQ();
    SimpleGrid<double>* fTheta = spmePatch->getTheta();
    SimpleGrid<Matrix3>* stressPrefactor = spmePatch->getStressPrefactor();

    // Multiply the transformed Q by B*C to get Theta
    IntVector localExtents = spmePatch->getLocalExtents();
    size_t xMax = localExtents.x();
    size_t yMax = localExtents.y();
    size_t zMax = localExtents.z();

    for (size_t kX = 0; kX < xMax; ++kX) {
      for (size_t kY = 0; kY < yMax; ++kY) {
        for (size_t kZ = 0; kZ < zMax; ++kZ) {
          std::complex<double> gridValue = (*Q)(kX, kY, kZ);

          // Calculate (Q*Q^)*(B*C)
          (*Q)(kX, kY, kZ) *= (*fTheta)(kX, kY, kZ);
          spmeFourierEnergy += std::abs((*Q)(kX, kY, kZ) * conj(gridValue));
          spmeFourierStress += std::abs((*Q)(kX, kY, kZ) * conj(gridValue)) * (*stressPrefactor)(kX, kY, kZ);
        }
      }
    }
  }
  reduceLocalQGrids();

  new_dw->put(sum_vartype(0.5 * spmeFourierEnergy), d_lb->spmeFourierEnergyLabel);
  new_dw->put(matrix_sum(0.5 * spmeFourierStress), d_lb->spmeFourierStressLabel);
}

void SPME::calculatePostTransform(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* materials,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  unsigned int numPatches = d_spmePatches.size();
  for (unsigned int p = 0; p < numPatches; p++) {
    SPMEPatch* spmePatch = d_spmePatches[p];
    unsigned int numMatls = materials->size();
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = materials->get(m);

      const Patch* patch = spmePatch->getPatch();
      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      constParticleVariable<double> pcharge;
      old_dw->get(pcharge, d_lb->pChargeLabel, pset);

      ParticleVariable<Vector> pforcenew;
      new_dw->getModifiable(pforcenew, d_lb->pForceLabel_preReloc, pset);

      PatchMaterialKey key(spmePatch->getPatch(), matl);
      std::vector<SPMEMapPoint>* gridMap = d_gridMap.find(key)->second;

      // Calculate electrostatic contribution to f_ij(r)
      mapForceFromGrid(spmePatch, gridMap, pset, pcharge, pforcenew);

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

  std::vector<SPMEPatch*>::iterator PatchIterator;
  for (PatchIterator = d_spmePatches.begin(); PatchIterator != d_spmePatches.end(); PatchIterator++) {
    SPMEPatch* spmePatch = *PatchIterator;

    int xdim = d_kLimits(0);
    int ydim = d_kLimits(0);
    int zdim = d_kLimits(0);

    SimpleGrid<dblcomplex>* Q = spmePatch->getQ();
    fftw_complex* array_fft = reinterpret_cast<fftw_complex*>(Q->getDataPtr());
    fftw_plan d_forwardTransformPlan = fftw_plan_dft_3d(xdim, ydim, zdim, array_fft, array_fft, FFTW_FORWARD, FFTW_MEASURE);
    fftw_execute(d_forwardTransformPlan);
  }
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

  std::vector<SPMEPatch*>::iterator PatchIterator;
  for (PatchIterator = d_spmePatches.begin(); PatchIterator != d_spmePatches.end(); PatchIterator++) {
    SPMEPatch* spmePatch = *PatchIterator;

    int xdim = d_kLimits.x();
    int ydim = d_kLimits.y();
    int zdim = d_kLimits.z();

    SimpleGrid<dblcomplex>* Q = spmePatch->getQ();
    fftw_complex* array_fft = reinterpret_cast<fftw_complex*>(Q->getDataPtr());
    fftw_plan d_backwardTransformPlan = fftw_plan_dft_3d(xdim, ydim, zdim, array_fft, array_fft, FFTW_BACKWARD, FFTW_MEASURE);
    fftw_execute(d_backwardTransformPlan);
  }
}

void SPME::reduceLocalQGrids()
{
  LinearArray3<dblcomplex>* localQ = d_Q->getDataArray();

  // If there's only one patch per proc
  if (d_spmePatches.size() == 1) {
    localQ->copyData(*(d_spmePatches[0]->getQ()->getDataArray()));
    return;
  }

  // >1 patch per proc; do the local Q reduction
  std::vector<SPMEPatch*>::iterator PatchIterator;
  for (PatchIterator = d_spmePatches.begin(); PatchIterator != d_spmePatches.end(); ++PatchIterator) {
    SPMEPatch* spmePatch = *PatchIterator;
    *localQ += *(spmePatch->getQ()->getDataArray());
  }
  d_Q->getDataArray()->copyData(*localQ);
}

bool SPME::checkConvergence()
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

  SimpleGrid<double> BGrid(localExtents, globalOffset, 0);  // No ghost cells; internal only

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
  std::vector<double> mp1 = SPME::generateMPrimeVector(d_kLimits.x());
  std::vector<double> mp2 = SPME::generateMPrimeVector(d_kLimits.y());
  std::vector<double> mp3 = SPME::generateMPrimeVector(d_kLimits.z());

  size_t xExtents = extents.x();
  size_t yExtents = extents.y();
  size_t zExtents = extents.z();

  int xOffset = offset.x();
  int yOffset = offset.y();
  int zOffset = offset.z();
  double PI = acos(-1.0);
  double PI2 = PI * PI;
  double invBeta2 = 1.0 / (d_ewaldBeta * d_ewaldBeta);
  double invVolFactor = 1.0 / (d_systemVolume * PI);

  std::cerr << "System Volume: " << d_systemVolume << endl;

  SimpleGrid<double> CGrid(extents, offset, 0);  // No ghost cells; internal only
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

  size_t xExtents = extents.x();
  size_t yExtents = extents.y();
  size_t zExtents = extents.z();

  int XOffset = offset.x();
  int YOffset = offset.y();
  int ZOffset = offset.z();

  double PI = acos(-1.0);
  double PI2 = PI * PI;
  double invBeta2 = 1.0 / (d_ewaldBeta * d_ewaldBeta);

  for (size_t kX = 0; kX < xExtents; ++kX) {
    for (size_t kY = 0; kY < yExtents; ++kY) {
      for (size_t kZ = 0; kZ < zExtents; ++kZ) {
        if (kX != 0 || kY != 0 || kZ != 0) {
          Vector m(mp1[kX + XOffset], mp2[kY + YOffset], mp3[kZ + ZOffset]);
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
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
    particleIndex pidx = *iter;
    Point position = particlePositions[pidx];
    particleId pid = particleIDs[pidx];
    Vector particleGridCoordinates;

    //Calculate reduced coordinates of point to recast into charge grid
    particleGridCoordinates = (position.asVector()) * d_inverseUnitCell;
    // ** NOTE: JBH --> We may want to do this with a bit more thought eventually, since multiplying by the InverseUnitCell
    //                  is expensive if the system is orthorhombic, however it's not clear it's more expensive than dropping
    //                  to call MDSystem->IsOrthorhombic() and then branching the if statement appropriately.

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
    SimpleGrid<double> chargeGrid(extents, particleGridOffset, 0);

    SimpleGrid<SCIRun::Vector> forceGrid(extents, particleGridOffset, 0);
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
  }
}

void SPME::mapChargeToGrid(SPMEPatch* spmePatch,
                           const std::vector<SPMEMapPoint>* gridMap,
                           ParticleSubset* pset,
                           constParticleVariable<double>& charges)
{
  // Reset charges before we start adding onto them.
  SimpleGrid<dblcomplex>* Q = spmePatch->getQ();
  Q->initialize(0.0);

  ParticleSubset::iterator particleIter;
  for (particleIter = pset->begin(); particleIter != pset->end(); particleIter++) {
    particleIndex pidx = *particleIter;

    double charge = charges[pidx];
    const SimpleGrid<double> chargeMap = (*gridMap)[pidx].getChargeGrid();
    int splineSupport = d_interpolatingSpline.getSupport();

    IntVector QAnchor = chargeMap.getOffset();  // Location of the 0,0,0 origin for the charge map grid
    IntVector SupportExtent = chargeMap.getExtents();  // Extents of the charge map grid
    int xBase = QAnchor.x();
    int yBase = QAnchor.y();
    int zBase = QAnchor.z();
    for (int xmask = 0; xmask < splineSupport; ++xmask) {
      for (int ymask = 0; ymask < splineSupport; ++ymask) {
        for (int zmask = 0; zmask < splineSupport; ++zmask) {
          dblcomplex val = charge * chargeMap(xmask, ymask, zmask);

          // We need only wrap in the positive direction.  Therefore QAnchor.i + imask will never be less than zero
          int x_anchor = xBase + xmask;
          if (x_anchor >= d_kLimits.x())
            x_anchor -= d_kLimits.x();

          int y_anchor = yBase + ymask;
          if (y_anchor >= d_kLimits.y())
            y_anchor -= d_kLimits.y();

          int z_anchor = zBase + zmask;
          if (z_anchor >= d_kLimits.z())
            z_anchor -= d_kLimits.z();

          (*Q)(x_anchor, y_anchor, z_anchor) += val;
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
  SimpleGrid<std::complex<double> >* Q = spmePatch->getQ();

  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
    particleIndex pidx = *iter;

    SimpleGrid<SCIRun::Vector> forceMap = (*gridMap)[pidx].getForceGrid();
    double charge = charges[pidx];

    SCIRun::Vector newForce = Vector(0, 0, 0);
    IntVector QAnchor = forceMap.getOffset();  // Location of the 0,0,0 origin for the force map grid
    IntVector supportExtent = forceMap.getExtents();  // Extents of the force map grid

    int xBase = QAnchor.x();
    int yBase = QAnchor.y();
    int zBase = QAnchor.z();

    int kX = d_kLimits.x();
    int kY = d_kLimits.y();
    int kZ = d_kLimits.z();

    int xExtent = supportExtent.x();
    int yExtent = supportExtent.y();
    int zExtent = supportExtent.z();

    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int x_anchor = xBase + xmask;
      if (x_anchor >= kX)
        x_anchor -= kX;
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_anchor = yBase + ymask;
        if (y_anchor >= kY)
          y_anchor -= kY;
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_anchor = zBase + zmask;
          if (z_anchor >= kZ)
            z_anchor -= kZ;

          double QReal = real((*Q)(x_anchor, y_anchor, z_anchor));
          double QMag = std::abs((*Q)(x_anchor, y_anchor, z_anchor));

          newForce += forceMap(xmask, ymask, zmask) * QReal * charge * d_inverseUnitCell;
        }
      }
    }

    //49227.43325439056
    /*
     *  f1=  -2.8440279063396052        2.6027710155811845       0.99069892785288971
     f2=  -1.2959262337781086       -2.2140864761359698        2.5686305300200476
     f3=   4.2899558647618461       -2.2060911777299874       -3.0672208465506792
     f4=   3.1706091601611490        2.0423127670957384       -4.2814464749644880
     f5=  -2.8145420677778201        2.3389711802598274        1.3094128311605495
     */

    if (pidx < 5) {
      std::cerr << " Force Check (" << pidx << "): " << newForce << endl;
      pforcenew[pidx] = newForce;
    }

  }
}

