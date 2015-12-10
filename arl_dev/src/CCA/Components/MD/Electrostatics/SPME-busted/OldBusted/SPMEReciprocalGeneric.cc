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

extern SCIRun::Mutex coutLock;

// Q Grid distribution and collection
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

// Fourier transform related routines
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

  // setup and copy data to-and-from this processor's portion of the global FFT array
  fftw_complex* localChunk = d_localFFTData.complexData;
  ptrdiff_t localN = d_localFFTData.numElements; // a (local_n * kLimits.y * kLimits.z) chunk of the global array
  ptrdiff_t localStart = d_localFFTData.startAddress;
  dblcomplex* nodeLocalData = recvbuf + localStart;
  size_t numElements = localN * d_kLimits[1] * d_kLimits[2];

  std::memcpy(localChunk, nodeLocalData, numElements * sizeof(dblcomplex));
  fftw_execute(d_forwardPlan);
  std::memcpy(nodeLocalData, localChunk, numElements * sizeof(dblcomplex));
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
  size_t numLocalAtomTypes = materials->size();

  for (size_t p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;
    IntVector localExtents      = currentSPMEPatch->getLocalExtents();

    for (size_t localAtomTypeIndex = 0; localAtomTypeIndex < numLocalAtomTypes; ++localAtomTypeIndex) {
      int atomType = materials->get(localAtomTypeIndex);
      SimpleGrid<double>*             fTheta = currentSPMEPatch->getTheta();
      SimpleGrid<Matrix3>*   stressPrefactor = currentSPMEPatch->getStressPrefactor();

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
    }  // end AtomType loop

    if (spme_dbg.active()) {
      coutLock.lock();
      std::cout.setf(std::ios_base::left);
      std::cout << std::setw(30) << Thread::self()->getThreadName();
      std::cout << "Uintah thread ID: " << std::setw(4) << Thread::self()->myid()
                << "Thread group: " <<  std::setw(10) <<Thread::self()->getThreadGroup()
                << "Patch: " <<  std::setw(4) <<patch->getID()
                << "Fourier-Energy: " << spmeFourierEnergy << std::endl;
      coutLock.unlock();
    }
  }  // end SPME Patch loop

  // put updated values for reduction variables into the DW
  new_dw->put(sum_vartype(0.5 * spmeFourierEnergy), d_label->electrostatic->rElectrostaticInverseEnergy);
  new_dw->put(matrix_sum(0.5 * spmeFourierStress), d_label->electrostatic->rElectrostaticInverseStress);
//  new_dw->put(sum_vartype(0.5 * spmeFourierEnergy), d_label->electrostaticReciprocalEnergyLabel);
//  new_dw->put(matrix_sum(0.5 * spmeFourierStress), d_label->electrostaticReciprocalStressLabel);

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

  // setup and copy data to-and-from this processor's portion of the global FFT array
  fftw_complex* localChunk = d_localFFTData.complexData;
  ptrdiff_t localN = d_localFFTData.numElements;
  ptrdiff_t localStart = d_localFFTData.startAddress;
  dblcomplex* nodeLocalData = recvbuf + localStart;
  size_t numElements = localN * d_kLimits[1] * d_kLimits[2];

  std::memcpy(localChunk, nodeLocalData, numElements * sizeof(dblcomplex));
  fftw_execute(d_backwardPlan);
  std::memcpy(nodeLocalData, localChunk, numElements * sizeof(dblcomplex));
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
    proc0thread0cout << "System Volume: " << d_systemVolume << std::endl;
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
                                    const IntVector& offset,
                                    const Uintah::Matrix3& inverseCell)
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
          m = m * inverseCell;
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
