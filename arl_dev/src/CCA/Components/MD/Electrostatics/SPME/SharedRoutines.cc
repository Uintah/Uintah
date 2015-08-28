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
 * SPME_newShared.cc
 *
 *  Created on: May 21, 2014
 *      Author: jbhooper
 */

#include <Core/Thread/Thread.h>

#include <CCA/Components/MD/Electrostatics/Ewald/InverseSpace/SPME/SPME.h>

using namespace Uintah;
//
///**
// * @brief Generates reduced Fourier grid vector. Generates the vector of values i/K_i for i = 0...K-1
// * @param KMax - Maximum number of grid points for direction
// * @param InterpolatingSpline - CenteredCardinalBSpline that determines the number of wrapping points necessary
// * @return std::vector<double> of the reduced coordinates for the local grid along the input lattice direction
// */
//inline void generateMFractionalVector(std::vector<double>& mFractional,
//                                      size_t kMax) const
//{
//  double kMaxInv = 1.0 / static_cast<double>(kMax);
//  for (size_t idx = 0; idx < kMax; ++idx) {
//    mFractional[idx] = static_cast<double>(idx) * kMaxInv;
//  }
//}

void SPME::generatePointScreeningMultipliers(const double& radius, // |R_ij|
                                                   double& B0,
                                                   double& B1,
                                                   double& B2,
                                                   double& B3)
{
  double beta_r = d_ewaldBeta * radius;
  double inv_r = 1.0 / radius;
  double inv_r2 = inv_r * inv_r;
  double twoBeta2 = 2.0 * d_ewaldBeta * d_ewaldBeta;
  double invBetaRootPi = 1.0 / (d_ewaldBeta * MDConstants::rootPI);
  double expNegBeta2r2 = exp(-beta_r * beta_r);

  B0 = erfc(beta_r)*inv_r;
  double rightSide = twoBeta2 * invBetaRootPi * expNegBeta2r2;
  B1 = (B0 + rightSide)*inv_r2;
  rightSide *= twoBeta2;
  B2 = (3.0 * B1 + rightSide)*inv_r2;
  rightSide *= twoBeta2;
  B3 = (5.0 * B2 + rightSide)*inv_r2;
}

void SPME::generateTholeScreeningMultipliers(const double& a_Thole,
                                             const double& sqrt_alphai_alphaj,
                                             const double& radius,
                                                   double& B1,
                                                   double& B2,
                                                   double& B3)
{
  double A = a_Thole / sqrt_alphai_alphaj;
  double r2 = radius * radius;
  double invR2 = 1.0 / (r2);
  double denominator = invR2 / radius;
  double u = radius * r2 * A;

  double expTerm = exp(-u);
  double polyTerm = 1.0;
  B1 = (1.0 - polyTerm * expTerm) * denominator; // lambda_3
  denominator *= invR2;                          //    1 - exp(-au^3)

  polyTerm += u;
  B2 = (1.0 - polyTerm * expTerm) * denominator; // lambda_5
  denominator *= invR2;                          //    1 - (1 + au^3)exp(-au^3)
  polyTerm += 0.6 * u * u;
  B3 = (1.0 - polyTerm * expTerm) * denominator; // lambda_7
  return;                                        //    1 -
                                                 //     (1+au^3  (3/5)a^2u^6)*
                                                 //      exp(-au^3)

}

void SPME::generateDipoleFunctionalTerms(const double&         q_i,
                                         const double&         q_j,
                                         const SCIRun::Vector& mu_i,
                                         const SCIRun::Vector& mu_j,
                                         const SCIRun::Vector& r_ij,
                                               double&         mu_jDOTr_ij,
                                               double&         G0,
                                               double&         G1_mu_q,
                                               double&         G1_mu_mu,
                                               double&         G2,
                                               SCIRun::Vector& gradG0,
                                               SCIRun::Vector& gradG1,
                                               SCIRun::Vector& gradG2)
{
  // GX and gradGX defined in:
  // Toukmaji et. al., J. Chem. Phys. 113(24), 10913-10927 (2000)
  // Eq. 2.10 and inline after Eq. 2.18
  double mu_iDOTmu_j = Dot(mu_i,mu_j);
  double mu_iDOTr_ij = Dot(mu_i,r_ij);
  mu_jDOTr_ij = Dot(mu_j,r_ij);  // Passed back for field calculation

  G0       =  q_i * q_j;
  G1_mu_q  =  q_j * mu_iDOTr_ij - q_i * mu_jDOTr_ij;
  G1_mu_mu =  mu_iDOTmu_j;
  G2       = -mu_iDOTr_ij * mu_jDOTr_ij;

  gradG0 =  MDConstants::V_ZERO;
  gradG1 =  q_i*mu_j - q_j * mu_i;
  gradG2 =  mu_i*(mu_jDOTr_ij) + mu_j*(mu_iDOTr_ij);

  return;
}

void SPME::generateBVectorChunk(std::vector<dblcomplex>&    bVector,
                                const int                   m_initial,
                                const int                   localGridExtent,
                                const int                   K) const
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

void SPME::generateMPrimeChunk(std::vector<double>& mPrimeChunk,
                               const int            m_initial,
                               const int            localGridExtent,
                               const int            K) const
{
  int halfMax = K/2;
  for (int Index = 0; Index < localGridExtent; ++Index) {
    mPrimeChunk[Index]  = static_cast<double> (m_initial + Index);
  }

  // Remaining points to subtract K from
  int splitStart = (halfMax + 1) - m_initial;
  if (splitStart < 0) splitStart = 0;
  for (int Index = splitStart; Index < localGridExtent; ++Index) {
    mPrimeChunk[Index] -= static_cast<double> (K);
  }
}

void SPME::calculateBGrid(SimpleGrid<double>&   BGrid) const
{
//  size_t limit_Kx = d_kLimits.x();
//  size_t limit_Ky = d_kLimits.y();
//  size_t limit_Kz = d_kLimits.z();

  IntVector gridExtent = BGrid.getExtents();
  IntVector gridOffset = BGrid.getOffset();

//  std::vector<double> mf1(limit_Kx);
//  std::vector<double> mf2(limit_Ky);
//  std::vector<double> mf3(limit_Kz);
//  SPME::generateMFractionalVector(mf1, limit_Kx);
//  SPME::generateMFractionalVector(mf2, limit_Ky);
//  SPME::generateMFractionalVector(mf3, limit_Kz);

  size_t xExtents = gridExtent.x();
  size_t yExtents = gridExtent.y();
  size_t zExtents = gridExtent.z();

  // gridExtent is without ghost grid points
  std::vector<dblcomplex> b1(xExtents);
  std::vector<dblcomplex> b2(yExtents);
  std::vector<dblcomplex> b3(zExtents);
  SPME::generateBVectorChunk(b1, gridOffset[0], gridExtent[0], d_kLimits[0]);
  SPME::generateBVectorChunk(b2, gridOffset[1], gridExtent[1], d_kLimits[1]);
  SPME::generateBVectorChunk(b3, gridOffset[2], gridExtent[2], d_kLimits[2]);

  for (size_t kX = 0; kX < xExtents; ++kX) {
    for (size_t kY = 0; kY < yExtents; ++kY) {
      for (size_t kZ = 0; kZ < zExtents; ++kZ) {
        BGrid(kX, kY, kZ) = norm(b1[kX]) * norm(b2[kY]) * norm(b3[kZ]);
      }
    }
  }
}

void SPME::calculateCGrid(SimpleGrid<double>& CGrid,
                          CoordinateSystem*   coordSys) const
{
  IntVector gridExtent  = CGrid.getExtents();
  IntVector gridOffset  = CGrid.getOffset();

  int xExtents = gridExtent[0];
  int yExtents = gridExtent[1];
  int zExtents = gridExtent[2];

  int xOffset = gridOffset[0];
  int yOffset = gridOffset[1];
  int zOffset = gridOffset[2];

//  std::vector<double> mp1(d_kLimits.x());
//  std::vector<double> mp2(d_kLimits.y());
//  std::vector<double> mp3(d_kLimits.z());
//  SPME::generateMPrimeVector(mp1, d_kLimits.x());
//  SPME::generateMPrimeVector(mp2, d_kLimits.y());
//  SPME::generateMPrimeVector(mp3, d_kLimits.z());

  std::vector<double> mp1Chunk(xExtents);
  std::vector<double> mp2Chunk(yExtents);
  std::vector<double> mp3Chunk(zExtents);

  generateMPrimeChunk(mp1Chunk, gridOffset[0], gridExtent[0], d_kLimits[0]);
  generateMPrimeChunk(mp2Chunk, gridOffset[1], gridExtent[1], d_kLimits[1]);
  generateMPrimeChunk(mp3Chunk, gridOffset[2], gridExtent[2], d_kLimits[2]);


  double PI2 = MDConstants::PI2;
  double invBeta2 = 1.0 / (d_ewaldBeta * d_ewaldBeta);
  double invVolFactor = 1.0 / (coordSys->getCellVolume() * MDConstants::PI);
  double PI2OverBeta2 = PI2 * invBeta2;



  SCIRun::Vector mReduced;
  for (int kX = 0; kX < xExtents; ++kX) {
    for (int kY = 0; kY < yExtents; ++kY) {
      for (int kZ = 0; kZ < zExtents; ++kZ) {
        if (kX + xOffset != 0 || kY + yOffset != 0 || kZ + zOffset != 0) {
          SCIRun::Vector m(mp1Chunk[kX], mp2Chunk[kY], mp3Chunk[kZ]);
          coordSys->toReduced(m,mReduced);
          double M2 = mReduced.length2();
          double factor = M2 * PI2OverBeta2;
          CGrid(kX, kY, kZ) = invVolFactor * exp(-factor) / M2;
//          std::cerr << "C(" << kX << "," << kY << "," << kZ << "): " << CGrid(kX, kY, kZ) << std::endl;
        }
      }
    }
  }
  if (xOffset == 0 && yOffset == 0 && zOffset == 0) {
    CGrid(xOffset, yOffset, zOffset) = 0;
  }
}

void SPME::calculateStressPrefactor(SimpleGrid<Matrix3>*    stressPrefactor,
                                    CoordinateSystem*       coordSys)
{
  IntVector gridExtent    =   stressPrefactor->getExtents();
  IntVector gridOffset    =   stressPrefactor->getOffset();

  int xExtents = gridExtent[0];
  int yExtents = gridExtent[1];
  int zExtents = gridExtent[2];

  int xOffset = gridOffset[0];
  int yOffset = gridOffset[1];
  int zOffset = gridOffset[2];

  std::vector<double> mp1Chunk(xExtents);
  std::vector<double> mp2Chunk(yExtents);
  std::vector<double> mp3Chunk(zExtents);

  generateMPrimeChunk(mp1Chunk, gridOffset[0], gridExtent[0], d_kLimits[0]);
  generateMPrimeChunk(mp2Chunk, gridOffset[1], gridExtent[1], d_kLimits[1]);
  generateMPrimeChunk(mp3Chunk, gridOffset[2], gridExtent[2], d_kLimits[2]);

//
//  std::vector<double> mp1(d_kLimits.x());
//  std::vector<double> mp2(d_kLimits.y());
//  std::vector<double> mp3(d_kLimits.z());
//
//  SPME::generateMPrimeVector(mp1, d_kLimits.x());
//  SPME::generateMPrimeVector(mp2, d_kLimits.y());
//  SPME::generateMPrimeVector(mp3, d_kLimits.z());

  double PI2 = MDConstants::PI2;
  double invBeta2 = 1.0 / (d_ewaldBeta * d_ewaldBeta);

//  int xOffset = offset.x();
//  int yOffset = offset.y();
//  int zOffset = offset.z();

  SCIRun::Vector mReduced;
  for (int kX = 0; kX < xExtents; ++kX) {
    for (int kY = 0; kY < yExtents; ++kY) {
      for (int kZ = 0; kZ < zExtents; ++kZ) {
        if ((kX + xOffset)!= 0 || (kY + yOffset)!= 0 || (kZ + zOffset)!= 0) {
          Vector m(mp1Chunk[kX], mp2Chunk[kY], mp3Chunk[kZ]);
          coordSys->toReduced(m,mReduced);
          double M2 = mReduced.length2();

          Matrix3 localStressContribution(-2.0*(1.0 + PI2*M2*invBeta2)/M2);

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
  if (xOffset == 0 && yOffset == 0 && zOffset == 0) {
    (*stressPrefactor)(xOffset, yOffset, zOffset) = Matrix3(0.0);
  }
}

void SPME::initializeLocalStorage(const ProcessorGroup* pg,
                                  const PatchSubset*    patches,
                                  const MaterialSubset* materials,
                                  DataWarehouse*        oldDW,
                                  DataWarehouse*        newDW,
                                  const MDLabel*        label)
{
  // Runs once per patch to avoid race conditions
  d_Q_nodeLocal->initialize(dblcomplex(0.0, 0.0));
  d_Q_nodeLocalScratch->initialize(dblcomplex(0.0, 0.0));
  SoleVariable<int> initializeQDummy;
  newDW->put(initializeQDummy,
             label->SPME_dep->dInitializeQ);
}

void SPME::reduceNodeLocalQ(const ProcessorGroup* /*pg*/,
                            const PatchSubset*      patches,
                            const MaterialSubset*   materials,
                            DataWarehouse*        /*oldDW*/,
                            DataWarehouse*          newDW,
                            const MDLabel*          label)
{
  size_t numPatches = patches->size();

  for (size_t p = 0; p < numPatches; ++p) {

    const Patch* patch = patches->get(p);
    // Extract current spmePatch

    const int patchID = patch->getID();

    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patchID)->second;

    SimpleGrid<dblcomplex>* Q_patchLocal = currentSPMEPatch->getQ();

    // Location of the local Q patches 0,0,0 origin
    IntVector localQOffset = Q_patchLocal->getOffset();

    // Size of the current local Q subset (with ghosts)
    IntVector localQExtent = Q_patchLocal->getExtentWithGhost();

    // Q_nodeLocal is a copy of the ENTIRE Q grid; this may be imfeasible
    // for high grid density and may therefore require a new approach.
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
      // We should probably put some IntVector wrapping into the
      // coordinate system to handle this across periodicity too. FIXME JBH 5/22/14
      // So long as x_global is always >= 0
      x_global %= xMax;
//      if (x_global >= xMax) { x_global -= xMax; }
//      if (x_global < 0)     { x_global += xMax; }
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_local = ymask;
        int y_global = yBase + ymask;
        // So long as y_global is always >= 0
        y_global %= yMax;
//        if (y_global >= yMax) { y_global -= yMax; }
//        if (y_global < 0)     { y_global += yMax; }
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_local = zmask;
          int z_global = zBase + zmask;
          // So long as z_global is always >= 0
          z_global %= zMax;
//          if (z_global >= zMax) { z_global -= zMax; }
//          if (z_global < 0)     { z_global += zMax; }

          // Recall d_Q_nodeLocal is a complete copy of the Q grid
          // for reduction across MPI threads
          (*d_Q_nodeLocal)(x_global,y_global,z_global)
                   += (*Q_patchLocal)(x_local,y_local,z_local);

        } // xmask
      } // ymask
    } // zmask
    d_Qlock.unlock();
    // Set dummy variables for taskgraph dependence
    size_t numAtomTypes = materials->size();
    for (size_t dummyType = 0; dummyType < numAtomTypes; ++dummyType) {
      int atomType = materials->get(dummyType);
      PerPatch<int> reduceQDummy;
      newDW->put(reduceQDummy,
                 label->SPME_dep->dReduceNodeLocalQ,
                 atomType,
                 patch);
    }

  } // Patch
}

void SPME::distributeNodeLocalQ(const ProcessorGroup* /*pg*/,
                                const PatchSubset*      patches,
                                const MaterialSubset*   materials,
                                DataWarehouse*        /*oldDW*/,
                                DataWarehouse*          newDW,
                                const MDLabel*          label)
{
  size_t numPatches = patches->size();

  for (size_t p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);

    // Extract current spmePatch
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

    SimpleGrid<dblcomplex>* Q_patchLocal = currentSPMEPatch->getQ();

    // Location of the local Q patches 0,0,0 origin
    IntVector localQOffset = Q_patchLocal->getOffset();

    // Size of the current local Q subset (with ghosts)
    IntVector localQExtent = Q_patchLocal->getExtentWithGhost();

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
      // Should work as long as x_global is always >= 0
      x_global %= xMax;
//      if (x_global >= xMax) { x_global -= xMax; }
//      if (x_global < 0)     { x_global += xMax; }
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_local = ymask;
        int y_global = yBase + ymask;
        y_global %= yMax;
//        if (y_global >= yMax) { y_global -= yMax; }
//        if (y_global < 0)     { y_global += yMax; }
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_local = zmask;
          int z_global = zBase + zmask;
          z_global %= zMax;
//          if (z_global >= zMax) { z_global -= zMax; }
//          if (z_global < 0)     { z_global += zMax; }
          // Recall d_Q_nodeLocal is a complete copy of the Q grid
          // for reduction across MPI threads
          (*Q_patchLocal)(x_local,y_local,z_local) =
              (*d_Q_nodeLocal)(x_global,y_global,z_global);
        } // zmask
      } // ymask
    } // xmask
    // Dummy material loop for dummy dependency variable
    size_t numAtomTypes = materials->size();
    for (size_t currType = 0; currType < numAtomTypes; ++currType) {
      int atomType = materials->get(currType);
      PerPatch<int> distributeQDummy;
      newDW->put(distributeQDummy,
                 label->SPME_dep->dDistributeNodeLocalQ,
                 atomType,
                 patch);

    }
  } // patch loop
}

// Fourier transform related routines
void SPME::transformRealToFourier(const ProcessorGroup* pg,
                                  const PatchSubset*    patches,
                                  const MaterialSubset* materials,
                                  DataWarehouse*        oldDW,
                                  DataWarehouse*        newDW,
                                  const MDLabel*        label)
{
  // Do the homebrew MPI_Allreduce on our local Q-grid before we do the
  // forward FFT
  int           totalElements   = d_kLimits(0) * d_kLimits(1) * d_kLimits(2);
  dblcomplex*   sendbuf         = d_Q_nodeLocal->getDataPtr();
  dblcomplex*   recvbuf         = d_Q_nodeLocalScratch->getDataPtr();
  MPI_Allreduce(sendbuf,
                recvbuf,
                totalElements,
                MPI_DOUBLE_COMPLEX,
                MPI_SUM,
                pg->getComm());

  // setup and copy data to-and-from this processor's portion of the global
  // FFT array
  ptrdiff_t     localStart  = d_localFFTData.startAddress;
  ptrdiff_t     localN      = d_localFFTData.numElements;

  // d_localFFTData.complexData is the memory for just this processor's
  // chunk of the FFT data.  It's size is not exactly the same size as the
  // pure data due to FFTW overhead requirements, so we can't just point
  // the FFTW invocation at the pure data chunk in d_Q_nodeLocal safely

  // Where FFTW wants the data to be
  fftw_complex* localChunk      = d_localFFTData.complexData;
  // Where we actually keep our data
  dblcomplex*   nodeLocalData   = recvbuf + localStart;

  size_t chunkSize = localN * d_kLimits[1] * d_kLimits[2] * sizeof(dblcomplex);
  std::memcpy(localChunk,                           //Copy to here
              nodeLocalData,                        //Copy from here
              chunkSize);                           //Copy this much

  fftw_execute(d_forwardPlan);                      // Voodoo happens here

  std::memcpy(nodeLocalData,                        //Copy to here
              localChunk,                           //Copy from here
              chunkSize);                           //Copy this much

  SoleVariable<int> xformRtoF;
  newDW->put(xformRtoF, label->SPME_dep->dTransformRealToFourier);
}

void SPME::calculateInFourierSpace(const ProcessorGroup*    pg,
                                   const PatchSubset*       patches,
                                   const MaterialSubset*    materials,
                                   DataWarehouse*           oldDW,
                                   DataWarehouse*           newDW,
                                   const MDLabel*           label)
{
  // TODO:  Determine if we actually need to do this to satisfy the dependency
  //        requirement.
  SoleVariable<int> xformRtoF;
  newDW->get(xformRtoF, label->SPME_dep->dTransformRealToFourier);

  Uintah::Matrix3   spmeFourierStress(0.0);
  double    spmeFourierEnergy = 0.0;

  size_t numPatches     = patches->size();
  size_t numAtomTypes   = materials->size();

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch*    patch = patches->get(patchIndex);

    SPMEPatch*  currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;
    IntVector   localExtents     = currentSPMEPatch->getLocalExtents();

    // We should probably remove this material loop; all materials are smeared
    // together at this point as a total charge distribution! FIXME
    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex) {
      int atomType = materials->get(typeIndex);

      SimpleGrid<double>*             fTheta = currentSPMEPatch->getTheta();
      SimpleGrid<Matrix3>*   stressPrefactor =
                                     currentSPMEPatch->getStressPrefactor();

      // Multiply the transformed Q by B*C to get Theta
// Stubbing out interface to swap dimensions for FFT efficiency
//    int systemMaxKDimension = d_system->getMaxKDimensionIndex();
//    int systemMidKDimension = d_system->getMidDimensionIndex();
//    int systemMinKDimension = d_system->getMinDimensionIndex();
      int systemMaxKDimension = 0; // X for now
      int systemMidKDimension = 1; // Y for now
      int systemMinKDimension = 2; // Z for now

      size_t xExtent = localExtents[systemMaxKDimension];
      size_t yExtent = localExtents[systemMidKDimension];
      size_t zExtent = localExtents[systemMinKDimension];

      // local-to-global Q coordinate translation.
      // This eliminates the copy to-and-from local SPMEPatches

      // Location of the local Q patches 0,0,0 origin
      IntVector localQOffset = currentSPMEPatch->getQ()->getOffset();
      int xBase = localQOffset[0];
      int yBase = localQOffset[1];
      int zBase = localQOffset[2];

      for (size_t kX = 0; kX < xExtent; ++kX) {
        for (size_t kY = 0; kY < yExtent; ++kY) {
          for (size_t kZ = 0; kZ < zExtent; ++kZ) {
            int x_global = xBase + kX;
            int y_global = yBase + kY;
            int z_global = zBase + kZ;

            // Calculate (Q*Q^)*(B*C)

            // Store Q
            std::complex<double> gridValue =
                (*d_Q_nodeLocalScratch)(x_global, y_global, z_global);

            // Q' = Q(BC) (for reverse transformation for force calc)
            (*d_Q_nodeLocalScratch)(x_global, y_global, z_global) *=
                (*fTheta)(kX, kY, kZ);

            // E = QBCQ^
            spmeFourierEnergy += std::abs(
                (*d_Q_nodeLocalScratch)(x_global, y_global, z_global)
                * conj(gridValue));

            // S^0 = Q(BC)Q^S' (S' is stress tensor prefactor)
            // For dipole systems there is an additional factor, S^1 which
            //  is calculated in the UpdateFieldsAndStress method.
            // It must be deferred because it relies on the converged value
            //  of the dipole, which we do not know at this point in execution.
            spmeFourierStress += std::abs(
                (*d_Q_nodeLocalScratch)(x_global, y_global, z_global)
                * conj(gridValue)) * (*stressPrefactor)(kX, kY, kZ);
          } // kZ
        } // kY
      } // kX
    }  // end atomType loop

//    if (spme_dbg.active()) {
//      coutLock.lock();
//      std::cout.setf(std::ios_base::left);
//      std::cout << std::setw(30) << Thread::self()->getThreadName();
//      std::cout << "Uintah thread ID: " << std::setw(4) << Thread::self()->myid()
//                << "Thread group: " <<  std::setw(10) <<Thread::self()->getThreadGroup()
//                << "Patch: " <<  std::setw(4) <<patch->getID()
//                << "Fourier-Energy: " << spmeFourierEnergy << std::endl;
//      coutLock.unlock();
//    }
    // dummies for remainder of inner subloop
    size_t numAtomTypes = materials->size();
    for (size_t currType = 0; currType < numAtomTypes; ++currType) {
      int atomType = materials->get(currType);
      PerPatch<int> calcInFourier;
      newDW->put(calcInFourier,
                 label->SPME_dep->dCalculateInFourierSpace,
                 atomType,
                 patch);

    }

  }  // end SPME Patch loop
  double chargeUnitNorm = 332.063712;
  // put updated values for reduction variables into the DW
  newDW->put(sum_vartype(0.5 * spmeFourierEnergy * chargeUnitNorm),
      label->electrostatic->rElectrostaticInverseEnergy);
  newDW->put(matrix_sum(0.5 * spmeFourierStress * chargeUnitNorm),
      label->electrostatic->rElectrostaticInverseStress);

}

void SPME::transformFourierToReal(const ProcessorGroup* pg,
                                  const PatchSubset*    patches,
                                  const MaterialSubset* materials,
                                  DataWarehouse*        oldDW,
                                  DataWarehouse*        newDW,
                                  const MDLabel*        label)
{
  // Do the homebrew MPI_Allreduce on our local Q-scratch-grid before we do the
  // reverse FFT
  int           totalElements   = d_kLimits(0) * d_kLimits(1) * d_kLimits(2);
  dblcomplex*   sendbuf         = d_Q_nodeLocalScratch->getDataPtr();
  dblcomplex*   recvbuf         = d_Q_nodeLocal->getDataPtr();
  MPI_Allreduce(sendbuf,
                recvbuf,
                totalElements,
                MPI_DOUBLE_COMPLEX,
                MPI_SUM,  // If forces aren't coming out right, look here!  FIXME
                pg->getComm());

  // FIXME -->  I --think-- the FFT should come before the reduction, but I'm not sure.
  //            Will leave things as they are for now.
  // setup and copy data to-and-from this processor's portion of the global FFT array
  fftw_complex* localChunk = d_localFFTData.complexData;
  ptrdiff_t localN = d_localFFTData.numElements;
  ptrdiff_t localStart = d_localFFTData.startAddress;
  dblcomplex* nodeLocalData = recvbuf + localStart;
  size_t numElements = localN * d_kLimits[1] * d_kLimits[2];

  std::memcpy(localChunk, nodeLocalData, numElements * sizeof(dblcomplex));
  fftw_execute(d_backwardPlan);
  std::memcpy(nodeLocalData, localChunk, numElements * sizeof(dblcomplex));

  SoleVariable<int> xformFtoR;
  newDW->put(xformFtoR, label->SPME_dep->dTransformFourierToReal);
}


