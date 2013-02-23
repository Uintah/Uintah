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

#include <CCA/Components/MD/SPME.h>
#include <CCA/Components/MD/CenteredCardinalBSpline.h>
#include <CCA/Components/MD/MapPoint.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/SimpleGrid.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
#include <complex>

#include <sci_values.h>
#include <sci_defs/fftw_defs.h>

using namespace Uintah;
using namespace SCIRun;

SPME::SPME()
{

}

SPME::~SPME()
{
}

SPME::SPME(const MDSystem* system,
           const double ewaldBeta,
           const bool isPolarizable,
           const double tolerance,
           const SCIRun::IntVector& kLimits,
           const int splineOrder) :
    ewaldBeta(ewaldBeta), polarizable(isPolarizable), polarizationTolerance(tolerance), kLimits(kLimits)
{
  interpolatingSpline = CenteredCardinalBSpline(splineOrder);
  electrostaticMethod = Electrostatics::SPME;

  // Initialize and check for proper construction
  SCIRun::IntVector localGridSize = localGridExtents + localGhostPositiveSize + localGhostNegativeSize;
  SimpleGrid<complex<double> > Q(localGridSize, localGridOffset, localGhostNegativeSize, localGhostPositiveSize);
  Q.initialize(complex<double>(0.0, 0.0));
}

std::vector<dblcomplex> SPME::generateBVector(const std::vector<double>& mFractional,
                                              const int initialVectorIndex,
                                              const int localGridExtent,
                                              const CenteredCardinalBSpline& interpolatingSpline) const
{
  double PI = acos(-1.0);
  double twoPI = 2.0 * PI;
  double orderM12PI = twoPI * (interpolatingSpline.getOrder() - 1);

  int halfSupport = interpolatingSpline.getHalfSupport();
  std::vector<dblcomplex> b(localGridExtent);
  std::vector<double> zeroAlignedSpline = interpolatingSpline.evaluate(0);

  double* localMFractional = mFractional[initialVectorIndex];  // Reset MFractional zero so we can index into it negatively
  for (int Index = 0; Index < localGridExtent; ++Index) {
    double internal = twoPI * localMFractional[Index];
    // Formula looks significantly different from given SPME for offset splines.
    //   See Essmann et. al., J. Chem. Phys. 103 8577 (1995). for conversion, particularly formula C3 pt. 2 (paper uses pt. 4)
    dblcomplex phi_N = 0.0;
    for (int denomIndex = -halfSupport; denomIndex <= halfSupport; ++denomIndex) {
      phi_N += dblcomplex(cos(internal * denomIndex), sin(internal * denomIndex));
    }
    b[Index] = 1.0 / phi_N;
  }
  return b;
}

SimpleGrid<double> SPME::calculateBGrid(const SCIRun::IntVector& localExtents,
                                        const SCIRun::IntVector& globalOffset) const
{
  size_t Limit_Kx = kLimits.x();
  size_t Limit_Ky = kLimits.y();
  size_t Limit_Kz = kLimits.z();

  std::vector<double> mf1 = SPME::generateMFractionalVector(Limit_Kx, interpolatingSpline);
  std::vector<double> mf2 = SPME::generateMFractionalVector(Limit_Ky, interpolatingSpline);
  std::vector<double> mf3 = SPME::generateMFractionalVector(Limit_Kz, interpolatingSpline);

  // localExtents is without ghost grid points
  std::vector<dblcomplex> b1 = generateBVector(mf1, globalOffset.x(), localExtents.x(), interpolatingSpline);
  std::vector<dblcomplex> b2 = generateBVector(mf2, globalOffset.y(), localExtents.y(), interpolatingSpline);
  std::vector<dblcomplex> b3 = generateBVector(mf3, globalOffset.z(), localExtents.z(), interpolatingSpline);

  SimpleGrid<double> BGrid(localExtents, globalOffset, 0);  // No ghost cells; internal only

  size_t XExtents = localExtents.x();
  size_t YExtents = localExtents.y();
  size_t ZExtents = localExtents.z();

  int XOffset = globalOffset.x();
  int YOffset = globalOffset.y();
  int ZOffset = globalOffset.z();

  for (size_t kX = 0; kX < XExtents; ++kX) {
    for (size_t kY = 0; kY < YExtents; ++kY) {
      for (size_t kZ = 0; kZ < ZExtents; ++kZ) {
        BGrid(kX, kY, kZ) = norm(b1[kX + XOffset]) * norm(b2[kY + YOffset]) * norm(b3[kZ + ZOffset]);
      }
    }
  }
  return BGrid;
}

SimpleGrid<double> SPME::calculateCGrid(const SCIRun::IntVector& extents,
                                        const SCIRun::IntVector& offset) const
{
  std::vector<double> mp1 = SPME::generateMPrimeVector(kLimits.x(), interpolatingSpline);
  std::vector<double> mp2 = SPME::generateMPrimeVector(kLimits.y(), interpolatingSpline);
  std::vector<double> mp3 = SPME::generateMPrimeVector(kLimits.z(), interpolatingSpline);

  size_t xExtents = extents.x();
  size_t yExtents = extents.y();
  size_t zExtents = extents.z();

  int xOffset = offset.x();
  int yOffset = offset.y();
  int zOffset = offset.z();

  double PI = acos(-1.0);
  double PI2 = PI * PI;
  double invBeta2 = 1.0 / (ewaldBeta * ewaldBeta);
  double invVolFactor = 1.0 / (systemVolume * PI);

  SimpleGrid<double> CGrid(extents, offset, 0);  // No ghost cells; internal only
  for (size_t kX = 0; kX < xExtents; ++kX) {
    for (size_t kY = 0; kY < yExtents; ++kY) {
      for (size_t kZ = 0; kZ < zExtents; ++kZ) {
        if (kX != 0 || kY != 0 || kZ != 0) {
          SCIRun::Vector m(mp1[kX + xOffset], mp2[kY + yOffset], mp3[kZ + zOffset]);

          m = m * inverseUnitCell;

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

SimpleGrid<Matrix3> SPME::calculateStressPrefactor(const SCIRun::IntVector& extents,
                                                   const SCIRun::IntVector& offset)
{
  std::vector<double> mp1 = SPME::generateMPrimeVector(kLimits.x(), interpolatingSpline);
  std::vector<double> mp2 = SPME::generateMPrimeVector(kLimits.y(), interpolatingSpline);
  std::vector<double> mp3 = SPME::generateMPrimeVector(kLimits.z(), interpolatingSpline);

  size_t XExtents = extents.x();
  size_t YExtents = extents.y();
  size_t ZExtents = extents.z();

  int XOffset = offset.x();
  int YOffset = offset.y();
  int ZOffset = offset.z();

  double PI = acos(-1.0);
  double PI2 = PI * PI;
  double invBeta2 = 1.0 / (ewaldBeta * ewaldBeta);

  SimpleGrid<Matrix3> StressPre(extents, offset, 0);  // No ghost cells; internal only
  for (size_t kX = 0; kX < XExtents; ++kX) {
    for (size_t kY = 0; kY < YExtents; ++kY) {
      for (size_t kZ = 0; kZ < ZExtents; ++kZ) {
        if (kX != 0 || kY != 0 || kZ != 0) {
          SCIRun::Vector m(mp1[kX + XOffset], mp2[kY + YOffset], mp3[kZ + ZOffset]);
          m = m * inverseUnitCell;
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

          StressPre(kX, kY, kZ) = localStressContribution;
        }
      }
    }
  }
  StressPre(0, 0, 0) = Matrix3(0);
  return StressPre;
}

// Interface implementations
void SPME::initialize(const MDSystem* system,
                      const PatchSubset* patches,
                      const MaterialSubset* matls)
{
  // We call SPME::initialize from the constructor or if we've somehow maintained our object across a system change

  // Note:  I presume the indices are the local cell indices without ghost cells
  localGridExtents = patch->getCellHighIndex() - patch->getCellLowIndex();

  // Holds the index to map the local chunk of cells into the global cell structure
  localGridOffset = patch->getCellLowIndex();

  // Get useful information from global system descriptor to work with locally.
  unitCell = system.getUnitCell();
  inverseUnitCell = system.getInverseCell();
  systemVolume = system.getVolume();

  // Alan:  Not sure what the correct syntax is here, but the idea is that we'll store the number of ghost cells
  //          along each of the min/max boundaries.  This lets us differentiate should we need to for centered and
  //          left/right shifted splines
  localGhostPositiveSize = patch->getExtraCellHighIndex() - patch->getCellHighIndex();
  localGhostNegativeSize = patch->getCellLowIndex() - patch->getExtraCellLowIndex();
  return;
}

void SPME::setup()
{
  // We should only have to do this if KLimits or the inverse cell changes
  // Calculate B and C
  SimpleGrid<double> fBGrid = calculateBGrid(localGridExtents, localGridOffset);
  SimpleGrid<double> fCGrid = calculateCGrid(localGridExtents, localGridOffset);
  // Composite B and C into Theta
  size_t xExtent = localGridExtents.x();
  size_t yExtent = localGridExtents.y();
  size_t zExtent = localGridExtents.z();
  for (size_t xidx = 0; xidx < xExtent; ++xidx) {
    for (size_t yidx = 0; yidx < yExtent; ++yidx) {
      for (size_t zidx = 0; zidx < zExtent; ++zidx) {
        fTheta(xidx, yidx, zidx) = fBGrid(xidx, yidx, zidx) * fCGrid(xidx, yidx, zidx);
      }
    }
  }
  stressPrefactor = calculateStressPrefactor(localGridExtents, localGridOffset);
}

void SPME::calculate()
{

//  // Patch dependent quantities
//  SCIRun::IntVector localGridExtents;             //!< Number of grid points in each direction for this patch
//  SCIRun::IntVector localGridOffset;              //!< Grid point index of local 0,0,0 origin in global coordinates
//  SCIRun::IntVector localGhostPositiveSize;       //!< Number of ghost cells on positive boundary
//  SCIRun::IntVector localGhostNegativeSize;       //!< Number of ghost cells on negative boundary

//  // Actually holds the data we're working with per patch
//  SimpleGrid<double> fTheta;            //!<
//  SimpleGrid<Matrix3> stressPrefactor;  //!<
//  SimpleGrid<complex<double> > Q;       //!<

  // Note:  Must run SPME->setup() each time there is a new box/K grid mapping (e.g. every step for NPT)
  //          This should be checked for in the system electrostatic driver

  std::vector<MapPoint> GridMap = SPME::generateChargeMap(pset, interpolatingSpline);
  bool converged = false;
  int numIterations = 0;
  while (!converged && (numIterations < MaxIterations)) {
    SPME::mapChargeToGrid(GridMap, pset, interpolatingSpline.getHalfSupport());  // Calculate Q(r)

    // Map the local patch's charge grid into the global grid and transform
    SPME::GlobalMPIReduceChargeGrid(GHOST::AROUND);  //Ghost points should get transferred here
    SPME::ForwardTransformGlobalChargeGrid();  // Q(r) -> Q*(k)
    // Once reduced and transformed, we need the local grid re-populated with Q*(k)
    SPME::MPIDistributeLocalChargeGrid(GHOST::NONE);

    // Multiply the transformed Q out
    size_t XExtent = localGridExtents.x();
    size_t YExtent = localGridExtents.y();
    size_t ZExtent = localGridExtents.z();
    double localEnergy = 0.0;  //Maybe should be global?
    Matrix3 localStress(0.0);  //Maybe should be global?
    for (size_t kX = 0; kX < XExtent; ++kX) {
      for (size_t kY = 0; kY < YExtent; ++kY) {
        for (size_t kZ = 0; kZ < ZExtent; ++kZ) {
          complex<double> GridValue = Q(kX, kY, kZ);
          Q(kX, kY, kZ) = GridValue * conj(GridValue) * fTheta(kX, kY, kZ);  // Calculate (Q*Q^)*(B*C)
          localEnergy += Q(kX, kY, kZ);
          localStress += Q(kX, kY, kZ) * stressPrefactor(kX, kY, kZ);
        }
      }
    }

    // Transform back to real space
    SPME::GlobalMPIReduceChargeGrid(GHOST::NONE);  //Ghost points should NOT get transferred here
    SPME::ReverseTransformGlobalChargeGrid();
    SPME::MPIDistributeLocalChargeGrid(GHOST::AROUND);

    //  This may need to be before we transform the charge grid back to real space if we can calculate
    //    polarizability from the fourier space component
    converged = true;
    if (polarizable) {
      // calculate polarization here
      // if (RMSPolarizationDifference > PolarizationTolerance) { ElectrostaticsConverged = false; }
      std::cerr << "Error:  Polarization not currently implemented!";
    }
    // Sanity check - Limit maximum number of polarization iterations we try
    ++numIterations;
  }
  SPME::GlobalReduceEnergy();
  SPME::GlobalReduceStress();  //Uintah framework?

}

void SPME::finalize()
{
  SPME::mapForceFromGrid(pset, ChargeMap);  // Calculate electrostatic contribution to f_ij(r)
  //Reduction for Energy, Pressure Tensor?
  // Something goes here, though I'm not sure what
  // Output?
}

std::vector<MapPoint> SPME::generateChargeMap(ParticleSubset* pset,
                                              CenteredCardinalBSpline& spline)
{
  size_t MaxParticleIndex = pset->numParticles();
  std::vector<MapPoint> ChargeMap;
  // Loop through particles
  for (size_t chargeIndex = 0; chargeIndex < MaxParticleIndex; ++chargeIndex) {
    int ParticleID = pset[chargeIndex]->GetParticleID();
    pset->getPointer()
    SCIRun::Vector ParticleGridCoordinates;

    //Calculate reduced coordinates of point to recast into charge grid
    ParticleGridCoordinates = ((pset[chargeIndex]->GetParticleCoordinates()).AsVector()) * inverseUnitCell;
    // ** NOTE: JBH --> We may want to do this with a bit more thought eventually, since multiplying by the InverseUnitCell
    //                  is expensive if the system is orthorhombic, however it's not clear it's more expensive than dropping
    //                  to call MDSystem->IsOrthorhombic() and then branching the if statement appropriately.

    // This bit is tedious since we don't have any cross-pollination between type Vector and type IntVector.
    // Should we put that in (requires modifying Uintah framework).
    SCIRun::Vector KReal, splineValues;
    SCIRun::IntVector particleGridOffset;
    for (size_t Index = 0; Index < 3; ++Index) {
      KReal[Index] = static_cast<double>(kLimits[Index]);  // For some reason I can't construct a Vector from an IntVector -- Maybe we should fix that instead?
      ParticleGridCoordinates[Index] *= KReal[Index];         // Recast particle into charge grid based representation
      particleGridOffset[Index] = static_cast<int>(ParticleGridCoordinates[Index]);  // Reference grid point for particle
      splineValues[Index] = ParticleGridCoordinates[Index] - particleGridOffset[Index];  // spline offset for spline function
    }
    vector<double> XSplineArray = spline.evaluate(splineValues[0]);
    vector<double> YSplineArray = spline.evaluate(splineValues[1]);
    vector<double> ZSplineArray = spline.evaluate(splineValues[2]);

    vector<double> XSplineDeriv = spline.derivative(splineValues[0]);
    vector<double> YSplineDeriv = spline.derivative(splineValues[1]);
    vector<double> ZSplineDeriv = spline.derivative(splineValues[2]);

//    MapPoint CurrentMapPoint(ParticleID, ParticleGridOffset, XSplineArray, YSplineArray, ZSplineArray);

    SimpleGrid<double> ChargeGrid(XSplineArray, YSplineArray, ZSplineArray, particleGridOffset, 0);
    SimpleGrid<SCIRun::Vector> ForceGrid(XSplineDeriv.size(), YSplineDeriv.size(), ZSplineDeriv.size(), particleGridOffset, 0);
    size_t XExtent = XSplineArray.size();
    size_t YExtent = YSplineArray.size();
    size_t ZExtent = ZSplineArray.size();
    for (size_t XIndex = 0; XIndex < XExtent; ++XIndex) {
      for (size_t YIndex = 0; YIndex < YExtent; ++YIndex) {
        for (size_t ZIndex = 0; ZIndex < ZExtent; ++ZIndex) {
          ChargeGrid(XIndex, YIndex, ZIndex) = XSplineArray[XIndex] * YSplineArray[YIndex] * ZSplineArray[ZIndex];
          ForceGrid(XIndex, YIndex, ZIndex) = SCIRun::Vector(XSplineDeriv[XIndex], YSplineDeriv[YIndex], ZSplineDeriv[ZIndex]);
        }
      }
    }
    MapPoint CurrentMapPoint(ParticleID, particleGridOffset, ChargeGrid, ForceGrid);
    ChargeMap.push_back(CurrentMapPoint);
  }
  return ChargeMap;
}

void SPME::mapChargeToGrid(const std::vector<MapPoint>& GridMap,
                           ParticleSubset* pset,
                           int HalfSupport)
{
  size_t MaxParticleIndex = pset->numParticles();
  Q.initialize(0.0);  // Reset charges before we start adding onto them.
  for (size_t ParticleIndex = 0; ParticleIndex < MaxParticleIndex; ++ParticleIndex) {
    double Charge = pset[ParticleIndex]->GetCharge();

    // !FIXME Alan
    return (&ChargeGrid);
    SimpleGrid<double> ChargeMap = GridMap[ParticleIndex]->ChargeMapAddress();  //FIXME -- return reference, don't copy

    SCIRun::IntVector QAnchor = ChargeMap.getOffset();  // Location of the 0,0,0 origin for the charge map grid
    SCIRun::IntVector SupportExtent = ChargeMap.getExtents();  // Extents of the charge map grid
    for (int XMask = -HalfSupport; XMask <= HalfSupport; ++XMask) {
      for (int YMask = -HalfSupport; YMask <= HalfSupport; ++YMask) {
        for (int ZMask = -HalfSupport; ZMask <= HalfSupport; ++ZMask) {
          Q(QAnchor.x() + XMask, QAnchor.y() + YMask, QAnchor.z() + ZMask) += Charge
                                                                              * ChargeMap(XMask + HalfSupport, YMask + HalfSupport,
                                                                                          ZMask + HalfSupport);
        }
      }
    }

  }
}

void SPME::mapForceFromGrid(const std::vector<MapPoint>& gridMap,
                            ParticleSubset* pset,
                            int halfSupport)
{
  constParticleVariable<Vector> pforce;
  constParticleVariable<double> pcharge;
  old_dw->get(pforce, pForceLabel, lpset);

  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
    particleIndex pidx = *iter;

    SimpleGrid<SCIRun::Vector> forceMap = gridMap[pidx]->ForceMapAddress();  // FIXME -- return reference, don't copy
    SCIRun::Vector newForce = pset[pidx]->GetForce();
    SCIRun::IntVector QAnchor = forceMap.getOffset();  // Location of the 0,0,0 origin for the force map grid
    SCIRun::IntVector supportExtent = forceMap.getExtents();  // Extents of the force map grid

    for (int xmask = -halfSupport; xmask <= halfSupport; ++xmask) {
      for (int ymask = -halfSupport; ymask <= halfSupport; ++ymask) {
        for (int zmask = -halfSupport; zmask <= halfSupport; ++zmask) {
          SCIRun::Vector currentForce;
          currentForce = forceMap(xmask + halfSupport, ymask + halfSupport, zmask + halfSupport)
                         * Q(QAnchor.x() + xmask, QAnchor.y() + ymask, QAnchor.z() + zmask);
          newForce += currentForce;
        }
      }
    }
    pset[pidx]->SetForce(newForce);
  }
}

