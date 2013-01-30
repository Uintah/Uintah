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

#include <CCA/Components/MD/CenteredCardinalBSpline.h>
#include <CCA/Components/MD/MapPoint.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/SPME.h>
#include <CCA/Components/MD/SimpleGrid.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>
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

SPME::SPME(int _numGridPoints,
           int _numGhostCells,
           int _splineOrder,
           bool _polarizable,
           double _ewaldBeta) :
    numGridPoints(_numGridPoints),
      numGhostCells(_numGhostCells),
      splineOrder(_splineOrder),
      polarizable(_polarizable),
      ewaldBeta(_ewaldBeta)
{

}

void SPME::initialize()
{

}

void SPME::setup()
{

}

void SPME::calculate()
{

}

void SPME::finalize()
{

}

void SPME::performSPME(const MDSystem& system,
                       const Patch* patch,
                       ParticleSubset* pset)
{
  // Extract necessary information from system object
  Matrix3 inverseCell = system.getCellInverse();
  double systemVolume = system.getVolume();
  IntVector localExtents = patch->getCellHighIndex() - patch->getCellLowIndex();
  IntVector globalOffset(numGhostCells, numGhostCells, numGhostCells);
  IntVector globalExtents(numGhostCells, numGhostCells, numGhostCells);

  SimpleGrid<double> fStressPre(localExtents, globalOffset, numGhostCells);
  SimpleGrid<double> fTheta(localExtents, globalOffset, numGhostCells);
  fStressPre.initialize(0.0);
  fTheta.initialize(0.0);

  std::vector<double> M1(localExtents.x()), M2(localExtents.y()), M3(localExtents.z());

  M1 = generateMPrimeVector(localExtents.x(), globalOffset.x(), globalExtents.x());
  M2 = generateMPrimeVector(localExtents.y(), globalOffset.y(), globalExtents.y());
  M3 = generateMPrimeVector(localExtents.z(), globalOffset.z(), globalExtents.z());

  // Box dimensions have changed since last integration time step; we need to update B and C
  if (system.newBox()) {
    SimpleGrid<double> fBGrid(localExtents, globalOffset, numGhostCells);
    SimpleGrid<double> fCGrid(localExtents, globalOffset, numGhostCells);
    fBGrid.initialize(0.0);
    fCGrid.initialize(0.0);

    calculateStaticGrids(localExtents, globalOffset, system, fBGrid, fCGrid, fStressPre, splineOrder, M1, M2, M3);

    // Check input parameters, might just need Inverse Cell, System Volume instead of whole MD_System
    //  Also probably need to fix this routine up again for current object model

//    fTheta = fBGrid * fCGrid;
  }
  std::vector<std::vector<MapPoint> > localGridMap;
  CenteredCardinalBSpline spline;
  localGridMap = createChargeMap(pset, spline);

  // Begin main SPME loop;
  SimpleGrid<double> Q;

  // set up FFT routine back/forward transform auxiliary data here
  fftw_complex forwardTransformData, backwardTransformData;

  bool converged = false;     // Calculate at least once
  Matrix3 StressTensorLocal = Matrix3(0);  // Holds this patches contribution to the overall stress tensor
  double EnergyLocal = 0;       // Holds this patches local contribution to the energy

  while (!converged) {    // Iterate over this subset until charge convergence
    mapChargeToGrid(localGridMap, pset);
//    Q.inPlaceFFT_RealToFourier(forwardTransformData);
//    Q.multiplyInPlace(fTheta);
//    EnergyLocal = Q.CalculateEnergyAndStress(M1, M2, M3, StressTensorLocal, fStressPre, fTheta);

    // Note: Justin - Can we extract energy easily from stress tensor?  If it's just tr(StressTensor) or somesuch, it would be preferable
    //   to have this routine pass back only the stress tensor.

    // NYI:  Q.CalculatePolarization(CurrentPolarizationGrid,OldPolarizationGrid);

    converged = true;  // No convergence if no polarization
    // if (polarizable) { converged=CurrentPolarizationGrid.CheckConvergence(OldPolarizationGrid); }
  }
  // We need to calculate new forces on particles while preserving old particle values
  mapForceFromGrid(localGridMap, pset);

  // We need to accumulate EnergyLocal and StressLocal here, as well as the NewParticleList with their associated forces on return
  return;
}

std::vector<std::vector<MapPoint> > SPME::createChargeMap(ParticleSubset* pset,
                                                          CenteredCardinalBSpline& spline)
{

}

std::vector<Point> SPME::calcReducedCoords(const std::vector<Point>& localRealCoordinates,
                                           const MDSystem& system)
{
  std::vector<Point> localReducedCoords;
  size_t idx;
  Point coord;  // Fractional coordinates; 3 - vector

  // bool Orthorhombic; true if simulation cell is orthorhombic, false if it's generic
  size_t numParticles = localRealCoordinates.size();
  Matrix3 inverseBox = system.getCellInverse();        // For generic coordinate systems
  if (!system.isOrthorhombic()) {
    for (idx = 0; idx < numParticles; ++idx) {
      coord = localRealCoordinates[idx];                   // Get non-ghost particle coordinates for this cell
      coord = (inverseBox * coord.asVector()).asPoint();   // InverseBox is a 3x3 matrix so this is a matrix multiplication = slow
      localReducedCoords.push_back(coord);                 // Reduced non-ghost particle coordinates for this cell
    }
  } else {
    for (idx = 0; idx < numParticles; ++idx) {
      coord = localRealCoordinates[idx];        // Get non-ghost particle coordinates for this cell
      coord(0) *= inverseBox(0, 0);
      coord(1) *= inverseBox(1, 1);
      coord(2) *= inverseBox(2, 2);               // 6 Less multiplications and additions than generic above
      localReducedCoords.push_back(coord);      // Reduced, non-ghost particle coordinates for this cell
    }
  }
  return localReducedCoords;
}

//No ghost points
void SPME::calculateStaticGrids(const IntVector& gridExtents,
                                const IntVector& offset,
                                const MDSystem& system,
                                SimpleGrid<double>& fBGrid,
                                SimpleGrid<double>& fCGrid,
                                SimpleGrid<double>& fStressPre,
                                int splineOrder,
                                const std::vector<double>& M1,
                                const std::vector<double>& M2,
                                const std::vector<double>& M3)
{
  Matrix3 inverseUnitCell;
  double ewaldBeta;
  IntVector subGridOffset, K;
  IntVector halfGrid = gridExtents / IntVector(2, 2, 2);
  inverseUnitCell = system.getCellInverse();

  const std::vector<double> ordinalSpline = calculateOrdinalSpline(splineOrder - 1, splineOrder);

  std::vector<dblcomplex> b1, b2, b3;

  // Generate vectors of b_i (=exp(i*2*Pi*(n-1)m_i/K_i)*sum_(k=0..p-2)M_n(k+1)exp(2*Pi*k*m_i/K_i)
  b1 = generateBVector(gridExtents.x(), M1, K.x(), splineOrder, &ordinalSpline);
  b2 = generateBVector(gridExtents.y(), M2, K.y(), splineOrder, &ordinalSpline);
  b3 = generateBVector(gridExtents.z(), M3, K.z(), splineOrder, &ordinalSpline);

  // Use previously calculated vectors to calculate our grids
  double PI, PI2, invBeta2, volFactor;

  PI = acos(-1.0);
  PI2 = PI * PI;
  invBeta2 = 1.0 / (ewaldBeta * ewaldBeta);
  volFactor = 1.0 / (PI * system.getVolume());

  size_t extentsX = gridExtents.x();
  size_t extentsY = gridExtents.y();
  size_t extentsZ = gridExtents.z();
  for (size_t kX = 0; kX < extentsX; ++kX) {
    for (size_t kY = 0; kY < extentsY; ++kY) {
      for (size_t kZ = 0; kZ < extentsZ; ++kZ) {

        fBGrid(kX, kY, kZ) = norm(b1[kX]) * norm(b2[kY]) * norm(b3[kZ]);  // Calculate B

        if (kX != 0 && kY != 0 && kZ != 0) {  // Calculate C and stress pre-multiplication factor
          Vector m(M1[kX], M2[kY], M3[kZ]), M;  //
          double M2, Factor;

          M = m * inverseUnitCell;
          M2 = M.length2();
          Factor = PI2 * M2 * invBeta2;

          fCGrid(kX, kY, kZ) = volFactor * exp(-Factor) / M2;
          fStressPre(kX, kY, kZ) = 2 * (1 + Factor) / M2;
        }
      }
    }
  }
  fCGrid(0, 0, 0) = 0.0;
  fStressPre(0, 0, 0) = 0.0;  // Exceptional values
}

SimpleGrid<double>& SPME::mapChargeToGrid(const std::vector<std::vector<MapPoint> > gridMap,
                                          const ParticleSubset* globalParticleList)
{

}

SimpleGrid<double>& SPME::mapForceFromGrid(const std::vector<std::vector<MapPoint> > gridMap,
                                           ParticleSubset* globalParticleList)
{

}

SimpleGrid<double> SPME::fC(const IntVector& gridExtents,
                            const IntVector& gridOffset,
                            const int numGhostCells,
                            const MDSystem& system)
{
  SimpleGrid<double> C(gridExtents, gridOffset, numGhostCells);
  Matrix3 inverseCell;
  double ewaldBeta;

  inverseCell = system.getCellInverse();
  IntVector halfGrid = gridExtents / IntVector(2, 2, 2);

  double PI = acos(-1.0);
  double PI2 = PI * PI;
  double invBeta2 = 1.0 / (ewaldBeta * ewaldBeta);
  double volFactor = 1.0 / (PI * system.getVolume());

  Vector M;
  int extentX = gridExtents.x();
  for (int m1 = 0; m1 < extentX; ++m1) {
    M[0] = m1;
    if (m1 > halfGrid.x()) {
      M[0] -= gridExtents.x();
    }
    int extentY = gridExtents.y();
    for (int m2 = 0; m2 < extentY; ++m2) {
      M[1] = m2;
      if (m2 > halfGrid.y()) {
        M[1] -= gridExtents.y();
      }
      int extentZ = gridExtents.z();
      for (int m3 = 0; m3 < extentZ; ++m3) {
        M[2] = m3;
        if (m3 > halfGrid.z()) {
          M[2] -= gridExtents.z();
        }
        // Calculate C point values
        if ((m1 != 0) && (m2 != 0) && (m3 != 0)) {  // Discount C(0,0,0)
          Vector tempM = M * inverseCell;
          double M2 = tempM.length2();
          double val = volFactor * exp(-PI2 * M2 * invBeta2) / M2;
          C(m1, m2, m3) = val;
        }
      }
    }
  }
  return C;
}

SimpleGrid<dblcomplex> SPME::fB(const IntVector& gridExtents,
                                const MDSystem& system,
                                int splineOrder)
{
  Matrix3 InverseCell;
  SimpleGrid<dblcomplex> B;

  InverseCell = system.getCellInverse();
  IntVector halfGrid = gridExtents / IntVector(2, 2, 2);
  Vector inverseGrid = Vector(1.0, 1.0, 1.0) / gridExtents;

  double PI = acos(-1.0);
  double orderM12PI = 2.0 * PI * (splineOrder - 1);

  vector<dblcomplex> b1(gridExtents.x()), b2(gridExtents.y()), b3(gridExtents.z());
  vector<double> ordinalSpline(splineOrder - 1);

  // Calculates Mn(0)..Mn(n-1)
  ordinalSpline = calculateOrdinalSpline(splineOrder - 1, splineOrder);

  // Calculate k_i = m_i/K_i
  for (int m1 = 0; m1 < gridExtents.x(); ++m1) {
    double kX = m1;
    if (m1 > halfGrid.x()) {
      kX = m1 - gridExtents.x();
    }
    kX /= gridExtents.x();
    dblcomplex num = dblcomplex(cos(orderM12PI * kX), sin(orderM12PI * kX));
    dblcomplex denom = ordinalSpline[0];  //
    for (int k = 1; k < splineOrder - 1; ++k) {
      denom += ordinalSpline[k] * dblcomplex(cos(orderM12PI * kX * k), sin(orderM12PI * kX * k));
    }
    b1[m1] = num / denom;
  }

  for (int m2 = 0; m2 < gridExtents.y(); ++m2) {
    double kY = m2;
    if (m2 > halfGrid.y()) {
      kY = m2 - gridExtents.y();
    }
    kY /= gridExtents.y();
    dblcomplex num = dblcomplex(cos(orderM12PI * kY), sin(orderM12PI * kY));
    dblcomplex denom = ordinalSpline[0];  //
    for (int k = 1; k < splineOrder - 1; ++k) {
      denom += ordinalSpline[k] * dblcomplex(cos(orderM12PI * kY * k), sin(orderM12PI * kY * k));
    }
    b2[m2] = num / denom;
  }

  for (int m3 = 0; m3 < gridExtents.z(); ++m3) {
    double kZ = m3;
    if (m3 > halfGrid.z()) {
      kZ = m3 - gridExtents.y();
    }
    kZ /= gridExtents.y();
    dblcomplex num = dblcomplex(cos(orderM12PI * kZ), sin(orderM12PI * kZ));
    dblcomplex denom = ordinalSpline[0];  //
    for (int k = 1; k < splineOrder - 1; ++k) {
      denom += ordinalSpline[k] * dblcomplex(cos(orderM12PI * kZ * k), sin(orderM12PI * kZ * k));
    }
    b3[m3] = num / denom;
  }

  // Calculate B point values
  for (int m1 = 0; m1 < gridExtents.x(); ++m1) {
    for (int m2 = 0; m2 < gridExtents.x(); ++m2) {
      for (int m3 = 0; m3 < gridExtents.x(); ++m3) {
        B(m1, m2, m3) = norm(b1[m1]) * norm(b2[m2]) * norm(b3[m3]);
      }
    }
  }
  return B;
}

vector<double> SPME::calculateOrdinalSpline(int orderMinusOne,
                                            int splineOrder)
{

}

std::vector<dblcomplex> SPME::generateBVector(int numPoints,
                                              const std::vector<double>& M,
                                              int max,
                                              int splineOrder,
                                              const std::vector<double>* splineCoeff)
{
  double PI = acos(-1.0);
  double orderM12PI = (splineOrder - 1) * 2.0 * PI;

  std::vector<dblcomplex> B(numPoints);
  for (int idx = 0; idx < numPoints; ++idx) {
    double k = M[idx] / max;
    dblcomplex numerator = dblcomplex(cos(orderM12PI * k), sin(orderM12PI * k));
    dblcomplex denominator;
    for (int p = 0; p < splineOrder - 1; ++p) {
      double k1 = M[idx] / max;
      denominator += (*splineCoeff)[p] * dblcomplex(cos(orderM12PI * k1 * p), sin(orderM12PI * k1 * p));
    }
    B[idx] = numerator / denominator;
  }
  return B;
}
