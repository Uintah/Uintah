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

#include <CCA/Components/MD/Electrostatics.h>
#include <CCA/Components/MD/SimpleGrid.h>
#include <CCA/Components/MD/MDSystem.h>
#include <Core/Grid/Patch.h>
#include <Core/Math/UintahMiscMath.h>
#include <Core/Math/MiscMath.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>

#include <iostream>
#include <complex>

#include <sci_values.h>

using namespace Uintah;
using namespace SCIRun;

Electrostatics::Electrostatics()
{

}

Electrostatics::~Electrostatics()
{

}

Electrostatics::Electrostatics(ElectroStaticsType type,
                               IntVector _globalExtents,
                               IntVector _globalOffset,
                               IntVector _localExtents) :
    electroStaticsType(type), globalExtents(_globalExtents), globalOffset(_globalOffset), localExtents(_localExtents)
{

}

void Electrostatics::performSPME(const MDSystem& system,
                                 const PatchSet* patches)
{
  // Extract necessary information from system object
  Matrix3 inverseCell = system.getCellInverse();
  double systemVolume = system.getVolume();
  ParticleList = LocalPatch->ExtractParticleSubset();

  cdGrid fStressPre(localExtents, globalOffset), fTheta(localExtents, globalOffset);

  vector<double> M1(localExtents.x()), M2(localExtents.y()), M3(localExtents.z());

  M1 = generateMVector(localExtents.x(), globalOffset.x(), globalExtents.x());
  M2 = generateMVector(localExtents.y(), globalOffset.y(), globalExtents.y());
  M3 = generateMVector(localExtents.z(), globalOffset.z(), globalExtents.z());

  // Box dimensions have changed since last integration time step; we need to update B and C
  if (system.newBox()) {
    SimpleGrid fBGrid(localExtents, globalOffset), fCGrid(localExtents, globalOffset);
    CalculateStaticGrids(localExtents, globalOffset, MD_System, fBGrid, fCGrid, fStressPre, M1, M2, M3);
    // Check input parameters, might just need Inverse Cell, System Volume instead of whole MD_System
    //  Also probably need to fix this routine up again for current object model

    fTheta = fBGrid * fCGrid;
  }
  SimpleGrid<vector<MapPoint<double> > > LocalGridMap;
  LocalGridMap = createChargeGridMap(localExtents, globalOffset, globalExtents, ParticleList/*other variables needed?*/);

  // Begin main SPME loop;
  SPME_Grid Q;
  // set up FFT routine back/forward transform auxiliary data here
  FFT_Data_Type ForwardTransformData, BackwardTransformData;

  bool converged = false;     // Calculate at least once
  Matrix3 StressTensorLocal = Matrix3(0);  // Holds this patches contribution to the overall stress tensor
  double EnergyLocal = 0;       // Holds this patches local contribution to the energy

  while (!converged) {    // Iterate over this subset until charge convergence
    Q.MapChargeToGrid(LocalGridMap, ParticleList);
    Q.InPlaceFFT_RealToFourier(ForwardTransformData);
    Q.MultiplyInPlace(fTheta);
    EnergyLocal = Q.CalculateEnergyAndStress(M1, M2, M3, StressTensorLocal, fStressPre, fTheta);
    // Note: Justin - Can we extract energy easily from stress tensor?  If it's just tr(StressTensor) or somesuch, it would be preferable
    //   to have this routine pass back only the stress tensor.
    // NYI:  Q.CalculatePolarization(CurrentPolarizationGrid,OldPolarizationGrid);
    converged = true;  // No convergence if no polarization
    // if (polarizable) { converged=CurrentPolarizationGrid.CheckConvergence(OldPolarizationGrid); }
  }
  // We need to calculate new forces on particles while preserving old particle values
  Q.MapForceFromGrid(LocalGridMap, NewParticleList);

  // We need to accumulate EnergyLocal and StressLocal here, as well as the NewParticleList with their associated forces on return
  return;
}

SPMEGrid<double> Electrostatics::initializeSPME(const MDSystem& system,
                                                const IntVector& ewaldMeshLimits,
                                                const Matrix3& cellInverse,
                                                const Matrix3& cell,
                                                const double& ewaldScale,
                                                int splineOrder)
{
  IntVector K, HalfK;
  K = ewaldMeshLimits;
  HalfK = K / IntVector(2, 2, 2);

  Vector KInverse = Vector(1.0, 1.0, 1.0) / K.asVector();

  double PI = acos(-1.0);
  double PiSquared = PI * PI;
  double InvBetaSquared = 1.0 / (ewaldScale * ewaldScale);

  SPMEGrid<double> B, C;

  // Calculate the C array
  if (system.isOrthorhombic()) {
    double CellVolume, As1, As2, As3;
    CellVolume = cell(0, 0) * cell(1, 1) * cell(2, 2);
    As1 = cellInverse(0, 0);
    As2 = cellInverse(1, 1);
    As3 = cellInverse(2, 2);

    double CPreFactor = 1.0 / (PI * CellVolume);

    for (int l1 = 0; l1 <= HalfK[0]; ++l1) {
      double m1 = l1 * As1;
      double m1OverK1 = m1 * KInverse[0];
      double MSquared = m1 * m1;
      for (int l2 = 0; l2 <= HalfK[1]; ++l2) {
        double m2 = l2 * As2;
        double m2OverK2 = m2 * KInverse[1];
        MSquared += m2 * m2;
        for (int l3 = 0; l3 <= HalfK[2]; ++l3) {
          double m3 = l3 * As3;
          double m3OverK3 = m3 * KInverse[3];
          MSquared += m3 * m3;

          double C = CPreFactor * exp(-PiSquared * MSquared * InvBetaSquared) / MSquared;
        }
      }
    }
  }

  // Assumes the vectors in cell are stored in rows
  Vector A1(cell(0, 0), cell(0, 1), cell(0, 2));
  Vector A2(cell(1, 0), cell(1, 1), cell(1, 2));
  Vector A3(cell(2, 0), cell(2, 1), cell(2, 2));

  // Assumes the vectors in CellInverse are stored in columns
  Vector AS1(cell(0, 0), cell(1, 0), cell(2, 0));
  Vector AS2(cell(0, 1), cell(1, 1), cell(2, 1));
  Vector AS3(cell(0, 2), cell(1, 2), cell(2, 2));

  // C(m1,m2,m3) = (1/PI*V)*(exp(-PI^2*M^2/Beta^2)/M^2)
  double cellVolume = Dot(Cross(A1, A2), A3);
  double CPreFactor = 1.0 / (PI * cellVolume);

  int K1 = ewaldMeshLimits[0];
  int halfK1 = K1 / 2;
  int K2 = ewaldMeshLimits[1];
  int halfK2 = K2 / 2;
  int K3 = ewaldMeshLimits[2];
  int halfK3 = K3 / 2;

  SPMEGrid<double> fieldGrid;
  double OneOverBeta2 = 1.0 / (ewaldScale * ewaldScale);

  // Orthorhombic
  if (system.isOrthorhombic()) {
    for (int m1 = 0; m1 <= halfK1; ++m1) {
      for (int m2 = 0; m2 <= halfK2; ++m2) {
        for (int m3 = 0; m3 <= halfK3; ++m3) {
          double MSquared = m1 * AS1 * m1 * AS1 + m2 * AS2 * m2 * AS2 + m3 * AS3 * m3 * AS3;
          fieldGrid((double)m1, (double)m2, (double)m3) = CPreFactor * exp(-PiSquared * MSquared * OneOverBeta2) / MSquared;
        }
      }
    }
  }
  return fieldGrid;
}

SPMEGridMap<double> Electrostatics::createChargeGridMap(const SPMEGrid<std::complex<double> >& grid,
                                                        const MDSystem& system,
                                                        const Patch* patch)
{
  // Note:  SubGridOffset maps the offset of the current patch's subgrid to the global grid numbering scheme.
  //        For example, a patch that iterated from global grid point 3,4,5 to 7,8,9 would have a SubGridOffset
  //        of:  {3,4,5}.

  IntVector EwaldMeshLimits = grid->GetMeshLimits();
  IntVector SplineOrder = SPME_GlobalGrid->GetSplineOrder();

  Vector PatchOffset = patch->get->SpatialOffset();
  Vector PatchExtent = CurrentPatch->SpatialExtent();  // HighCorner-LowCorner, where HighCorner is the max(X,Y,Z) of the subspace, LowCorner is
                                                       //   min(X,Y,Z)

  ParticleIterator ParticleList = CurrentPatch->ParticleVector();

  Matrix3 cellInverse = system.getCellInverse();

  Vector InverseMeshLimits = 1.0 / EwaldMeshLimits;

  IntVector SubGridIndexOffset, SubGridIndexExtent;
  {
    // Contain these temp variables
    Vector TempOffset = PatchOffset * cellInverse;
    Vector TempExtent = PatchExtent * cellInverse;

    for (size_t Ind = 0; Ind < 3; ++Ind) {
      SubGridIndexOffset = floor(TempOffset[Ind] * static_cast<double>(EwaldMeshLimits[Ind]));  // Generate index offset to map local grid to global
      SubGridIndexExtent = floor(TempExtent[Ind] * static_cast<double>(EwaldMeshLimits[Ind]));  // Generate index count to determine # grid points inside patch
                                                                                                //   not including ghost grid points
    }
  }
  SPME_Grid_Map LocalGrid[SubGridIndexExtent[0] + SplineOrder][SubGridIndexExtent[1] + SplineOrder][SubGridIndexExtent[2]
                                                                                                    + SplineOrder];
  // 3D array of Vector<SPME_Map> type, messy data structure.  Suggestions?
  // Goal here is to create a sub-grid which maps to the patch local + extended "ghost" grid points, and to save the particle to grid charge mapping coefficients
  // so that we don't have to re-generate them again.  This step is essentially necessary every time step, or every time the global "Solve SPME Charge" routine
  // is done.

  int Global_Shift = 0;
  if (SystemData->Orthorhombic()) {  //  Take a few calculation shortcuts to save some matrix multiplication.  Worth it?  Not sure..
    Vector InverseGridMapping = EwaldMeshLimits * cellInverse;  // Orthorhomibc, so CellInverse is diagonal and we can pre-multiply

    for (ParticlePointer = ParticleList.begin(); ParticlePointer != ParticleList.end(); ++ParticlePointer) {
      Vector U_Current = ParticlePointer->GetCoordinateVector();                          // Extract coordinate vector from particle
      U_Current *= InverseGridMapping;                                                        // Convert to reduced coordinates
      IntVector CellIndex;
      for (size_t Ind = 0; Ind < 3; ++Ind) {
        CellIndex[Ind] = floor(U_Current[Ind]) - SubGridIndexOffset[Ind] + Global_Shift;
      }  // Vector floor + shift by element

      vector<Vector> Coeff_Array(SplineOrder), Deriv_Array(SplineOrder);
      CalculateSpline(U_Current, Coeff_Array, Deriv_Array, SplineOrder);
      for (int IndX = 0; IndX < SplineOrder; ++IndX) {
        for (int IndY = 0; IndY < SplineOrder; ++IndY) {
          for (int IndZ = 0; IndZ < SplineOrder; ++IndZ) {
            double Coefficient = Coeff_Array[IndX].x * Coeff_Array[IndY].y * Coeff_Array[IndZ].z;  // Calculate appropriate coefficient
            Vector Gradient(Deriv_Array[IndX].x, Deriv_Array[IndY].y, Deriv_Array[IndZ].z);  // Extract appropriate derivative vector
            (LocalGrid[CellIndex.x + IndX][CellIndex.y + IndY][CellIndex.z + IndZ]).AddMapPoint(ParticlePointer->GetGlobalHandle(),
                                                                                                Coefficient, Gradient);
          }
        }
      }
    }
  } else {
    for (ParticlePointer = ParticleList.begin(); ParticlePointer != ParticleList.end(); ++ParticlePointer) {
      Vector U_Current = ParticlePointer->GetCOordinateVector();  // Extract coordinate vector from particle
      U_Current *= cellInverse;                                    // Full matrix multiplication to get (X,Y,Z) for non-orthorhombic
      for (size_t Ind = 0; Ind < 3; ++Ind) {
        U_Current[Ind] *= EwaldMeshLimits[Ind];
      }
      for (size_t Ind = 0; Ind < 3; ++Ind) {
        CellIndex[Ind] = floor(U_Current[Ind]) + Global_Shift;
      }    // Vector floor + shift by element

      vector<Vector> Coeff_Array(SplineOrder), Deriv_Array(SplineOrder);
      CalculateSpline(U_Current, Coeff_Array, Deriv_Array, SplineOrder);
      for (int IndX = 0; IndX < SplineOrder; ++IndX) {
        for (int IndY = 0; IndY < SplineOrder; ++IndY) {
          for (int IndZ = 0; IndZ < SplineOrder; ++IndZ) {
            double Coefficient = Coeff_Array[IndX].x * Coeff_Array[IndY].y * Coeff_Array[IndZ].z;  // Calculate appropriate coefficient
            Vector Gradient(Deriv_Array[IndX].x, Deriv_Array[IndY].y, Deriv_Array[IndZ].z);  // Extract appropriate derivative vector
            (LocalGrid[CellIndex.x + IndX][CellIndex.y + IndY][CellIndex.z + IndZ]).AddMapPoint(ParticlePointer->GetGlobalHandle(),
                                                                                                Coefficient, Gradient);
          }
        }
      }
    }
  }
  return;
}

void Electrostatics::calculateStaticGrids(const IntVector& gridExtents,
                                          const IntVector& offset,
                                          const MDSystem& system,
                                          SimpleGrid<std::complex<double> >& fBGrid,
                                          SimpleGrid<double>& fCGrid,
                                          SimpleGrid<std::complex<double> >& fStressPre,
                                          Vector& M1,
                                          Vector& M2,
                                          Vector& M3)
{
  Matrix3 inverseUnitCell;
  double ewaldBeta;
  IntVector subGridOffset, K;

  IntVector halfGrid = gridExtents / IntVector(2, 2, 2);

  inverseUnitCell = system.getCellInverse();
  ewaldBeta = system.getEwaldBeta();

  std::vector<double> ordinalSpline(splineOrder - 1);
  ordinalSpline = calculateOrdinalSpline(splineOrder);

  std::vector<std::complex<double> > b1, b2, b3;
  // Generate vectors of b_i (=exp(i*2*Pi*(n-1)m_i/K_i)*sum_(k=0..p-2)M_n(k+1)exp(2*Pi*k*m_i/K_i)

  b1 = generateBVector(GridExtents.x(), M1, K.x(), splineOrder, ordinalSpline);
  b2 = generateBVector(GridExtents.y(), M1, K.y(), splineOrder, ordinalSpline);
  b3 = generateBVector(GridExtents.z(), M1, K.z(), splineOrder, ordinalSpline);

  // Use previously calculated vectors to calculate our grids
  double PI, PI2, invBeta2, volFactor;

  PI = acos(-1.0);
  PI2 = PI * PI;
  invBeta2 = 1.0 / (ewaldBeta * ewaldBeta);
  volFactor = 1.0 / (PI * system.getVolume());

  for (size_t kX = 0; kX < GridExtents.x(); ++kX) {
    for (size_t kY = 0; kY < GridExtents.y(); ++kY) {
      for (size_t kZ = 0; kZ < GridExtents.z(); ++kZ) {

        fB[kX][kY][kZ] = norm(b1[kX]) * norm(b2[kY]) * norm(b3[kZ]);  // Calculate B

        if (kX != kY != kZ != 0) {  // Calculate C and stress premultiplication factor
          Vector m(M1[kX], M2[kY], M3[kZ]), M;
          double M2, Factor;

          M = m * inverseUnitCell;
          M2 = M.length2();
          Factor = PI2 * M2 * invBeta2;

          fC[kX][kY][kZ] = VolFactor * exp(-Factor) / M2;
          StressPreMult[kX][kY][kZ] = 2 * (1 + Factor) / M2;
        }
      }
    }
  }
  fC[0][0][0] = 0.0;
  StressPremult[0][0][0] = 0.0;  // Exceptional values
}

std::vector<Point> Electrostatics::calcReducedCoords(const std::vector<Point>& localRealCoordinates,
                                                     const MDSystem& system,
                                                     const Transformation3D<std::complex<double> >& invertSpace)
{
  vector<Point> localReducedCoords;

  // bool Orthorhombic; true if simulation cell is orthorhombic, false if it's generic
  if (!system.isOrthorhombic())
    for (size_t Index = 0; Index < NumParticlesInCell; ++Index) {
      CoordType s;        // Fractional coordinates; 3 - vector
      s = ParticleList[Index].GetCoordinates();        // Get non-ghost particle coordinates for this cell
      s *= InverseBox;       // For generic coordinate systems; InverseBox is a 3x3 matrix so this is a matrix multiplication = slow
      localRealCoordinates.push_back(s);        // Reduced non-ghost particle coordinates for this cell
    }
  else {
    for (size_t Index = 0; Index < NumParticlesInCell; ++Index) {
      CoordType s;        // Fractional coordinates; 3-vector
      s = ParticleList[Index].GetCoordinates();        // Get non-ghost particle coordinates for this cell
      s(0) *= invertSpace(0, 0);
      s(1) *= invertSpace(1, 1);
      s(2) *= invertSpace(2, 2);        // 6 Less multiplications and additions than generic above
      localRealCoordinates.push_back(s);        // Reduced non-ghost particle coordinates for this cell
    }
  }

  return localReducedCoords;
}

SimpleGrid<double> Electrostatics::fC(const IntVector& gridExtents,
                                      const MDSystem& system)
{
  SimpleGrid<double> C;  // C(gridExtents);
  Matrix3 inverseCell;
  double ewaldBeta;

  inverseCell = system.getCellInverse();
  ewaldBeta = system.getEwaldBeta();

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
          double tempM = M * inverseCell;
          double M2 = tempM.length2(tempM);
          double val = volFactor * exp(-PI2 * M2 * invBeta2) / M2;
          C(m1, m2, m3) = val;
        }
      }
    }
  }
  return C;
}

SimpleGrid<std::complex<double> > Electrostatics::fB(const IntVector& gridExtents,
                                                     const MDSystem& system,
                                                     int splineOrder)
{
  Matrix3 InverseCell;
  SimpleGrid<std::complex<double> > B;

  InverseCell = system.getCellInverse();
  IntVector halfGrid = gridExtents / IntVector(2, 2, 2);
  Vector inverseGrid = Vector(1.0, 1.0, 1.0) / gridExtents;

  double PI = acos(-1.0);
  double orderM12PI = 2.0 * PI * (splineOrder - 1);

  vector<std::complex<double> > b1(gridExtents.x()), b2(gridExtents.y()), b3(gridExtents.z());
  vector<std::complex<double> > ordinalSpline(splineOrder - 1);

  // Calculates Mn(0)..Mn(n-1)
  ordinalSpline = calculateOrdinalSpline(splineOrder - 1, splineOrder);

  // Calculate k_i = m_i/K_i
  for (int m1 = 0; m1 < gridExtents.x(); ++m1) {
    double kX = m1;
    if (m1 > halfGrid.x()) {
      kX = m1 - gridExtents.x();
    }
    kX /= gridExtents.x();
    std::complex<double> num = std::complex<double>(cos(orderM12PI * kX), sin(orderM12PI * kX));
    std::complex<double> denom = ordinalSpline[0];  //
    for (int k = 1; k < splineOrder - 1; ++k) {
      denom += ordinalSpline[k] * std::complex<double>(cos(orderM12PI * kX * k), sin(orderM12PI * kX * k));
    }
    b1[m1] = num / denom;
  }

  for (int m2 = 0; m2 < gridExtents.y(); ++m2) {
    double kY = m2;
    if (m2 > halfGrid.y()) {
      kY = m2 - gridExtents.y();
    }
    kY /= gridExtents.y();
    std::complex<double> num = std::complex<double>(cos(orderM12PI * kY), sin(orderM12PI * kY));
    std::complex<double> denom = ordinalSpline[0];  //
    for (int k = 1; k < splineOrder - 1; ++k) {
      denom += ordinalSpline[k] * std::complex<double>(cos(orderM12PI * kY * k), sin(orderM12PI * kY * k));
    }
    b2[m2] = num / denom;
  }

  for (int m3 = 0; m3 < gridExtents.z(); ++m3) {
    double kZ = m3;
    if (m3 > halfGrid.z()) {
      kZ = m3 - gridExtents.y();
    }
    kZ /= gridExtents.y();
    std::complex<double> num = std::complex<double>(cos(orderM12PI * kZ), sin(orderM12PI * kZ));
    std::complex<double> denom = ordinalSpline[0];  //
    for (int k = 1; k < splineOrder - 1; ++k) {
      denom += ordinalSpline[k] * std::complex<double>(cos(orderM12PI * kZ * k), sin(orderM12PI * kZ * k));
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
}

vector<std::complex<double> > Electrostatics::calculateOrdinalSpline(int orderMinusOnesplineOrder,
                                                                     int splineOrder)
{

}

std::vector<std::complex<double> > Electrostatics::generateBVector(int points,
                                                                   const std::vector<double>& M,
                                                                   int max,
                                                                   int splineOrder,
                                                                   const std::vector<double>& splineCoeff)
{
  double PI = acos(-1.0);
  double orderM12PI = (splineOrder - 1) * 2.0 * PI;

  std::vector<std::complex<double> > b(points);
  for (int idx = 0; idx < points; ++idx) {
    double k = M[idx] / max;
    std::complex<double> numerator = std::complex<double>(cos(orderM12PI * k), sin(orderM12PI * k));
    std::complex<double> denominator;
    for (int p = 0; p < splineOrder - 1; ++p) {
//      double k1 = M[idx] / max;
      denominator += splineCoeff[p] * std::complex<double>(cos(orderM12PI * k1 * p), sin(orderM12PI * k1 * p));
    }
    b[idx] = numerator / denominator;
  }
  return b;
}
