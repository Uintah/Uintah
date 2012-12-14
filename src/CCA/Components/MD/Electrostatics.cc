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

void Electrostatics::performSPME(MDSystem& system,
                                 const PatchSet* patches)
{
//  // Extract necessary information from system object
//  Matrix3 InverseCell;
//  double  SystemVolume;
//
//   InverseCell = MD_System->GetUnitCellInverse();
//  SystemVolume = MD_System->GetUnitCellVolume();
//
//  // Extract SPME information from electrostatics subsystem object
//  IntVector SPMEGrid_Extents;
//  TotalSPMEGrid_Extents = SPME_Data->GetGridExtents();
//
//  // Extract spatial information from local patch data
//  IntVector LocalGridExtents, LocalGridOffset;
//  LocalGridExtents = LocalPatch->GetGridExtents();
//   LocalGridOffset = LocalPatch->GetGridOffset();
//      ParticleList = LocalPatch->ExtractParticleSubset();
//
//  cdGrid fStressPre(LocalGridExtents,LocalGridOffset), fTheta(LocalGridExtents,LocalGridOffset);
//
//  vector<double> M1(LocalGridExtents.x()),M2(LocalGridExtents.y()),M3(LocalGridExtents.z());
//
//  M1 = GenerateMVector(LocalGridExtents.x(),LocalGridOffset.x(),TotalSPMEGrid_Extents.x());
//  M2 = GenerateMVector(LocalGridExtents.y(),LocalGridOffset.y(),TotalSPMEGrid_Extents.y());
//  M3 = GenerateMVector(LocalGridExtents.z(),LocalGridOffset.z(),TotalSPMEGrid_Extents.z());
//
//  if (MD_System->NewBox()) { // Box dimensions have changed since last integration time step; we need to update B and C
//    SimpleGrid fBGrid(LocalGridExtents,LocalGridOffset), fCGrid(LocalGridExtents,LocalGridOffset);
//    CalculateStaticGrids(LocalGridExtents,LocalGridOffset,MD_System,fBGrid,fCGrid,fStressPre,M1,M2,M3);
//    // Check input parameters, might just need Inverse Cell, System Volume instead of whole MD_System
//    //  Also probably need to fix this routine up again for current object model
//
//    fTheta=fBGrid*fCGrid;
//  }
//  SimpleGrid<vector<MapPoint >> LocalGridMap;
//  LocalGridMap = CreateGridMap(LocalGridExtents,LocalGridOffset,TotalSPMEGrid_Extents,ParticleList/*other variables needed?*/);
//
//  // Begin main SPME loop;
//  SPME_Grid Q;
//  // set up FFT routine back/forward transform auxiliary data here
//  FFT_Data_Type ForwardTransformData,BackwardTransformData;
//
//  bool converged=false;     // Calculate at least once
//  Matrix StressTensorLocal=Matrix(0); // Holds this patches contribution to the overall stress tensor
//  double EnergyLocal=0;       // Holds this patches local contribution to the energy
//
//  while (!converged) {    // Iterate over this subset until charge convergence
//    Q.MapChargeToGrid(LocalGridMap, ParticleList);
//    Q.InPlaceFFT_RealToFourier(ForwardTransformData);
//    Q.MultiplyInPlace(fTheta);
//    EnergyLocal = Q.CalculateEnergyAndStress(M1,M2,M3,StressTensorLocal,fStressPre,fTheta);
//    // Note: Justin - Can we extract energy easily from stress tensor?  If it's just tr(StressTensor) or somesuch, it would be preferable
//    //   to have this routine pass back only the stress tensor.
//    // NYI:  Q.CalculatePolarization(CurrentPolarizationGrid,OldPolarizationGrid);
//    converged=true;  // No convergence if no polarization
//    // if (polarizable) { converged=CurrentPolarizationGrid.CheckConvergence(OldPolarizationGrid); }
//  }
//  // We need to calculate new forces on particles while preserving old particle values
//  Q.MapForceFromGrid(LocalGridMap, NewParticleList);
//
//  // We need to accumulate EnergyLocal and StressLocal here, as well as the NewParticleList with their associated forces on return
//  return;
}

SPMEGrid Electrostatics::initializeSPME(const IntVector& ewaldMeshLimits,
                                        const Matrix3& cellInverse,
                                        const Matrix3& cell,
                                        const double& ewaldScale,
                                        const int& splineOrder)
{
  IntVector K, HalfK;
  K = EwaldMeshLimits;
  HalfK = K / 2;

  Vector KInverse = 1.0 / K;

  double PiSquared = PI * PI;
  double InvBetaSquared = 1.0 / (EwaldScale * EwaldScale);

  SPME_Grid B, C;
  // Calculate the C array
  if (Orthorhombic) {
    double CellVolume, As1, As2, As3;
    CellVolume = Cell(0, 0) * Cell(1, 1) * Cell(2, 2);
    As1 = CellInverse(0, 0);
    As2 = CellInverse(1, 1);
    As3 = CellInverse(2, 2);

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

  Vector& A1, A2, A3;
  A1 = Cell.ExtractRow(0);  // Assumes the vectors in Cell are stored in rows
  A2 = Cell.ExtractRow(1);
  A3 = Cell.ExtractRow(2);

  Vector& As1, As2, As3;
  As1 = Cell.ExtractCol(0);  // Assumes the vectors in CellInverse are stored in columns
  As2 = Cell.ExtractCol(1);
  As3 = Cell.ExtractCol(2);

  double CellVolume = (A1.Cross(A2)).Dot(A3);
  // C(m1,m2,m3) = (1/PI*V)*(exp(-PI^2*M^2/Beta^2)/M^2)

  double CPreFactor = 1.0 / (PI * CellVolume);

  int K1 = EwaldMeshLimits[0];
  int HalfK1 = K1 / 2;
  int K2 = EwaldMeshLimits[1];
  int HalfK2 = K2 / 2;
  int K3 = EwaldMeshLimits[2];
  int HalfK3 = K3 / 2;

  SPME_Grid FieldGrid;
  double PiSquared = PI * PI;
  double OneOverBeta2 = 1.0 / (EwaldScale * EwaldScale);

  // Orthorhombic
  if (Orthorhombic) {
    for (int m1 = 0; m1 <= HalfK1; ++m1) {
      for (int m2 = 0; m2 <= HalfK2; ++m2) {
        for (int m3 = 0; m3 <= HalfK3; ++m3) {
          double MSquared = m1 * As1 * m1 * As2 + m2 * As2 * m2 * As2 + m3 * As3 * m3 * As3;
          FieldGrid[m1][m2][m3] = CPreFactor * exp(-PiSquared * MSquared * OneOverBeta2) / MSquared;
        }
      }
    }
  }
return NULL
}

SPMEGridMap<double> Electrostatics::createChargeGridMap(SPMEGrid<std::complex<double> >& grid,
                                                        MDSystem& system,
                                                        const Patch* patch)
{
//  // Note:  SubGridOffset maps the offset of the current patch's subgrid to the global grid numbering scheme.
//    //        For example, a patch that iterated from global grid point 3,4,5 to 7,8,9 would have a SubGridOffset
//    //        of:  {3,4,5}.
//
//    IntVector        EwaldMeshLimits = SPME_GlobalGrid->GetMeshLimits();
//    IntVector            SplineOrder = SPME_GlobalGrid->GetSplineOrder();
//
//    Vector               PatchOffset = CurrentPatch->SpatialOffset();
//    Vector               PatchExtent = CurrentPatch->SpatialExtent();  // HighCorner-LowCorner, where HighCorner is the max(X,Y,Z) of the subspace, LowCorner is
//                                                                       //   min(X,Y,Z)
//
//    ParticleIterator    ParticleList = CurrentPatch->ParticleVector();
//
//    Matrix3              CellInverse = SystemData->CellInverse();
//
//    Vector         InverseMeshLimits = 1.0/EwaldMeshLimits;
//
//
//    IntVector   SubGridIndexOffset, SubGridIndexExtent;
//    {
//      // Contain these temp variables
//      Vector  TempOffset=PatchOffset*CellInverse;
//      Vector  TempExtent=PatchExtent*CellInverse;
//
//      for(size_t Ind=0; Ind < 3; ++Ind) {
//        SubGridIndexOffset = floor(TempOffset[Ind]*static_cast<double> (EwaldMeshLimits[Ind]));  // Generate index offset to map local grid to global
//        SubGridIndexExtent = floor(TempExtent[Ind]*static_cast<double> (EwaldMeshLimits[Ind]));  // Generate index count to determine # grid points inside patch
//                                                                                                 //   not including ghost grid points
//      }
//    }
//    SPME_Grid_Map LocalGrid[SubGridIndexExtent[0]+SplineOrder][SubGridIndexExtent[1]+SplineOrder][SubGridIndexExtent[2]+SplineOrder];
//    // 3D array of Vector<SPME_Map> type, messy data structure.  Suggestions?
//    // Goal here is to create a sub-grid which maps to the patch local + extended "ghost" grid points, and to save the particle to grid charge mapping coefficients
//    // so that we don't have to re-generate them again.  This step is essentially necessary every time step, or every time the global "Solve SPME Charge" routine
//    // is done.
//
//    int Global_Shift=0;
//    if (SystemData->Orthorhombic()) {  //  Take a few calculation shortcuts to save some matrix multiplication.  Worth it?  Not sure..
//      Vector InverseGridMapping=EwaldMeshLimits*CellInverse; // Orthorhomibc, so CellInverse is diagonal and we can pre-multiply
//      for (ParticlePointer = ParticleList.begin(); ParticlePointer != ParticleList.end(); ++ParticlePointer) {
//        Vector U_Current = ParticlePointer->GetCoordinateVector();                              // Extract coordinate vector from particle
//        U_Current *= InverseGridMapping;                                                        // Convert to reduced coordinates
//        IntVector CellIndex;
//        for (size_t Ind=0; Ind < 3; ++Ind) { CellIndex[Ind] = floor(U_Current[Ind]) - SubGridIndexOffset[Ind] + Global_Shift; }  // Vector floor + shift by element
//
//        vector<Vector> Coeff_Array(SplineOrder), Deriv_Array(SplineOrder);
//        CalculateSpline(U_Current, Coeff_Array, Deriv_Array, SplineOrder);
//        for (int IndX=0; IndX < SplineOrder; ++IndX) {
//          for (int IndY=0; IndY < SplineOrder; ++IndY) {
//            for (int IndZ=0; IndZ < SplineOrder; ++IndZ) {
//              double Coefficient=Coeff_Array[IndX].x*Coeff_Array[IndY].y*Coeff_Array[IndZ].z;            // Calculate appropriate coefficient
//              Vector Gradient(Deriv_Array[IndX].x,Deriv_Array[IndY].y,Deriv_Array[IndZ].z);              // Extract appropriate derivative vector
//              (LocalGrid[CellIndex.x+IndX][CellIndex.y+IndY][CellIndex.z+IndZ]).AddMapPoint(ParticlePointer->GetGlobalHandle(),Coefficient,Gradient);
//            }
//          }
//        }
//      }
//    }
//    else {
//      for (ParticlePointer = ParticleList.begin(); ParticlePointer != ParticleList.end(); ++ParticlePointer) {
//        Vector U_Current = ParticlePointer->GetCOordinateVector(); // Extract coordinate vector from particle
//        U_Current *= CellInverse;                                           // Full matrix multiplication to get (X,Y,Z) for non-orthorhombic
//        for (size_t Ind=0; Ind < 3; ++Ind) { U_Current[Ind] *= EwaldMeshLimits[Ind]; }
//        for (size_t Ind=0; Ind < 3; ++Ind) { CellIndex[Ind] = floor(U_Current[Ind]) + Global_Shift; }    // Vector floor + shift by element
//
//        vector<Vector> Coeff_Array(SplineOrder), Deriv_Array(SplineOrder);
//        CalculateSpline(U_Current, Coeff_Array, Deriv_Array, SplineOrder);
//        for (int IndX=0; IndX < SplineOrder; ++IndX) {
//          for (int IndY=0; IndY < SplineOrder; ++IndY) {
//            for (int IndZ=0; IndZ < SplineOrder; ++IndZ) {
//              double Coefficient=Coeff_Array[IndX].x*Coeff_Array[IndY].y*Coeff_Array[IndZ].z;            // Calculate appropriate coefficient
//              Vector Gradient(Deriv_Array[IndX].x,Deriv_Array[IndY].y,Deriv_Array[IndZ].z);     // Extract appropriate derivative vector
//              (LocalGrid[CellIndex.x+IndX][CellIndex.y+IndY][CellIndex.z+IndZ]).AddMapPoint(ParticlePointer->GetGlobalHandle(),Coefficient,Gradient);
//            }
//          }
//        }
//      }
//    }
//    return;
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
//  Matrix3             InverseUnitCell;
//    double              Ewald_Beta;
//    IntVector           GridExtents, SubGridOffset, K;
//
//    IntVector           HalfGrid=GridExtents/2;
//
//    InverseUnitCell = MD_System->CellInverse();
//          EwaldBeta = MD_System->EwaldDampingCoefficient();
//
//        GridExtents = LocalGrid->GetExtents();          // Extents of the local subgrid
//      SubGridOffset = LocalGrid->GetOffsetVector();     // Offset needed to map local subgrid into global grid
//                  K = LocalGrid->GetGlobalExtent();     // Global grid size (K1,K2,K3)
//
//
//    vector<double> M1, M2, M3;
//    // Generate vectors of m_i/K_i
//
//    M1 = GenerateMVector(GridExtents.x(),SubGridOffset.x(),K.x());
//    M2 = GenerateMVector(GridExtents.y(),SubGridOffset.y(),K.y());
//    M3 = GenerateMVector(GridExtents.z(),SubGridOffset.z(),K.z());
//
//    vector<double> OrdinalSpline(SplineOrder-1);
//    OrdinalSpline=GenerateOrdinalSpline(SplineOrder);
//
//    vector<complex<double>> b1, b2, b3;
//    // Generate vectors of b_i (=exp(i*2*Pi*(n-1)m_i/K_i)*sum_(k=0..p-2)M_n(k+1)exp(2*Pi*k*m_i/K_i)
//
//    b1 = GeneratebVector(GridExtents.x(),M1,K.x(),SplineOrder,OrdinalSpline);
//    b2 = GeneratebVector(GridExtents.y(),M1,K.y(),SplineOrder,OrdinalSpline);
//    b3 = GeneratebVector(GridExtents.z(),M1,K.z(),SplineOrder,OrdinalSpline);
//
//
//    // Use previously calculated vectors to calculate our grids
//    double PI, PI2, InvBeta2, VolFactor
//
//           PI = acos(-1.0);
//          PI2 = PI*PI;
//     InvBeta2 = 1.0/(Ewald_Beta*Ewald_Beta);
//    VolFactor = 1.0/(PI*MD_System->GetVolume());
//
//    for (size_t kX=0; kX < GridExtents.x(); ++kX) {
//      for (size_t kY=0; kY < GridExtents.y(); ++kY) {
//        for (size_t kZ=0; kZ < GridExtents.z(); ++kZ) {
//
//          fB[kX][kY][kZ] = norm(b1[kX])*norm(b2[kY])*norm(b3[kZ]);  // Calculate B
//
//          if ( kX != kY != kZ != 0) {  // Calculate C and stress premultiplication factor
//            Vector m(M1[kX],M2[kY],M3[kZ]), M;
//            double M2, Factor;
//
//                 M = m*InverseUnitCell;
//                M2 = M.length2();
//            Factor = PI2*M2*InvBeta2;
//
//                       fC[kX][kY][kZ] = VolFactor*exp(-Factor)/M2;
//            StressPreMult[kX][kY][kZ] = 2*(1+Factor)/M2;
//          }
//        }
//      }
//    }
//    fC[0][0][0]=0.0;
//    StressPremult[0][0][0]=0.0;  // Exceptional values
}

std::vector<Point> Electroststics::calcReducedCoords(const std::vector<Point>& localRealCoordinates,
                                                     const Transformation3D<std::complex<double> >& invertSpace)
{

  vector < Point > localReducedCoords;

  if (!Orthorhombic)  // bool Orthorhombic; true if simulation cell is orthorhombic, false if it's generic
    for (size_t Index = 0; Index < NumParticlesInCell; ++Index) {
      CoordType s;        // Fractional coordinates; 3 - vector
      s = ParticleList[Index].GetCoordinates();        // Get non-ghost particle coordinates for this cell
      s *= InverseBox;       // For generic coordinate systems; InverseBox is a 3x3 matrix so this is a matrix multiplication = slow
      Local_ReducedCoords.push_back(s);        // Reduced non-ghost particle coordinates for this cell
    }
  else {
    for (size_t Index = 0; Index < NumParticlesInCell; ++Index) {
      CoordType s;        // Fractional coordinates; 3-vector
      s = ParticleList[Index].GetCoordinates();        // Get non-ghost particle coordinates for this cell
      s(0) *= Invert_Space(0, 0);
      s(1) *= Invert_Space(1, 1);
      s(2) *= Invert_Space(2, 2);        // 6 Less multiplications and additions than generic above
      Local_ReducedCoords.push_back(s);        // Reduced non-ghost particle coordinates for this cell
    }
  }

  return Local_ReducedCoords;
}
SimpleGrid<double> Electrostatics::fC(const IntVector& gridExtents,
                                      const MDSystem& system)
{
  Matrix3 InverseCell;
  SimpleGrid<double> C;  //C(gridExtents);
  double ewaldBeta, inverseCell;

  inverseCell = system.cellInverse();
  ewaldBeta = system.getEwaldBeta();

  IntVector HalfGrid = gridExtents / IntVector(2, 2, 2);

  double PI = acos(-1.0);
  double PI2 = PI * PI;
  double InvBeta2 = 1.0 / (Ewald_Beta * Ewald_Beta);

  double VolFactor = 1.0 / (PI * MDSystem->getVolume());

  Vector M;
  for (size_t m1 = 0; m1 < GridExtents.x(); ++m1) {
    M[0] = m1;
    if (m1 > HalfGrid.x())
      M[0] -= GridExtents.x();
    for (size_t m2 = 0; m2 < GridExtents.y(); ++m2) {
      M[1] = m2;
      if (m2 > HalfGrid.y())
        M[1] -= GridExtents.y();
      for (size_t m3 = 0; m3 < GridExtents.z(); ++m3) {
        M[2] = m3;
        if (m3 > HalfGrid.z())
          M[2] -= GridExtents.z();
        // Calculate C point values
        if (!(m1 == 0) && !(m2 == 0) && !(m3 == 0)) {  // Discount C(0,0,0)
          double TempM = M * InverseCell;
          double M2 = TempM.length2(TempM);
          double Val = VolFactor * exp(-PI2 * M2 * InvBeta2) / M2;
          C[m1][m2][m3] = Val;
        }
      }
    }
  }
  C[m1][m2][m3] = 0.0;
  return C;
}

SimpleGrid<std::complex<double> > Electrostatics::fB(const IntVector& gridExtents,
                                                     const MDSystem& system)
{
  Matrix3 InverseCell;
//  SimpleGrid<std::complex<double> > B(GridExtents.x(), GridExtents.y(), GridExtents, z());

  InverseCell = system->CellInverse();
  IntVector HalfGrid = GridExtents / 2;
  Vector InverseGrid = 1.0 / GridExtents;

  double PI = acos(-1.0);
  double OrderM12PI = 2.0 * PI * (SplineOrder - 1);

  vector<complex<double> > b1(GridExtents.x()), b2(GridExtents.y()), b3(GridExtents.z());
  vector<complex<double> > OrdinalSpline(SplineOrder - 1);

  OrdinalSpline = CalculateOrdinalSpline(SplineOrder - 1, SplineOrder);  // Calculates Mn(0)..Mn(n-1)

  // Calculate k_i = m_i/K_i
  for (size_t m1 = 0; m1 < GridExtents.x(); ++m1) {
    double kX = m1;
    if (m1 > HalfGrid.x())
      kX = m1 - GridExtents.x();
    kX /= GridExtents.x();
    complex<double> num = complex(cos(OrderM12PI * kX), sin(OrderM12PI * kX));
    complex<double> denom = OrdinalSpline[0];  //
    for (size_t k = 1; k < SplineOrder - 1; ++k) {
      denom += OrdinalSpline[k] * complex(cos(OrderM12PI * kX * k), sin(OrderM12PI * kX * k));
    }
    b1[m1] = num / denom;
  }

  for (size_t m2 = 0; m2 < GridExtents.y(); ++m2) {
    double kY = m2;
    if (m2 > HalfGrid.y())
      kY = m2 - GridExtents.y();
    kY /= GridExtents.y();
    std::complex<double> num = complex(cos(OrderM12PI * kY), sin(OrderM12PI * kY));
    std::complex<double> denom = OrdinalSpline[0];  //
    for (size_t k = 1; k < SplineOrder - 1; ++k) {
      denom += OrdinalSpline[k] * complex(cos(OrderM12PI * kY * k), sin(OrderM12PI * kY * k));
    }
    b2[m2] = num / denom;
  }

  for (size_t m3 = 0; m3 < GridExtents.z(); ++m3) {
    double kZ = m3;
    if (m3 > HalfGrid.z())
      kZ = m3 - GridExtents.y();
    kZ /= GridExtents.y();
    std::complex<double> num = complex(cos(OrderM12PI * kZ), sin(OrderM12PI * kZ));
    std::complex<double> denom = OrdinalSpline[0];  //
    for (size_t k = 1; k < SplineOrder - 1; ++k) {
      denom += OrdinalSpline[k] * std::complex<double>(cos(OrderM12PI * kZ * k), sin(OrderM12PI * kZ * k));
    }
    b3[m3] = num / denom;
  }

  for (size_t m1 = 0; m1 < GridExtents.x(); ++m1) {
    for (size_t m2 = 0; m2 < GridExtents.x(); ++m2) {
      for (size_t m3 = 0; m3 < GridExtents.x(); ++m3) {
        // Calculate B point values
        B[m1][m2][m3] = norm(b1[m1]) * norm(b2[m2]) * norm(b3[m3]);
      }
    }
  }
}

std::vector<std::complex<double> > GeneratebVector(const int& points,
                                                   const std::vector<double>& M,
                                                   const int& max,
                                                   const int& splineOrder,
                                                   const std::vector<double>& splineCoeff)
{
  double PI = acos(-1.0);
  double OrderM12PI = (splineOrder - 1) * 2.0 * PI;

  std::vector<std::complex<double> > b(points);
  for (int idx = 0; idx < points; ++idx) {
    double k = M[idx] / max;
    std::complex<double> Numerator = std::complex<double>(cos(OrderM12PI * k), sin(OrderM12PI * k));
    std::complex<double> Denominator;
    for (int p = 0; p < splineOrder - 1; ++p) {
      Denominator += splineCoeff[p] * std::complex<double>(cos(OrderM12PI * k1 * p), sin(OrderM12PI * k1 * p));
    }
    b[idx] = Numerator / Denominator;
  }
  return b;
}
