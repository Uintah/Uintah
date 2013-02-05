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

SPME::SPME(const MDSystem& SimulationSystem,
           const double _EwaldBeta,
           const bool _IsPolarizable,
           const double _Tolerance,
           const SCIRun::IntVector& _K,
           const CenteredCardinalBSpline& _Spline)
          :EwaldBeta(_EwaldBeta),
           polarizable(_IsPolarizable),
           PolarizationTolerance(_Tolerance),
           KLimits(_K),
           InterpolatingSpline(_Spline)
{
  ElectrostaticMethod = SPME;
  // Initialize and check for proper construction
  this->initialize(SimulationSystem);
  this->setup();
  SCIRun::IntVector localGridSize = localGridExtents + localGhostPositiveSize + localGhostNegativeSize;
  SimpleGrid<complex<double> > Q(localGridSize,localGridOffset,localGhostNegativeSize,localGhostPositiveSize);
  Q.initialize(complex<double> (0.0,0.0));
}

std::vector<dblcomplex> SPME::generateBVector( const std::vector<double>& MFractional,
                                               const int InitialVectorIndex,
                                               const int LocalGridExtent,
                                               const CenteredCardinalBSpline& InterpolatingSpline) const
{
  double PI = acos(-1.0);
  double TwoPI = 2.0*PI;
  double orderM12PI = TwoPI*(InterpolatingSpline.Order()-1);

  int HalfSupport=InterpolatingSpline.HalfSupport();
  std::vector<dblcomplex> b(LocalGridExtent);
  std::vector<double> ZeroAlignedSpline=InterpolatingSpline.Evaluate(0);

  double* LocalMFractional=MFractional[InitialVectorIndex]; // Reset MFractional zero so we can index into it negatively
  for (size_t Index=0; Index < LocalGridExtent; ++Index) {
    double Internal=TwoPI*LocalMFractional[Index];
     // Formula looks significantly different from given SPME for offset splines.
     //   See Essmann et. al., J. Chem. Phys. 103 8577 (1995). for conversion, particularly formula C3 pt. 2 (paper uses pt. 4)
     dblcomplex Phi_N=0.0;
     for (int DenomIndex=-HalfSupport; DenomIndex <= HalfSupport; ++DenomIndex) {
        Phi_N += dblcomplex(cos(Internal*DenomIndex),sin(Internal*DenomIndex));
     }
     b[Index]=1.0/Phi_N;
  }
  return b;
}

SimpleGrid<double> SPME::CalculateBGrid(const SCIRun::IntVector& localExtents,
                                        const SCIRun::IntVector& globalOffset) const {

     size_t Limit_Kx = KLimits.x();
     size_t Limit_Ky = KLimits.y();
     size_t Limit_Kz = KLimits.z();

     std::vector<double> mf1 = SPME::generateMFractionalVector(Limit_Kx, InterpolatingSpline);
     std::vector<double> mf2 = SPME::generateMFractionalVector(Limit_Ky, InterpolatingSpline);
     std::vector<double> mf3 = SPME::generateMFractionalVector(Limit_Kz, InterpolatingSpline);

     // localExtents is without ghost grid points
     std::vector<dblcomplex> b1 = generateBVector(mf1, globalOffset.x(), localExtents.x(), InterpolatingSpline);
     std::vector<dblcomplex> b2 = generateBVector(mf2, globalOffset.y(), localExtents.y(), InterpolatingSpline);
     std::vector<dblcomplex> b3 = generateBVector(mf3, globalOffset.z(), localExtents.z(), InterpolatingSpline);

     SimpleGrid<double> BGrid(localExtents,globalOffset,0); // No ghost cells; internal only

     size_t XExtents=localExtents.x();
      size_t YExtents=localExtents.y();
      size_t ZExtents=localExtents.z();

      int XOffset=globalOffset.x();
      int YOffset=globalOffset.y();
      int ZOffset=globalOffset.z();

     for (size_t kX = 0; kX < XExtents; ++kX) {
        for (size_t kY = 0; kY < YExtents; ++kY) {
           for (size_t kZ = 0; kZ < ZExtents; ++kZ) {
              BGrid(kX, kY, kZ) = norm(b1[kX+XOffset])*norm(b2[kY+YOffset])*norm(b3[kZ+ZOffset]);
         }
      }
   }
   return BGrid;
}

SimpleGrid<double> SPME::CalculateCGrid(const SCIRun::IntVector& Extents,
                                        const SCIRun::IntVector& Offset) const {

   std::vector<double> mp1 = SPME::generateMPrimeVector(KLimits.x(), InterpolatingSpline);
   std::vector<double> mp2 = SPME::generateMPrimeVector(KLimits.y(), InterpolatingSpline);
   std::vector<double> mp3 = SPME::generateMPrimeVector(KLimits.z(), InterpolatingSpline);

   size_t XExtents=Extents.x();
   size_t YExtents=Extents.y();
   size_t ZExtents=Extents.z();

   int XOffset=Offset.x();
   int YOffset=Offset.y();
   int ZOffset=Offset.z();

   double PI=acos(-1.0);
   double PI2=PI*PI;
   double invBeta2 = 1.0/(EwaldBeta*EwaldBeta);
   double invVolFactor = 1.0/(SystemVolume*PI);

   SimpleGrid<double> CGrid(Extents,Offset,0); // No ghost cells; internal only
   for (size_t kX = 0; kX < XExtents; ++kX) {
      for (size_t kY = 0; kY < YExtents; ++kY) {
         for (size_t kZ=0; kZ < ZExtents; ++kZ) {
            if (kX != 0 || kY != 0 || kZ != 0) {
               SCIRun::Vector m(mp1[kX+XOffset],mp2[kY+YOffset],mp3[kZ+ZOffset]);

               m *= InverseUnitCell;

               double M2=m.length2();
               double factor=PI2*M2*invBeta2;
               CGrid(kX, kY, kZ)=invVolFactor*exp(-factor)/M2;
            }
         }
      }
   }
   CGrid(0, 0, 0)=0;
   return CGrid;
}

SimpleGrid<Matrix3> SPME::CalculateStressPrefactor(const SCIRun::IntVector& Extents, const SCIRun::IntVector& Offset) {

   std::vector<double> mp1 = SPME::generateMPrimeVector(KLimits.x(), InterpolatingSpline);
   std::vector<double> mp2 = SPME::generateMPrimeVector(KLimits.y(), InterpolatingSpline);
   std::vector<double> mp3 = SPME::generateMPrimeVector(KLimits.z(), InterpolatingSpline);

   size_t XExtents=Extents.x();
   size_t YExtents=Extents.y();
   size_t ZExtents=Extents.z();

   int XOffset=Offset.x();
   int YOffset=Offset.y();
   int ZOffset=Offset.z();

   double PI=acos(-1.0);
   double PI2=PI*PI;
   double invBeta2 = 1.0/(EwaldBeta*EwaldBeta);

   SimpleGrid<Matrix3>StressPre(Extents,Offset,0); // No ghost cells; internal only
   for (size_t kX = 0; kX < XExtents; ++kX) {
      for (size_t kY = 0; kY < YExtents; ++kY) {
         for (size_t kZ=0; kZ < ZExtents; ++kZ) {
            if (kX!=0 || kY!=0 || kZ!=0) {
               SCIRun::Vector m(mp1[kX+XOffset],mp2[kY+YOffset],mp3[kZ+ZOffset]);
               m *= InverseUnitCell;
               double M2=m.length2();
               Matrix3 LocalStressContribution(-2.0*(1.0+PI2*M2*invBeta2)/M2);

               // Multiply by fourier vectorial contribution
               for (size_t s1=0; s1 < 3; ++s1) {
                 for (size_t s2=0; s2 < 3; ++s2) {
                   LocalStressContribution(s1,s2) *= (m[s1]*m[s2]);
                 }
               }

               // Account for delta function
               for (size_t Delta=0; Delta < 3; ++Delta) {
                 LocalStressContribution(Delta,Delta) += 1.0;
               }

               StressPre(kX, kY, kZ)=LocalStressContribution;
            }
         }
      }
   }
   StressPre(0, 0, 0)=Matrix3(0);
   return StressPre;
}

// Interface implementations
void SPME::initialize(const MDSystem& SimulationSystem)
{
  // We call SPME::initialize from the constructor or if we've somehow maintained our object across a system change

  // Note:  I presume the indices are the local cell indices without ghost cells
  localGridExtents = patch->getCellHighIndex() - patch->getCellLowIndex();

  // Holds the index to map the local chunk of cells into the global cell structure
  localGridOffset  = patch->getCellLowIndex();

  // Get useful information from global system descriptor to work with locally.
  UnitCell         = SimulationSystem->UnitCell();
  InverseUnitCell  = SimulationSystem->InverseUnitCell();
  SystemVolume     = SimulationSystem->Volume();

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
  SimpleGrid<double> fBGrid = CalculateBGrid(localGridExtents,localGridOffset);
  SimpleGrid<double> fCGrid = CalculateCGrid(localGridExtents,localGridOffset);
  // Composite B and C into Theta
  size_t XExtent = localGridExtents.x();
  size_t YExtent = localGridExtents.y();
  size_t ZExtent = localGridExtents.z();
  for (size_t XIndex=0; XIndex < XExtent; ++XIndex) {
    for (size_t YIndex=0; YIndex < YExtent; ++YIndex) {
      for (size_t ZIndex=0; ZIndex < ZExtent; ++ZIndex) {
        fTheta(XIndex, YIndex, ZIndex) = fBGrid(XIndex, YIndex, ZIndex)*fCGrid(XIndex, YIndex, ZIndex);
      }
    }
  }

  StressPrefactor = CalculateStressPrefactor;
}

void SPME::calculate()
{
  // Note:  Must run SPME->setup() after every time there is a new box/K grid mapping (e.g. every step for NPT)
  //          This should be checked for in the system electrostatic driver
  vector<vector<ChargeMapPoints> > ChargeMap=SPME::generateChargeMap(pset,InterpolatingSpline);
  bool ElectrostaticsConverged=false;
  int NumberofIterations=0;
  while (!ElectrostaticsConverged && (NumberofIterations < MaxIterations)) {
    SPME::MapChargeToGrid(pset,ChargeMap); // Calculate Q(r)

    // Map the local patch's charge grid into the global grid and transform
    SPME::GlobalMPIReduceChargeGrid();
    SPME::ForwardTransformGlobalChargeGrid(); // Q(r) -> Q*(k)
    // Once reduced and transformed, we need the local grid repopulated with Q*(k)
    SPME::MPIDistributeLocalChargeGrid();

    // Multiply the transformed Q out
    size_t XExtent=localGridExtents.x();
    size_t YExtent=localGridExtents.y();
    size_t ZExtent=localGridExtents.z();
    double localEnergy=0.0;
    Matrix3 localStress(0.0);
    for (int kX=0; kX < XExtent; ++kX) {
      for (int kY=0; kY < YExtent; ++kY) {
        for (int kZ=0; kZ < ZExtent; ++kZ) {
          complex<double> GridValue=Q(kX, kY, kZ);
          Q(kX, kY, kZ) = GridValue*conj(GridValue)*fTheta(kX, kY, kZ);  // Calculate (Q*Q^)*(B*C)
          localEnergy += Q(kX, kY, kZ);
          localStress += Q(kX, kY, kZ)*StressPrefactor(kX, kY, kZ);
        }
      }
    }

    // Transform back to real space
    SPME::GlobalMPIReduceChargeGrid();
    SPME::ReverseTransformGlobalChargeGrid();
    SPME::MPIDistributeLocalChargeGrid();

    //  This may need to be before we transform the charge grid back to real space if we can calculate
    //    polarizability from the fourier space component
    ElectrostaticsConverged = true;
    if (polarizable) {
      // calculate polarization here
      // if (RMSPolarizationDifference > PolarizationTolerance) { ElectrostaticsConverged = false; }
      std::cerr << "Error:  Polarization not currently implemented!";
    }
    // Sanity check - Limit maximum number of polarization iterations we try
    ++NumberofIterations;
  }
  SPME::MapForcesFromGrid(pset,ChargeMap); // Calculate electrostatic contribution to f_ij(r)
}

void SPME::finalize()
{
  // Something goes here, though I'm not sure what
}

std::vector<std::vector<MapPoint> > SPME::GenerateChargeMap(ParticleSubset* pset,
                                                            CenteredCardinalBSpline& spline)
{
  /*  WORK IN PROGRESS
    I think we can make this a vector<SimpleGrid<MapPoint> > to make iteration and overlay far easier
    JBH - 2-4-2013

  int MaxParticleIndex=localParticleSet->size();
  std::vector<SimpleGrid<MapPoint> > ChargeMap;
  // Loop through particles
  for (size_t ChargeIndex=0; ChargeIndex < MaxChargeIndex; ++ChargeIndex) {
    int ParticleID = pst[ChargeIndex]->GetParticleID();

    vector<MapPoint> ParticleMap;


  }
  */
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



SimpleGrid<double>& SPME::mapChargeToGrid(const std::vector<std::vector<MapPoint> > gridMap,
                                          const ParticleSubset* globalParticleList)
{

}

SimpleGrid<double>& SPME::mapForceFromGrid(const std::vector<std::vector<MapPoint> > gridMap,
                                           ParticleSubset* globalParticleList)
{

}


