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

#include <CCA/Components/MD/Electrostatics/Ewald/InverseSpace/SPME/SPMEPatch.h>
#include <Core/Geometry/IntVector.h>

#include <iostream>

using namespace Uintah;


SPMEPatch::SPMEPatch()
{

}

SPMEPatch::~SPMEPatch()
{
  delete d_Q_patchLocal;

  delete d_stressPrefactor;

  if (d_theta) {
    delete d_theta;
  }

//  for (int AtomTypeIndex=0; AtomTypeIndex < d_chargeMapVector.size(); ++AtomTypeIndex) {
//    delete d_chargeMapVector[AtomTypeIndex];
//  }
}

SPMEPatch::SPMEPatch(       IntVector   kGridExtents,
                            IntVector   kGridOffset,
                            IntVector   plusGhostExtents,
                            IntVector   minusGhostExtents,
                     const  Patch*      patch,
                            double      patchVolumeFraction,
                            int         splineSupport,
                            MDSystem*   system)
                    :d_localExtents(kGridExtents),
                     d_globalOffset(kGridOffset),
                     d_posGhostExtents(plusGhostExtents),
                     d_negGhostExtents(minusGhostExtents),
                     d_patch(patch)
{
  d_Q_patchLocal        = scinew SimpleGrid<dblcomplex>(kGridExtents,
                                                        kGridOffset,
                                                        MDConstants::IV_ZERO,
                                                        splineSupport);
  d_stressPrefactor     = scinew SimpleGrid<Matrix3>(kGridExtents,
                                                     kGridOffset,
                                                     MDConstants::IV_ZERO,
                                                     0);
  d_theta               = scinew SimpleGrid<double>(kGridExtents,
                                                    kGridOffset,
                                                    MDConstants::IV_ZERO,
                                                    0);

  // Pre-allocate memory for charge maps.
  size_t numAtomTypes = system->getNumAtomTypes();
  const int estimatedMaximumMultiplier = 2;
  const IntVector IV_FLAG(-1,-1,-1);
  const IntVector IV_SPLINE(splineSupport,splineSupport,splineSupport);
  SimpleGrid<double> sg_doubleNull(IV_SPLINE, IV_FLAG, IV_FLAG, 0);
   sg_doubleNull.fill(0.0);
   SimpleGrid<SCIRun::Vector> sg_VectorNull(IV_SPLINE, IV_FLAG, IV_FLAG, 0);
   sg_VectorNull.fill(Vector(0.0, 0.0, 0.0));
   SimpleGrid<Matrix3> sg_Matrix3Null(IV_SPLINE, IV_FLAG, IV_FLAG, 0);
   sg_Matrix3Null.fill(Matrix3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
   SPMEMapPoint nullMap(-1, IV_FLAG, sg_doubleNull, sg_VectorNull, sg_Matrix3Null);

//   Uintah::Matrix3 m3ZERO = Uintah::Matrix3(0.0,0.0,0.0,  0.0,0.0,0.0,  0.0,0.0,0.0);
//   {
//   // Test 1
////   SimpleGrid<double> doubleCopy(sg_doubleNull);
////   // Test 2
////   SimpleGrid<SCIRun::Vector> VectorCopy(sg_VectorNull);
////   // Test 3
////   SimpleGrid<Matrix3> Matrix3Copy(sg_Matrix3Null);
////   // Test 4 for crashing of SimpleGrid vs. crashing of Matrix3
////   std::vector<SimpleGrid<double> > vSGDouble(10,sg_doubleNull);
////   // Test 5
////   std::vector<SimpleGrid<SCIRun::Vector> > vSGVector(10,sg_VectorNull);
//     // Test 6.1
//     std::vector<Uintah::Matrix3> vMatrix3(10, m3ZERO );
//     // Test 6.2
//     LinearArray3<Uintah::Matrix3> laMatrix3(2,2,2, m3ZERO );
//   // Test 6
////   std::vector<SimpleGrid<Uintah::Matrix3> > vSGMatrix(10,sg_Matrix3Null);
//   }


  d_chargeMapVector = std::vector< std::vector<SPMEMapPoint> >(numAtomTypes);
  for (size_t AtomType = 0; AtomType < numAtomTypes; ++AtomType) {
    // Initial buffer is 2*relative fraction patch comprises of entire system*total number of atoms of type
    size_t totalNumberOfType = system->getNumAtomsOfType(AtomType);
    size_t numberBuffered = ceil(totalNumberOfType * estimatedMaximumMultiplier * patchVolumeFraction);
    d_chargeMapVector[AtomType] = std::vector<SPMEMapPoint> (numberBuffered, nullMap);
  }

  /*
   * --------------------------------------------------------------------------
   * Reserve charge-map memory for each patch/material set
   * --------------------------------------------------------------------------
   * In general, if we have a patch that occupies a cube of volume v, the entire system
   * occupies a space of volume V, and we have N_i sites of type i.
   * The site density of the atoms of type i are N_i/V for the entire system.
   *
   * If we assume that the local density should mimic the global density, then n_i/v == N_i/V,
   * where n_i is the number of sites in the local subspace.  The number of local sites (n_i)
   * is therefore just n_i=(v/V)*N_i.  For normal, liquid-like systems we would expect the
   * density amplification due to packing effects to be in the range of 2-3.
   *
   * So, put all together, if we have a patch with extents x,y,z and the total system has extents X,Y,Z:
   *
   * (x/X * y/Y * z/Z)*N_i*(a factor between 2 and 3, but since we may have to adjust and want to double
   * when we do, let's choose 2) is the number you're looking for.  You would presumably allocate this
   * at initialization for every patch object you have and only reallocate by doubling if you exceed that
   * number. (We may have a large number of re-allocations on occasion, but it should taper off quickly
   * after the first re-allocation.)
   *
   * Note that (x/X * y/Y * z/Z) is shorthand for v/V.  For systems that are not orthorhombic, the first term will
   * not make much sense (v/V is always valid, though).
   *
   * Currently, Uintah currently is not designed to deal with skew axes and non-orthorhombic
   * systems.
   */

//  // FIXME TODO All of this crap can be removed if we simply make a particle variable for the SPME Map Point
//  const int estimatedMaximumMultiplier = 2;
//  size_t numAtomTypes = system->getNumAtomTypes();
//  // FIXME 04/11/14
//  d_chargeMapVector = std::vector<std::vector<SPMEMapPoint> >(numAtomTypes);
//
//  for (size_t AtomType = 0; AtomType < numAtomTypes; ++AtomType) {
//    size_t totalNumberOfType = system->getNumAtomsOfType(AtomType);
//    size_t numberBuffered = totalNumberOfType * estimatedMaximumMultiplier * patchVolumeFraction;
//    for (size_t mapIndex = 0; mapIndex < numberBuffered; ++mapIndex) {
//      // Instantiate a null map point to reserve the appropriate memory
//      SPMEMapPoint tempMapPoint(-1, IV_FLAG, sg_doubleNull, sg_VectorNull, sg_Matrix3Null);
//      // And build vector directly
//      d_chargeMapVector[AtomType].push_back(tempMapPoint);
//    }
//  }

////  const int estimatedMaximumMultiplier = 2;
//  size_t numAtomTypes = system->getNumAtomTypes();
//  d_chargeMapVector.resize(numAtomTypes);
//  for (size_t AtomType = 0; AtomType < numAtomTypes; ++AtomType) {
//    std::vector<SPMEMapPoint> gridmapTemp;
////    gridmapTemp.reserve(10);
//    d_chargeMapVector.push_back(gridmapTemp);
//
//    //int totalNumberOfAtomType = system->getNumAtomsOfType(AtomType);
//    //int estimatedLocalNumberOfAtomType = patchVolumeFraction * estimatedMaximumMultiplier * totalNumberOfAtomType;
//    //std::vector<SPMEMapPoint>* gridmap = new std::vector<SPMEMapPoint>();
//    //gridmap.reserve(estimatedLocalNumberOfAtomType);
//    //(*gridmap).reserve(10);
//    //d_chargeMapVector[AtomType] = *gridmap;
//    //d_chargeMapVector[AtomType].reserve(10);
//    //std::cout << d_chargeMapVector[AtomType].capacity() << " XXXXX " << endl;
//  }
}

void SPMEPatch::verifyChargeMapAllocation(const int dataSize,
                                          const int globalAtomTypeIndex) {
  // Checks to see if we can accommodate dataSize items.  If so, we return.
  // If not, we keep doubling memory until we have more than enough,
  // copy the old vector to the new one, and then return.
  int currentVectorSize =
      static_cast<int> (d_chargeMapVector[globalAtomTypeIndex].capacity());
  if (dataSize <= currentVectorSize) {
    return;
  }
  int newVectorSize = currentVectorSize * 2;
  while (dataSize > newVectorSize) {
    newVectorSize *= 2;
  }

  // Pre-allocate memory for entire new vector
  IntVector currentMapPointExtents;
  // Get grid extents from a current grid
  currentMapPointExtents =
      ((d_chargeMapVector[globalAtomTypeIndex][0]).getChargeGrid())->getExtents();

  // Set up a dummy grid point for building the new vector
  const IntVector IV_FLAG(-1,-1,-1);
  SimpleGrid<double> sg_doubleNull(currentMapPointExtents, IV_FLAG, IV_FLAG, 0);
  sg_doubleNull.fill(0.0);

  SimpleGrid<Vector> sg_VectorNull(currentMapPointExtents, IV_FLAG, IV_FLAG, 0);
  sg_VectorNull.fill(Vector(0.0, 0.0, 0.0));

  SimpleGrid<Matrix3> sg_Matrix3Null(currentMapPointExtents, IV_FLAG, IV_FLAG, 0);
  sg_Matrix3Null.fill(Matrix3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  SPMEMapPoint nullMap(-1,
                       IV_FLAG,
                       sg_doubleNull,
                       sg_VectorNull,
                       sg_Matrix3Null);

  // Resize the current vector with a copy of the nulled mapPoint;
  // Since this happens as we're building a new set of mapPoints, and since
  // there is no correspondence between the last set of mapPoints and the
  // current, there is no need to copy old data.
  d_chargeMapVector[globalAtomTypeIndex].resize(newVectorSize,nullMap);
  return;
}
