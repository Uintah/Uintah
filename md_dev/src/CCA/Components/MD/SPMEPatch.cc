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

#include <CCA/Components/MD/SPMEPatch.h>
#include <Core/Geometry/IntVector.h>

#include <iostream>
#define IV_ZERO IntVector(0,0,0)

using namespace Uintah;


SPMEPatch::SPMEPatch()
{

}

SPMEPatch::~SPMEPatch()
{
  delete d_Q_patchLocal;
//  delete d_stressPrefactor;
  delete d_theta;

//  for (int AtomTypeIndex=0; AtomTypeIndex < d_chargeMapVector.size(); ++AtomTypeIndex) {
//    delete d_chargeMapVector[AtomTypeIndex];
//  }
}

SPMEPatch::SPMEPatch(IntVector extents,
                     IntVector offset,
                     IntVector plusGhostExtents,
                     IntVector minusGhostExtents,
                     const Patch* patch,
                     double patchVolumeFraction,
                     int splineSupport,
                     MDSystem* system) :
      d_localExtents(extents),
      d_globalOffset(offset),
      d_posGhostExtents(plusGhostExtents),
      d_negGhostExtents(minusGhostExtents),
      d_patch(patch)
{
  d_Q_patchLocal    = scinew SimpleGrid<dblcomplex>(extents, offset, IV_ZERO, splineSupport);
  d_stressPrefactor = scinew SimpleGrid<Matrix3>(extents, offset, IV_ZERO, 0);
  d_theta           = scinew SimpleGrid<double>(extents, offset, IV_ZERO, 0);

  // Pre-allocate memory for charge maps.
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

//  const int estimatedMaximumMultiplier = 2;
  size_t numAtomTypes = system->getNumAtomTypes();
  d_chargeMapVector.resize(numAtomTypes);
  for (size_t AtomType = 0; AtomType < numAtomTypes; ++AtomType) {
    std::vector<SPMEMapPoint> gridmapTemp;
    gridmapTemp.reserve(10);
    d_chargeMapVector.push_back(gridmapTemp);
    //int totalNumberOfAtomType = system->getNumAtomsOfType(AtomType);
    //int estimatedLocalNumberOfAtomType = patchVolumeFraction * estimatedMaximumMultiplier * totalNumberOfAtomType;
    //std::vector<SPMEMapPoint>* gridmap = new std::vector<SPMEMapPoint>();
    //gridmap.reserve(estimatedLocalNumberOfAtomType);
    //(*gridmap).reserve(10);
    //d_chargeMapVector[AtomType] = *gridmap;
    //d_chargeMapVector[AtomType].reserve(10);
    //std::cout << d_chargeMapVector[AtomType].capacity() << " XXXXX " << endl;
  }
}

void SPMEPatch::verifyChargeMapAllocation(const int dataSize, const int globalAtomTypeIndex) {
  // Checks to see if we can accommodate dataSize items.  If so, we return.  If not, we reserve
  //   twice as much memory, copy the old vector to the new one, and then return.
  int currentVectorSize = static_cast<int> (d_chargeMapVector[globalAtomTypeIndex].capacity());
  if (dataSize <= currentVectorSize) { return; }
  int newVectorSize = currentVectorSize * 2;
  while (dataSize < newVectorSize) { newVectorSize *= 2; }
  vector<SPMEMapPoint> tempStorage = d_chargeMapVector[globalAtomTypeIndex];
  d_chargeMapVector[globalAtomTypeIndex].reserve(static_cast<size_t> (newVectorSize));
  // Not sure we need the insert?
  d_chargeMapVector[globalAtomTypeIndex].insert(d_chargeMapVector[globalAtomTypeIndex].begin(),tempStorage.begin(),tempStorage.end());
  return;
}
