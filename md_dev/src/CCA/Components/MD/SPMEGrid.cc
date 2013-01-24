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

#include <CCA/Components/MD/SPMEGrid.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Math/UintahMiscMath.h>
#include <Core/Math/MiscMath.h>

#include <iostream>

#include <sci_values.h>

using namespace Uintah;
using namespace SCIRun;

template<class T>
SPMEGrid<T>& SPMEGrid<T>::mapChargeToGrid(const SimpleGrid<std::vector<MapPoint<T> > > gridMap,
                                          const ParticleSubset& globalParticleList)
{
//  IntVector Extent = LocalGridMap.GetLocalExtents();  // Total number of grid points on the local grid including ghost points (X,Y,Z)
//  IntVector Initial = Extent + LocalGridMap.GetLocalShift();  // Offset of lowest index points in LOCAL coordinate system.
//                                                              //  e.g. for SplineOrder = N, indexing from -N/2 to X + N/2 would have a shift of N/2 and an extent
//                                                              //       of X+N, indexing from 0 to N would have a shift of 0 and an extent of X+N
//
//  ParticleIterator ParticleList = CurrentPatch->ParticleVector();
//
//  for (size_t X = Initial[0]; X < Extent[0]; ++X) {
//    for (size_t Y = Initial[1]; Y < Extent[1]; ++Y) {
//      for (size_t Z = Initial[2]; Z < Extent[2]; ++Z) {
//        (LocalGridCopy[X][Y][Z]).IncrementGridCharge((LocalGridMap[X][Y][Z]).MapChargeFromAtoms(ParticleList));
//      }
//    }
//  }
//  return;
}

template<class T>
SPMEGrid<T>& SPMEGrid<T>::mapForceFromGrid(const SimpleGrid<std::vector<MapPoint<T> > > gridMap,
                                           ParticleSubset& globalParticleList) const
{

}

template<class T>
double SPMEGrid<T>::calculateEnergyAndStress(const vector<double>& M1,
                                             const vector<double>& M2,
                                             const vector<double>& M3,
                                             Matrix3 stressTensor,
                                             const cdGrid& stressPrefactor,
                                             const cdGrid& thetaRecip)
{

}

template<class T>
SPMEGrid<T>& SPMEGrid<T>::calculateField()
{
  // if (FieldValid == false) { calculate field grid; set FieldValid == true; return *this; }
  return *this;
}

template<class T>
SPMEGrid<T>& SPMEGrid<T>::inPlaceFFT_RealToFourier(/*Data for FFTW3 routine goes here */)
{

}

template<class T>
SPMEGrid<T>& SPMEGrid<T>::inPlaceFFT_FourierToReal(/*Data for FFTW3 routine goes here */)
{

}

