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

#include <CCA/Components/MD/SPMEGridPoint.h>
#include <CCA/Components/MD/SPMEGridMap.h>
#include <Core/Math/UintahMiscMath.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MiscMath.h>

#include <iostream>

#include <sci_values.h>

using namespace Uintah;
using namespace SCIRun;

template<class T>
SPMEGridPoint<T>::SPMEGridPoint()
{

}

template<class T>
SPMEGridPoint<T>::~SPMEGridPoint()
{

}

template<class T>
void SPMEGridPoint<T>::mapChargeFromAtoms()
{
  unsigned int NumMappedAtoms = d_mappedAtoms.size();
  double gridPointCharge = 0.0;
  double chargeCoefficeintWeight = 0.0;
  for (unsigned int idx = 0; idx < NumMappedAtoms; ++idx) {
    double weight = d_mappedAtoms[idx].ChargeWeight();
    gridPointCharge += weight * d_mappedAtoms[idx].GetParticleCharge();
    chargeCoefficeintWeight += weight;
  }
}

template<class T>
void SPMEGridPoint<T>::mapChargeToAtoms()
{
  unsigned int NumMappedAtoms = d_mappedAtoms.size();
  double InverseChargeWeight = 1.0 / d_totalChargeCoefficientWeight;
  for (unsigned int idx = 0; idx < NumMappedAtoms; ++idx) {
    double AtomCharge = d_mappedAtoms[idx].GetParticleCharge();
    double AtomWeight = d_mappedAtoms[idx].ChargeWeight();
    AtomCharge += (AtomWeight * InverseChargeWeight) * d_gridPointCharge;  // Add to the current atom charge the contribution at the grid point from this atom
    d_mappedAtoms[idx].setParticleCharge(AtomCharge);
  }
}

template<class T>
void SPMEGridPoint<T>::addMapPoint(ParticleVariable<double>& pv,
                                   const double& weight,
                                   const Vector& gradient)
{
  SPMEMapPoint<double> newMapPoint(pv, weight, gradient);
  d_mappedAtoms.push_back(newMapPoint);
}

