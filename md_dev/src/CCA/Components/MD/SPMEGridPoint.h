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

#ifndef UINTAH_SPME_GRID_POINT_H
#define UINTAH_SPME_GRID_POINT_H

#include <CCA/Components/MD/MapPoint.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <vector>
#include <list>
#include <complex>

namespace Uintah {

using SCIRun::Vector;
using SCIRun::IntVector;

template<class T> class SPMEGridPoint {

  public:
    SPMEGridPoint();

    ~SPMEGridPoint();

    void mapChargeToAtoms();

    void mapForceToAtoms();

    void mapChargeFromAtoms();

    void addMapPoint(ParticleVariable<double>& pv,
                     const double& weight,
                     const Vector& gradient);

  private:
    double d_gridPointCharge;
    double d_totalChargeContributionWeight;
    double d_totalChargeCoefficientWeight;
    Vector d_field;
    std::vector<MapPoint<double> > d_mappedAtoms;

};

}  // End namespace Uintah

#endif
