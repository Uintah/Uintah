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

#ifndef UINTAH_MD_MAPPOINT_H
#define UINTAH_MD_MAPPOINT_H

#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>

#include <vector>

namespace Uintah {

typedef int particleIndex;
typedef int particleId;

using SCIRun::Vector;
using SCIRun::IntVector;

class Point;
class Vector;
class IntVector;

/**
 *  @class MapPoint
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper
 *  @date   January, 2012
 *
 *  @brief
 *
 *  @param
 */
class MapPoint {

  public:

    /**
     * @brief
     * @param
     * @return
     */
    MapPoint();

    /**
     * @brief
     * @param
     * @return
     */
    ~MapPoint();

    /**
     * @brief
     * @param
     * @return
     */
    inline const double getParticleCharge() const
    {
      // TODO Figure out what charge is
//      return Particle->Charge();
//      return this->globalParticleSubset;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline const particleIndex getParticleID() const
    {
      return this->particleID;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline const IntVector getGridIndex() const
    {
      return this->gridIndex;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline const double getCoefficient() const
    {
      return this->coefficient;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline const Vector getForceVector() const
    {
      return this->forceVector;
    }

  private:

    // Some manner of mapping the provided particle pointer/reference to a storable way to index the particle
    particleIndex particleID;
    IntVector gridIndex;
    double coefficient;
    Vector forceVector;

};

}  // End namespace Uintah

#endif
