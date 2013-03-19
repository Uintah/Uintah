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

#ifndef UINTAH_MD_SPMEMAPPOINT_H
#define UINTAH_MD_SPMEMAPPOINT_H

#include <CCA/Components/MD/SimpleGrid.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <vector>

namespace Uintah {

typedef int particleIndex;
typedef int particleId;

class Point;
class Vector;
class ParticleSubset;

/**
 *  @class SPMEMapPoint
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper
 *  @date   February, 2013
 *
 *  @brief
 *
 *  @param
 */
class SPMEMapPoint {

  public:

    /**
     * @brief
     * @param
     * @return
     */
    SPMEMapPoint();

    /**
     * @brief
     * @param
     * @return
     */
    ~SPMEMapPoint();

    /**
     * @brief
     * @param
     * @return
     */
    SPMEMapPoint(particleIndex particleID,
                 IntVector gridOffset,
                 SimpleGrid<double> chargeGrid,
                 SimpleGrid<SCIRun::Vector> forceGrid);

    /**
     * @brief
     * @param
     * @return
     */
    inline const particleIndex getParticleID() const
    {
      return d_particleID;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline const IntVector getGridOffset() const
    {
      return d_gridOffset;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline const SimpleGrid<double>& getChargeGrid() const
    {
      return d_chargeGrid;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline const SimpleGrid<SCIRun::Vector>& getForceGrid() const
    {
      return d_forceGrid;
    }

  private:

    particleIndex d_particleID;
    IntVector d_gridOffset;
    SimpleGrid<double> d_chargeGrid;
    SimpleGrid<SCIRun::Vector> d_forceGrid;

};

}  // End namespace Uintah

#endif
