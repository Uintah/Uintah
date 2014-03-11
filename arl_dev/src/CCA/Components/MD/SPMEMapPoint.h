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

#ifndef UINTAH_MD_SPMEMAPPOINT_H
#define UINTAH_MD_SPMEMAPPOINT_H

#include <CCA/Components/MD/SimpleGrid.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <vector>

namespace Uintah {

  typedef int particleIndex;
  typedef int particleId;

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
                   SimpleGrid<SCIRun::Vector> forceGrid,
                   SimpleGrid<Matrix3> dipoleForceGrid);

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

      inline const SimpleGrid<Vector>& getGradientGrid() const
		{
    	  return d_forceGrid;
		}

      inline const SimpleGrid<Vector>& getPotentialDerivativeGrid() const
		{
    	  // d_forceGrid is the derivative of the potential at a point
    	  return d_forceGrid;
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

      inline const SimpleGrid<Matrix3>& getDipoleGrid() const
	  {
    	  return d_polarizableForceGrid;
	  }

    private:

      particleIndex d_particleID;
      IntVector d_gridOffset;
      SimpleGrid<double> d_chargeGrid;
      SimpleGrid<SCIRun::Vector> d_forceGrid;
      SimpleGrid<Matrix3> d_polarizableForceGrid;

  };

}  // End namespace Uintah

#endif
