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

#ifndef UINTAH_MD_SPMEPatch_H
#define UINTAH_MD_SPMEPatch_H

#include <CCA/Components/MD/SimpleGrid.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Math/Matrix3.h>

#include <complex>

namespace Uintah {

/**
 *  @class SPMEPatch
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper
 *  @date   February, 2013
 *
 *  @brief
 *
 *  @param
 */
class SPMEPatch {

  public:

    /**
     * @brief Default constructor
     * @param
     */
    SPMEPatch();

    /**
     * @brief Default destructor
     * @param
     */
    ~SPMEPatch();

    /**
     * @brief
     * @param
     * @param
     * @param
     */
    SPMEPatch(IntVector extents,
              IntVector offset,
              IntVector plusGhostExtents,
              IntVector minusGhostExtents);

    /**
     * @brief
     * @param
     * @return
     */
    inline IntVector getLocalExtents() const
    {
      return this->localExtents;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline IntVector getGlobalOffset() const
    {
      return this->globalOffset;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline IntVector getPosGhostExtents() const
    {
      return this->posGhostExtents;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline IntVector getNegGhostExtents() const
    {
      return this->negGhostExtents;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline SimpleGrid<double> getTheta() const
    {
      return this->theta;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline void setTheta(SimpleGrid<double> theta)
    {
      this->theta = theta;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline SimpleGrid<Matrix3> getStressPrefactor() const
    {
      return this->stressPrefactor;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline void setStressPrefactor(SimpleGrid<Matrix3> stressPrefactor)
    {
      this->stressPrefactor = stressPrefactor;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline SimpleGrid<complex<double> > getQ() const
    {
      return this->Q;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline void setQ(SimpleGrid<complex<double> > q)
    {
      this->Q = q;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline ParticleSubset* getPset() const
    {
      return this->pset;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline void setPset(ParticleSubset* pset)
    {
      this->pset = pset;
    }

  private:

    SPMEPatch(const SPMEPatch& patch);
    SPMEPatch& operator=(const SPMEPatch& patch);

    // Patch dependent quantities
    IntVector localExtents;       //!< Number of grid points in each direction for this patch
    IntVector globalOffset;       //!< Grid point index of local 0,0,0 origin in global coordinates

    // Store the number of ghost cells  along each of the min/max boundaries
    // This lets us differentiate should we need to for centered and  left/right shifted splines
    IntVector posGhostExtents;    //!< Number of ghost cells on positive boundary
    IntVector negGhostExtents;     //!< Number of ghost cells on negative boundary

    SimpleGrid<double> theta;             //!<
    SimpleGrid<Matrix3> stressPrefactor;  //!<
    SimpleGrid<complex<double> > Q;       //!<

    ParticleSubset* pset;                 //!<
};

}  // End namespace Uintah

#endif
