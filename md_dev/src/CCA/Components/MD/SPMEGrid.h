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

#ifndef UINTAH_MD_SPMEGRID_H
#define UINTAH_MD_SPMEGRID_H

#include <CCA/Components/MD/SimpleGrid.h>
#include <CCA/Components/MD/MapPoint.h>
#include <Core/Math/Matrix3.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <complex>

namespace Uintah {

using Uintah::Matrix3;

typedef SimpleGrid<std::complex<double> > cdGrid;

/**
 *  @class SPMEGrid
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper
 *  @date   December, 2012
 *
 *  @brief
 *
 *  @param T The data type for this SPMEGrid. Should be double or std::complex<double>>.
 */
template<class T> class SPMEGrid {

  public:

    /**
     * @brief Map points (charges) onto the underlying grid.
     * @param
     * @return
     */
    SPMEGrid<T>& mapChargeToGrid(const SimpleGrid<std::vector<MapPoint<T> > > gridMap,
                                 const ParticleSubset& globalParticleList);

    /**
     * @brief Map forces from grid back to points.
     * @param
     * @return
     */
    SPMEGrid<T>& mapForceFromGrid(const SimpleGrid<std::vector<MapPoint<T> > > gridMap,
                                  ParticleSubset& globalParticleList) const;

    /**
     * @brief Multiply Q with the specified SimpleGrid.
     * @param gridIn The multiplier
     * @return SPMEGrid<T>& The result (*this) of the in-place multiplication.
     */
    inline SPMEGrid<T>& multiplyInPlace(const cdGrid& gridIn)
    {
      Q *= gridIn;
      d_fieldValid = false;
      return *this;
    }

    /**
     * @brief Calculates local energy of charge grid.
     * @param
     * @param
     * @param
     * @param
     * @param
     * @param
     * @return
     */
    double calculateEnergyAndStress(const vector<double>& M1,
                                    const vector<double>& M2,
                                    const vector<double>& M3,
                                    Matrix3 stressTensor,
                                    const cdGrid& stressPrefactor,
                                    const cdGrid& thetaRecip);

    /**
     * @brief
     * @param
     * @return
     */
    SPMEGrid<T>& calculateField();  // if (FieldValid == false) { calculate field grid; set FieldValid == true; return *this; }

    /**
     * @brief Transforms 'Q' from real to fourier space
     * @param
     * @return
     */
    SPMEGrid<T>& inPlaceFFT_RealToFourier(/*Data for FFTW3 routine goes here */);

    /**
     * @brief Transforms 'Q' from fourier to real space
     * @param
     * @return
     */
    SPMEGrid<T>& inPlaceFFT_FourierToReal(/*Data for FFTW3 routine goes here */);

   /**
    * @brief Pass through indexing of Value array.
    * @param x The x component of the 3D cell index.
    * @param y The y component of the 3D cell index.
    * @param z The z component of the 3D cell index.
    * @return T& A reference to the value at index [x,y,z] in this SimpleGrid.
    */
   inline Vector operator()(const int& x,
                             const int& y,
                             const int& z)
   {
     return this->d_field(x,y,z);
   }

   /**
    * @brief Pass through indexing of Value array.
    * @param x The x component of the 3D cell index.
    * @param y The y component of the 3D cell index.
    * @param z The z component of the 3D cell index.
    * @return T The value at index [x,y,z] in this SimpleGrid.
    */
    inline Vector operator()(const int& x,
                             const int& y,
                             const int& z) const
   {
     return this->d_field(x, y, z);
   }

  private:
    cdGrid Q;                    //!<
    SimpleGrid<Vector> d_field;  //!<
    bool d_fieldValid;           //!< =false;

    // Alan - Must deal with ghosts here in the same way as in SimpleGrid;  Alternatively we can get rid of extent/offset for SPME_Grid
    //        entirely and pass through extent/offset calls to the underlying SimpleGrid data type.  If we do this, then we need to ensure
    //        registration of Q and Field extent/offset on construction.  This probably isn't a bad idea, since ideally we will construct
    //        an SPME_Grid instance with either Q set and Field zero'ed out, or both Q and Field zero'd out, so we can just construct
    //        Field as a subprocess of constructing the SPME_Grid object by extracting the relevant data from the Q grid and passing it to
    //        Field's constructor.  This is just more complex construction than I'm used to, so I'm not going to even take a stab at that here.

};

}  // End namespace Uintah

#endif
