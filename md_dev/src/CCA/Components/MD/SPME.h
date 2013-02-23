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

#ifndef UINTAH_MD_ELECTROSTATICS_SPME_H
#define UINTAH_MD_ELECTROSTATICS_SPME_H

#include <CCA/Components/MD/Electrostatics.h>
#include <CCA/Components/MD/CenteredCardinalBSpline.h>
#include <CCA/Components/MD/SimpleGrid.h>
#include <Core/Grid/Variables/Array3.h>

#include <vector>

#include <sci_defs/fftw_defs.h>

namespace Uintah {

typedef std::complex<double> dblcomplex;
typedef SimpleGrid<complex<double> > cdGrid;

using SCIRun::Vector;
using SCIRun::IntVector;

class MDSystem;
class Patch;
class ParticleSubset;
class IntVector;
class Point;
class Vector;
class Matrix3;
class MapPoint;

/**
 *  @class SPME
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper
 *  @date   January, 2013
 *
 *  @brief
 *
 *  @param
 */
class SPME : public Electrostatics {

  public:

    /**
     * @brief
     * @param
     */
    SPME();

    /**
     * @brief
     * @param
     */
    ~SPME();

    /**
     * @brief
     * @param
     * @param
     * @param
     * @param
     */
    SPME(const MDSystem* _System,
         const double _EwaldBeta,
         const bool _polarizable,
         const double _PolarizationTolerance,
         const SCIRun::IntVector& _K,
         const int splineOrder);

    /**
     * @brief
     * @param
     * @return
     */
    void initialize(const MDSystem* system,
                    const PatchSubset* patches,
                    const MaterialSubset* matls);

    /**
     * @brief
     * @param
     * @return
     */
    void setup();

    /**
     * @brief
     * @param
     * @return
     */
    void calculate();

    /**
     * @brief
     * @param
     * @return
     */
    void finalize();

    /**
     * @brief
     * @param
     * @return
     */
    inline ElectrostaticsType getType() const
    {
      return this->electrostaticMethod;
    }

  private:

    /**
     * @brief
     * @param
     * @return
     */
    std::vector<MapPoint> generateChargeMap(ParticleSubset* pset,
                                            CenteredCardinalBSpline& Spline);

    /**
     * @brief Map points (charges) onto the underlying grid.
     * @param
     * @return
     */
    void mapChargeToGrid(const std::vector<MapPoint>& GridMap,
                         ParticleSubset* pset,
                         int HalfSupport);

    /**
     * @brief Map forces from grid back to points.
     * @param
     * @return
     */
    void mapForceFromGrid(const std::vector<MapPoint>& GridMap,
                          ParticleSubset* pset,
                          int HalfSupport);

    /**
     * @brief
     * @param
     * @return
     */
    SimpleGrid<double> fC(const IntVector& gridExtents,
                          const IntVector& gridOffset,
                          const int numGhostCells,
                          const MDSystem& system);

    /**
     * @brief
     * @param
     * @return
     */
    SimpleGrid<dblcomplex> fB(const IntVector& gridExtents,
                              const MDSystem& system,
                              const int splineOrder);

    /**
     * @brief
     * @param
     * @return
     */
    vector<double> calculateOrdinalSpline(const int orderMinusOne,
                                          const int splineOrder);

    /**
     * @brief
     * @param
     * @return
     */
    vector<dblcomplex> generateBVector(const std::vector<double>& M,
                                       const int InitialIndex,
                                       const int LocalExtent,
                                       const CenteredCardinalBSpline&) const;
    /**
     * @brief Generates the local portion of the B grid (see. Essmann et. al., J. Phys. Chem. 103, p 8577, 1995)
     *          Equation 4.8
     * @param Extents - The number of internal grid points on the current processor
     * @param Offsets - The global mapping of the local (0,0,0) coordinate into the global grid index
     * @return A SimpleGrid<double> of B(m1,m2,m3)=|b1(m1)|^2 * |b2(m2)|^2 * |b3(m3)|^2
     */
    SimpleGrid<double> calculateBGrid(const SCIRun::IntVector& Extents,
                                      const SCIRun::IntVector& Offsets) const;

    /**
     * @brief Generates the local portion of the C grid (see. Essmann et. al., J. Phys. Chem. 103, p 8577, 1995)
     *          Equation 3.9
     * @param Extents - The number of internal grid points on the current processor
     * @param Offsets - The global mapping of the local (0,0,0) coordinate into the global grid index
     * @return A SimpleGrid<double> of C(m1,m2,m3)=(1/(PI*V))*exp(-PI^2*M^2/Beta^2)/M^2
     */
    SimpleGrid<double> calculateCGrid(const SCIRun::IntVector& Extents,
                                      const SCIRun::IntVector& Offsets) const;

    SimpleGrid<Matrix3> calculateStressPrefactor(const SCIRun::IntVector& Extents,
                                                 const SCIRun::IntVector& Offset);

    /**
     * @brief Generates split grid vector.
     *        Generates the vector of points from 0..K/2 in the first half of the array, followed by -K/2..-1
     * @param kMax - Maximum number of grid points for direction
     * @param InterpolatingSpline - CenteredCardinalBSpline that determines the number of wrapping points necessary
     * @return Returns a vector<double> of (0..[m=K/2],[K/2-K]..-1);
     */
    inline vector<double> generateMPrimeVector(unsigned int kMax,
                                               const CenteredCardinalBSpline& spline) const
    {
      int numPoints = kMax + spline.getSupport();  // For simplicity, store the whole vector
      std::vector<double> mPrime(numPoints);

      int halfSupport = spline.getHalfSupport();
      size_t halfMax = kMax / 2;

      // Pre wrap on the left and right sides as necessary for spline support
      std::vector<double> leftMost(mPrime[halfSupport]);

      // Seed internal array (without wrapping)
      for (size_t idx = 0; idx <= halfMax; ++idx) {
        leftMost[idx] = idx;
      }

      for (size_t idx = halfMax + 1; idx < kMax; ++idx) {
        leftMost[idx] = (double)(idx - kMax);
      }

      // Right end wraps into m=i portion
      for (int idx = 0; idx < halfSupport; ++idx) {
        mPrime[kMax + idx] = (double)idx;
      }

      // Left end wraps into m=i-KMax portion
      for (int idx = -3; idx < 0; ++idx) {
        leftMost[idx] = idx;  // i = KMax - abs(idx) = KMax + idx; i-KMax = KMax + idx - KMax = idx
      }

      return mPrime;
    }

    /**
     * @brief Generates reduced Fourier grid vector.
     *        Generates the vector of values i/K_i for i = 0..K-1
     * @param KMax - Maximum number of grid points for direction
     * @param InterpolatingSpline - CenteredCardinalBSpline that determines the number of wrapping points necessary
     * @return Returns a vector<double> of the reduced coordinates for the local grid along the input lattice direction
     */
    inline vector<double> generateMFractionalVector(unsigned int kMax,
                                                    const CenteredCardinalBSpline& interpolatingSpline) const
    {
      int numPoints = kMax + interpolatingSpline.getSupport();  // For simplicity, store the whole vector
      std::vector<double> mFractional(numPoints);
      int halfSupport = interpolatingSpline.getHalfSupport();

      //  Pre wrap on the left and right sides as necessary for spline support
      std::vector<double> leftMost(mFractional[halfSupport]);
      double kMaxInv = (double)kMax;
      for (size_t idx = -3; idx < 0; ++idx) {
        leftMost[-idx] = (static_cast<double>(kMax - idx)) * kMaxInv;
      }

      for (size_t idx = 0; idx < kMax; ++idx) {
        mFractional[idx] = (double)idx * kMaxInv;
      }

      std::vector<double> rightMost(mFractional[kMax]);
      for (int idx = 0; idx < halfSupport; ++idx) {
        rightMost[idx] = (double)idx * kMaxInv;
      }

      return mFractional;
    }

    // Values fixed on instantiation
    ElectrostaticsType electrostaticMethod;         //!< Implementation type for long range electrostatics
    double ewaldBeta;						                    //!< The Ewald calculation damping coefficient
    bool polarizable;				                    	  //!< Use polarizable Ewald formulation
    double polarizationTolerance;                   //!< Tolerance threshold for polarizable system
    SCIRun::IntVector kLimits;                      //!< Number of grid divisions in each direction
    CenteredCardinalBSpline interpolatingSpline;    //!< Spline object to hold info for spline calculation

//    // Patch dependent quantities
//    SCIRun::IntVector localGridExtents;             //!< Number of grid points in each direction for this patch
//    SCIRun::IntVector localGridOffset;		          //!< Grid point index of local 0,0,0 origin in global coordinates
//    SCIRun::IntVector localGhostPositiveSize;       //!< Number of ghost cells on positive boundary
//    SCIRun::IntVector localGhostNegativeSize;       //!< Number of ghost cells on negative boundary

    // Variables we'll get from the MDSystem instance to make life easier
    Matrix3 unitCell;           //!< Unit cell lattice parameters
    Matrix3 inverseUnitCell;    //!< Inverse lattice parameters
    double systemVolume;        //!< Volume of the unit cell

//    // Actually holds the data we're working with
//    SimpleGrid<double> fTheta;            //!<
//    SimpleGrid<Matrix3> stressPrefactor;  //!<
//    SimpleGrid<complex<double> > Q;       //!<

    // FFT Related variables
    fftw_complex forwardTransformPlan;    //!<
    fftw_complex backwardTransformPlan;   //!<

};

}  // End namespace Uintah

#endif
