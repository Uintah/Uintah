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

#ifndef UINTAH_MD_ELECTROSTATICS_SPME_H
#define UINTAH_MD_ELECTROSTATICS_SPME_H

#include <CCA/Components/MD/Electrostatics.h>
#include <CCA/Components/MD/CenteredCardinalBSpline.h>
#include <CCA/Components/MD/SimpleGrid.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/MD/SPMEPatch.h>
#include <Core/Grid/Variables/ComputeSet.h>

#include <vector>

#include <sci_defs/fftw_defs.h>

namespace Uintah {

using namespace SCIRun;

typedef std::complex<double> dblcomplex;
typedef SimpleGrid<complex<double> > cdGrid;

class MDSystem;
class SPMEMapPoint;
class ParticleSubset;

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
    SPME(MDSystem* system,
         const double ewaldBeta,
         const bool polarizable,
         const double polarizationTolerance,
         const IntVector& kLimits,
         const int splineOrder);

    /**
     * @brief
     * @param
     * @param
     * @param
     * @return
     */
    void initialize(const PatchSubset* patches,
                    const MaterialSubset* materials,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

    /**
     * @brief
     * @param None
     * @return None
     */
    void setup();

    /**
     * @brief
     * @param None
     * @return None
     */
    void calculate();

    /**
     * @brief
     * @param None
     * @return None
     */
    void finalize();

    /**
     * @brief
     * @param None
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
     * @param
     * @return
     */
    std::vector<SPMEMapPoint> generateChargeMap(ParticleSubset* pset,
                                                CenteredCardinalBSpline& spline);

    /**
     * @brief Map points (charges) onto the underlying grid.
     * @param
     * @param
     * @param
     * @return
     */
    void mapChargeToGrid(const std::vector<SPMEMapPoint>& gridMap,
                         ParticleSubset* pset,
                         int halfSupport);

    /**
     * @brief Map forces from grid back to points.
     * @param
     * @return
     */
    void mapForceFromGrid(const std::vector<SPMEMapPoint>& gridMap,
                          ParticleSubset* pset,
                          int halfSupport);

    /**
     * @brief
     * @param
     * @param
     * @return
     */
    vector<double> calculateOrdinalSpline(const int orderMinusOne,
                                          const int splineOrder);

    /**
     * @brief
     * @param
     * @param
     * @param
     * @param
     * @return
     */
    vector<dblcomplex> generateBVector(const std::vector<double>& M,
                                       const int initialIndex,
                                       const int localExtent,
                                       const CenteredCardinalBSpline& spline) const;
    /**
     * @brief Generates the local portion of the B grid (see. Essmann et. al., J. Phys. Chem. 103, p 8577, 1995)
     *          Equation 4.8
     * @param Extents - The number of internal grid points on the current processor
     * @param Offsets - The global mapping of the local (0,0,0) coordinate into the global grid index
     *
     * @return A SimpleGrid<double> of B(m1,m2,m3)=|b1(m1)|^2 * |b2(m2)|^2 * |b3(m3)|^2
     */
    SimpleGrid<double> calculateBGrid(const IntVector& extents,
                                      const IntVector& offsets) const;

    /**
     * @brief Generates the local portion of the C grid (see. Essmann et. al., J. Phys. Chem. 103, p 8577, 1995)
     *          Equation 3.9
     * @param Extents - The number of internal grid points on the current processor
     * @param Offsets - The global mapping of the local (0,0,0) coordinate into the global grid index
     *
     * @return A SimpleGrid<double> of C(m1,m2,m3)=(1/(PI*V))*exp(-PI^2*M^2/Beta^2)/M^2
     */
    SimpleGrid<double> calculateCGrid(const IntVector& extents,
                                      const IntVector& offsets) const;

    /**
     * @brief
     * @param extents - The number of internal grid points on the current processor
     * @param offsets - The global mapping of the local (0,0,0) coordinate into the global grid index
     *
     * @return A SimpleGrid<Matrix3>
     */
    SimpleGrid<Matrix3> calculateStressPrefactor(const IntVector& extents,
                                                 const IntVector& offset);

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
     *        Generates the vector of values i/K_i for i = 0...K-1
     * @param KMax - Maximum number of grid points for direction
     * @param InterpolatingSpline - CenteredCardinalBSpline that determines the number of wrapping points necessary
     * @return Returns a vector<double> of the reduced coordinates for the local grid along the input lattice direction
     */
    inline vector<double> generateMFractionalVector(size_t kMax,
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
    MDSystem* system;                               //!<
    double ewaldBeta;						                    //!< The Ewald calculation damping coefficient
    bool polarizable;				                    	  //!< Use polarizable Ewald formulation
    double polarizationTolerance;                   //!< Tolerance threshold for polarizable system
    IntVector kLimits;                              //!< Number of grid divisions in each direction
    CenteredCardinalBSpline interpolatingSpline;    //!< Spline object to hold info for spline calculation
    std::vector<SPMEPatch> spmePatches;             //!< Assuming multiple patches, these are the pieces of the SPME grid

    // Variables we'll get from the MDSystem instance to make life easier
    Matrix3 unitCell;           //!< Unit cell lattice parameters
    Matrix3 inverseUnitCell;    //!< Inverse lattice parameters
    double systemVolume;        //!< Volume of the unit cell

    // FFT Related variables
    fftw_complex forwardTransformPlan;    //!<
    fftw_complex backwardTransformPlan;   //!<

};

}  // End namespace Uintah

#endif
