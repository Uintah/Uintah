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
#include <Core/Grid/Variables/ParticleVariable.h>

#include <vector>

#include <sci_defs/fftw_defs.h>

namespace Uintah {

using namespace SCIRun;

typedef std::complex<double> dblcomplex;
typedef int particleIndex;
typedef int particleId;

class MDSystem;
class SPMEMapPoint;
class ParticleSubset;
class MDLabel;

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
    void initialize();

    /**
     * @brief
     * @param None
     * @return None
     */
    void setup(const ProcessorGroup* pg,
               const PatchSubset* patches,
               const MaterialSubset* materials,
               DataWarehouse* old_dw,
               DataWarehouse* new_dw);

    /**
     * @brief
     * @param
     * @return None
     */
    void calculate(const ProcessorGroup* pg,
                   const PatchSubset* patches,
                   const MaterialSubset* materials,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw);

    /**
     * @brief
     * @param None
     * @return None
     */
    void finalize(const ProcessorGroup* pg,
                  const PatchSubset* patches,
                  const MaterialSubset* materials,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw);

    /**
     * @brief
     * @param None
     * @return
     */
    inline ElectrostaticsType getType() const
    {
      return d_electrostaticMethod;
    }

    /**
     * @brief
     * @param None
     * @return
     */
    inline void setMDLabel(MDLabel* lb)
    {
      d_lb = lb;
    }

  private:

    /**
     * @brief
     * @param
     * @param
     * @return
     */
    std::vector<SPMEMapPoint> generateChargeMap(ParticleSubset* pset,
                                                constParticleVariable<Point> particlePositions,
                                                constParticleVariable<long64> particleIDs,
                                                CenteredCardinalBSpline& spline);

    /**
     * @brief Map points (charges) onto the underlying grid.
     * @param
     * @param
     * @param
     * @return
     */
    void mapChargeToGrid(SPMEPatch* spmePatch,
                         const std::vector<SPMEMapPoint>& gridMap,
                         ParticleSubset* pset,
                         constParticleVariable<double> charges,
                         int halfSupport);

    /**
     * @brief Map forces from grid back to points.
     * @param
     * @return
     */
    void mapForceFromGrid(SPMEPatch* spmePatch,
                          const std::vector<SPMEMapPoint>& gridMap,
                          ParticleSubset* pset,
                          ParticleVariable<Vector> pforcenew,
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
     * @return A SimpleGrid<Matrix3> pointer.
     */
    SimpleGrid<Matrix3>* calculateStressPrefactor(const IntVector& extents,
                                                  const IntVector& offset);

    /**
     * @brief Generates split grid vector.
     *        Generates the vector of points from 0..K/2 in the first half of the array, followed by -K/2..-1
     * @param kMax - Maximum number of grid points for direction
     * @param spline - CenteredCardinalBSpline that determines the number of wrapping points necessary
     * @return std::vector<double> of (0..[m=K/2],[K/2-K]..-1);
     */
    inline std::vector<double> generateMPrimeVector(unsigned int kMax,
                                                    const CenteredCardinalBSpline& spline) const
    {
      int halfSupport = spline.getHalfMaxSupport();
      int numPoints = kMax + 2 * halfSupport;  // For simplicity, store the whole vector
      std::vector<double> mPrime(numPoints);

      size_t halfMax = kMax / 2;

      // Seed internal array (without wrapping)
      int TrueZeroIndex = halfSupport;  // The zero index of the unwrapped array embedded in the wrapped array

      for (int idx = TrueZeroIndex; idx <= halfMax + TrueZeroIndex; ++idx) {
        mPrime[idx] = static_cast<double>(idx - TrueZeroIndex);
      }
      for (int Index = halfMax + TrueZeroIndex + 1; Index < TrueZeroIndex + kMax; ++Index) {
        mPrime[Index] = static_cast<double>(Index - TrueZeroIndex - static_cast<int>(kMax));
      }

      // Wrapped ends of the vector
      for (int idx = 1; idx <= halfSupport; ++idx) {
        // Left wrapped end
        mPrime[TrueZeroIndex - idx] = mPrime[TrueZeroIndex - idx + kMax];
        mPrime[TrueZeroIndex + kMax + idx - 1] = mPrime[TrueZeroIndex + idx - 1];  //-1 offsets for 0 based array
      }

      return mPrime;
    }
    /**
     * @brief Generates reduced Fourier grid vector. Generates the vector of values i/K_i for i = 0...K-1
     * @param KMax - Maximum number of grid points for direction
     * @param InterpolatingSpline - CenteredCardinalBSpline that determines the number of wrapping points necessary
     * @return std::vector<double> of the reduced coordinates for the local grid along the input lattice direction
     */
    inline std::vector<double> generateMFractionalVector(size_t kMax,
                                                         const CenteredCardinalBSpline& interpolatingSpline) const
    {
      int halfSupport = interpolatingSpline.getHalfMaxSupport();
      int numPoints = kMax + 2 * halfSupport;  // For simplicity, store the whole vector
      std::vector<double> mFractional(numPoints);

      double kMaxInv = 1.0 / static_cast<double>(kMax);

      int TrueZeroIndex = halfSupport;  // Location of base array zero index
      for (int idx = TrueZeroIndex; idx < TrueZeroIndex + kMax; ++idx) {
        mFractional[idx] = static_cast<double>(idx - TrueZeroIndex) * kMaxInv;
      }

      // Wrapped ends of the vector
      for (size_t idx = 1; idx <= halfSupport; ++idx) {
        // Left wrapped end
        mFractional[TrueZeroIndex - idx] = mFractional[TrueZeroIndex - idx + kMax];
        mFractional[TrueZeroIndex + kMax + idx - 1] = mFractional[TrueZeroIndex + idx - 1];  //-1 offsets for 0 based array
      }

      return mFractional;
    }
    /**
     * @brief Perform all calculations preceding the FFT transform of the charge grid to Fourier space.
     * @param const ProcessorGroup* pg -- All processors processing SPME patches
     * @param const PatchSubset* patches -- Patches to be processed by this thread
     * @param const MaterialSubset* materials -- Material subset belonging to this patch
     * @param DataWarehouse* old_dw -- Last time step's data warehouse
     * @param DataWarehouse* new_dw -- This time step's data warehouse
     * @return None
     */
    void calculatePreTransform(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* materials,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);

    /**
     * @brief Perform all Fourier space calculations.
     * @param const ProcessorGroup* pg -- All processors processing SPME patches
     * @param const PatchSubset* patches -- Patches to be processed by this thread
     * @param const MaterialSubset* materials -- Material subset belonging to this patch
     * @param DataWarehouse* old_dw -- Last time step's data warehouse
     * @param DataWarehouse* new_dw -- This time step's data warehouse
     * @return None
     */
    void calculateInFourierSpace(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* materials,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw);

    /**
     * @brief Perform calculations proceeding the FFT transform from Fourier to real space.
     * @param const ProcessorGroup* pg -- All processors processing SPME patches
     * @param const PatchSubset* patches -- Patches to be processed by this thread
     * @param const MaterialSubset* materials -- Material subset belonging to this patch
     * @param DataWarehouse* old_dw -- Last time step's data warehouse
     * @param DataWarehouse* new_dw -- This time step's data warehouse
     * @return None
     */
    void calculatePostTransform(const ProcessorGroup* pg,
                                const PatchSubset* patches,
                                const MaterialSubset* materials,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);
    /**
     * @brief Perform necessary operation to transform Q grid to fourier space
     * @param const ProcessorGroup* pg -- All processors processing SPME patches
     * @param const PatchSubset* patches -- Patches to be processed by this thread
     * @param const MaterialSubset* materials -- Material subset belonging to this patch
     * @param DataWarehouse* old_dw -- Last time step's data warehouse
     * @param DataWarehouse* new_dw -- This time step's data warehouse
     * @return None
     *
     */
    void transformRealToFourier(const ProcessorGroup* pg,
                                const PatchSubset* patches,
                                const MaterialSubset* materials,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);
    /**
     * @brief Perform necessary operation to transform Q grid to fourier space
     * @param const ProcessorGroup* pg -- All processors processing SPME patches
     * @param const PatchSubset* patches -- Patches to be processed by this thread
     * @param const MaterialSubset* materials -- Material subset belonging to this patch
     * @param DataWarehouse* old_dw -- Last time step's data warehouse
     * @param DataWarehouse* new_dw -- This time step's data warehouse
     * @return None
     *
     */
    void transformFourierToReal(const ProcessorGroup*pg,
                                const PatchSubset* patches,
                                const MaterialSubset* materials,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    /**
     * @brief Checks for convergence of polarizability calculation
     * @param None
     * @return Bool - true if converged, false if not
     */
    bool checkConvergence();

    /**
     * @brief
     * @param
     * @return
     */
    inline bool getPolarizableCalculation() const
    {
      return d_polarizable;
    }

    // Values fixed on instantiation
    ElectrostaticsType d_electrostaticMethod;         //!< Implementation type for long range electrostatics
    MDSystem* d_system;                               //!< A handle to the MD simulation system object
    MDLabel* d_lb;                                    //!< A handle on the set of MD specific labels
    double d_ewaldBeta;						                    //!< The Ewald calculation damping coefficient
    bool d_polarizable;				                    	  //!< Use polarizable Ewald formulation
    double d_polarizationTolerance;                   //!< Tolerance threshold for polarizable system
    IntVector d_kLimits;                              //!< Number of grid divisions in each direction
    CenteredCardinalBSpline d_interpolatingSpline;    //!< Spline object to hold info for spline calculation
    std::vector<SPMEPatch*> d_spmePatches;            //!< Assuming multiple patches, these are the pieces of the SPME grid

    // Variables we'll get from the MDSystem instance to make life easier
    Matrix3 d_unitCell;           //!< Unit cell lattice parameters
    Matrix3 d_inverseUnitCell;    //!< Inverse lattice parameters
    double d_systemVolume;        //!< Volume of the unit cell

    // FFT Related variables
    fftw_plan d_forwardTransformPlan;    //!< FFTW plan for how the forward  3D transform will be executed
    fftw_plan d_backwardTransformPlan;   //!< FFTW plan for how the backward 3D transform will be executed

};

}  // End namespace Uintah

#endif
