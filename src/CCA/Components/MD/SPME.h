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
class MapPoint;
class CenteredCardinalBSpline;

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
    SPME(int _numGridPoints,
         int _numGhostCells,
         int _splineOrder,
         bool _polarizable,
         double _ewaldBeta);

    /**
     * @brief
     * @param
     * @return
     */
    void initialize();

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
    inline ElectroStaticsType getType() const
    {
      return this->electrostaticsType;
    }

    friend class MD;

  private:

    /**
     * @brief
     * @param
     * @return
     */
    void performSPME(const MDSystem& system,
                     const Patch* patch,
                     ParticleSubset* pset);

    /**
     * @brief
     * @param
     * @return
     */
    std::vector<std::vector<MapPoint> > createChargeMap(ParticleSubset* pset,
                                                        CenteredCardinalBSpline& spline);

    /**
     * @brief
     * @param
     * @return
     */
    std::vector<Point> calcReducedCoords(const std::vector<Point>& localRealCoordinates,
                                         const MDSystem& system);

    /**
     * @brief No ghost points
     * @param
     * @return
     */
    void calculateStaticGrids(const IntVector& gridExtents,
                              const IntVector& offset,
                              const MDSystem& system,
                              SimpleGrid<double>& fBGrid,
                              SimpleGrid<double>& fCGrid,
                              SimpleGrid<double>& fStressPre,
                              int splineOrder,
                              const std::vector<double>& M1,
                              const std::vector<double>& M2,
                              const std::vector<double>& M3);

    /**
     * @brief Map points (charges) onto the underlying grid.
     * @param
     * @return
     */
    SimpleGrid<double>& mapChargeToGrid(const std::vector<std::vector<MapPoint> > gridMap,
                                        const ParticleSubset* globalParticleList);

    /**
     * @brief Map forces from grid back to points.
     * @param
     * @return
     */
    SimpleGrid<double>& mapForceFromGrid(const std::vector<std::vector<MapPoint> > gridMap,
                                         ParticleSubset* globalParticleList);

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
    vector<dblcomplex> generateBVector(int numPoints,
                                       const std::vector<double>& M,
                                       int max,
                                       int splineOrder,
                                       const std::vector<double>* splineCoeff);

    /**
     * @brief
     * @param
     * @return
     */
    inline vector<double> generateMPrimeVector(unsigned int points,
                                               int shift,
                                               int max) const
    {
      std::vector<double> m(points);
      int halfMax = max / 2;

      for (size_t i = 0; i < points; ++i) {
        m[i] = i + shift;
        if (m[i] > halfMax) {
          m[i] -= max;
        }
      }
      return m;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline vector<double> generateMVector(unsigned int points,
                                          int shift,
                                          int max) const
    {
      std::vector<double> m(points);
      int halfMax = max / 2;

      for (size_t i = 0; i < points; ++i) {
        m[i] = i + shift;
        if (m[i] > halfMax) {
          m[i] -= max;
        }
      }
      return m;
    }

    cdGrid Q;
    ElectroStaticsType electrostaticsType;   //!<
    int numGridPoints;                       //!<
    int numGhostCells;                       //!<
    int splineOrder;                         //!<
    bool polarizable;                        //!<
    double ewaldBeta;                        //!< The Ewald damping coefficient

};

}  // End namespace Uintah

#endif
