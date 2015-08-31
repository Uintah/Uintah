/*
 *
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
 *
 * ----------------------------------------------------------
 * Util.h
 *
 *  Created on: May 6, 2014
 *      Author: jbhooper
 */

#ifndef MDUTIL_H_
#define MDUTIL_H_

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>

#include <Core/Math/Matrix3.h>

#include <Core/Grid/Variables/VarLabel.h>

#include <CCA/Components/MD/SimpleGrid.h>

#include <vector>

namespace Uintah {

  typedef SimpleGrid<double>                        doubleGrid;
  typedef SimpleGrid<SCIRun::Vector>                vectorGrid;
  typedef SimpleGrid<Uintah::Matrix3>               matrixGrid;

  class MDConstants {
    public:
      // Physical constants
      static const double Avogadro;
      static const double kB_SI;                        // Boltzmann's constant
      static const double epsilon_0;
      static const double PI;                           // Pi

      static const double PI2;                          // Pi^2
      static const double PI_Over_2;                    // Pi/2
      static const double rootPI;                       // sqrt(Pi)
      static const double orthogonalAngle;              // 90 degrees
      static const double degToRad;                     // (Pi/2) / 180.0
      static const double radToDeg;                     // 180.0  / (Pi/2)
      static const double zeroTol;                      // Double approx. of zero
      // Compound constants in internal units
      static const double electrostaticForceConstant;   // 1/4PiEps_0 in internal units
      static const double kB;

      static const double defaultDipoleMixRatio;
      static const double defaultPolarizationTolerance;
      static const SCIRun::IntVector    IV_ZERO;
      static const SCIRun::IntVector    IV_ONE;
      static const SCIRun::IntVector    IV_X;
      static const SCIRun::IntVector    IV_Y;
      static const SCIRun::IntVector    IV_Z;
      static const SCIRun::Vector       V_ZERO;
      static const SCIRun::Vector       V_ONE;
      static const SCIRun::Vector       V_X;
      static const SCIRun::Vector       V_Y;
      static const SCIRun::Vector       V_Z;
      static const Uintah::Matrix3      M3_I;
      static const Uintah::Matrix3      M3_0;
  };


}




#endif /* MDUTIL_H_ */
