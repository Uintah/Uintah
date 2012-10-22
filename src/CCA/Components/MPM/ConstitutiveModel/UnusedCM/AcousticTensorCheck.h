/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef __ACOUSTICTENSOR_CHECK_H__
#define __ACOUSTICTENSOR_CHECK_H__

#include "StabilityCheck.h"     
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Math/TangentModulusTensor.h>

namespace Uintah {

  /*! \class AcousticTensorCheck
   *  \brief Checks the loss of ellipticity/hyperbolicity using the 
   *         the determinant of the acoustic tensor.
   *  \author  Biswajit Banerjee, \n
   *           C-SAFE and Department of Mechanical Engineering,\n
   *           University of Utah.\n

   The acoustic tensor (\f$ \mathbf{A} \f$) is given by
   \f[
   \mathbf{A} = \mathbf{n}\bullet\mathbf{C}\bullet\mathbf{n}
   \f]
   where \f$ \mathbf{n} \f$ is a unit vector in the direction normal to the 
   instability, and \f$ \mathbf{C} \f$ is the algorithmic tangent modulus. 

   Material loses ellipticity when \f$ \det\mathbf{A} \f$ changes sign from 
   positive to negative.

   Based on Tahoe implementation of DetCheckT.h v.1.14.
  */
  class AcousticTensorCheck : public StabilityCheck {

  public:
         
    //! Construct an object that can be used to check stability
    AcousticTensorCheck(ProblemSpecP& ps);
    AcousticTensorCheck(const AcousticTensorCheck* atc);

    //! Destructor of stability check
    ~AcousticTensorCheck();
         
    /*! Check the stability and return the direction of instability
      if any */
    bool checkStability(const Matrix3& cauchyStress,
                        const Matrix3& deformRate,
                        const TangentModulusTensor& tangentModulus,
                        Vector& direction);

  private:

    /*! Check for localization from
     *  the ellipticity of the tangent modulus
     *  \return true if the acoustic tensor is not positive definite
     *  false otherwise.  Also return normal to surface of localization. */
    bool isLocalized(const TangentModulusTensor& tangentModulus,
                     Vector& normal);

    /*! Find approximate local minima */
    void findApproxLocalMins(double** detA, 
                             int** localmin, 
                             const TangentModulusTensor& C);

    /*! Form the acoustic tensor 
     *  The Acoustic tensor is given by 
        \f[
           A_{ik}(\mathbf{n}) = (1/\rho) C_{ijkl} n_j n_l
        \f]
        where \f$ \rho \f$ is the density,
              \f$ \mathbf{n} \f$ is a vector, and
              \f$ \mathbf{C} \f$ is the tangent modulus tensor.
    */
    void formAcousticTensor(const Vector& normal,
                            const TangentModulusTensor& C,
                            Matrix3& A);

    /*! Choose new normal */
    Vector chooseNewNormal(Vector& prevnormal, 
                           Matrix3& J);

    /*! Choose normal from normal set */
    Vector chooseNormalFromNormalSet(vector<Vector>& normalSet, 
                                     const TangentModulusTensor& C);

    // Prevent copying of this class and copy constructor
    //AcousticTensorCheck(const AcousticTensorCheck &);
    AcousticTensorCheck& operator=(const AcousticTensorCheck &);

  private:

    int d_sweepInc; /** Incremental angle of sweep. 
                        Should be an integral divisor of 360 */
    int d_numTheta; /** Number of checks in the theta direction.
                        Should 360/sweepIncrement and should be even */
    int d_numPhi;   /** Number of checks in the phi direction.
                        Should 90/sweepIncrement+1*/
  };
} // End namespace Uintah
      
#endif  // __ACOUSTICTENSOR_CHECK_H__

