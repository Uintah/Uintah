#ifndef __ACOUSTICTENSOR_CHECK_H__
#define __ACOUSTICTENSOR_CHECK_H__

#include "StabilityCheck.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>

namespace Uintah {

  /*! \class AcousticTensorCheck
   *  \brief Checks the loss of ellipticity/hyperbolicity using the 
   *         the determinant of the acoustic tensor.
   *  \author  Biswajit Banerjee, \n
   *           C-SAFE and Department of Mechanical Engineering,\n
   *           University of Utah.\n
   *           Copyright (C) 2003 Container Dynamics Group\n

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

    //! Destructor of stability check
    ~AcousticTensorCheck();
	 
    /*! Check the stability and return the direction of instability
      if any */
    bool checkStability(const Matrix3& cauchyStress,
                        const Matrix3& strainMeasure,
                        TangentModulusTensor& tangentModulus,
                        Vector& direction);

  private:

    /*! Check for localization from
     *  the ellipticity of the tangent modulus
     *  \return true if the acoustic tensor is not positive definite
     *  false otherwise.  Also return normal to surface of localization. */
    bool isLocalized(TangentModulusTensor& tangentModulus,
                     Vector& normal);

    /*! Find approximate local minima */
    void findApproxLocalMins(double** detA, 
                             int** localmin, 
                             TangentModulusTensor& C);

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
                            TangentModulusTensor& C,
                            Matrix3& A);

    /*! Choose new normal */
    Vector chooseNewNormal(Vector& prevnormal, 
                           Matrix3& J);

    /*! Choose normal from normal set */
    Vector chooseNormalFromNormalSet(vector<Vector>& normalSet, 
                                     TangentModulusTensor& C);

    // Prevent copying of this class and copy constructor
    AcousticTensorCheck(const AcousticTensorCheck &);
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

