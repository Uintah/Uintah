#ifndef __DRUCKER_CHECK_H__
#define __DRUCKER_CHECK_H__

#include "StabilityCheck.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>

namespace Uintah {

  /*! \class DruckerCheck
   *  \brief Checks the loss of ellipticity/hyperbolicity using the 
   *         the Drucker stability postulate.
   *  \author  Biswajit Banerjee, \n
   *           C-SAFE and Department of Mechanical Engineering,\n
   *           University of Utah.\n
   *           Copyright (C) 2003 Container Dynamics Group\n

   The material is assumed to become unstable when
   \f[
      \dot\sigma:D^p \le 0
   \f]
  */
  class DruckerCheck : public StabilityCheck {

  public:
	 
    //! Construct an object that can be used to check stability
    DruckerCheck(ProblemSpecP& ps);

    //! Destructor of stability check
    ~DruckerCheck();
	 
    /*! Check the stability.

      \return true if unstable
      \return false if stable
    */
    bool checkStability(const Matrix3& stress,
                        const Matrix3& deformRate,
                        const TangentModulusTensor& tangentModulus,
                        Vector& direction);

  private:


    // Prevent copying of this class and copy constructor
    DruckerCheck(const DruckerCheck &);
    DruckerCheck& operator=(const DruckerCheck &);
  };
} // End namespace Uintah
      
#endif  // __DRUCKER_CHECK_H__

