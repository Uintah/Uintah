#ifndef __DRUCKERBECKER_CHECK_H__
#define __DRUCKERBECKER_CHECK_H__

#include "StabilityCheck.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>

namespace Uintah {

  /*! \class DruckerBeckerCheck
   *  \brief Checks the loss of ellipticity/hyperbolicity using both the 
   *         the Drucker and Becker stability postulate.
   *  \author  Biswajit Banerjee, \n
   *           C-SAFE and Department of Mechanical Engineering,\n
   *           University of Utah.\n
   *           Copyright (C) 2003 Container Dynamics Group\n
   */
  class DruckerBeckerCheck : public StabilityCheck {

  public:
	 
    //! Construct an object that can be used to check stability
    DruckerBeckerCheck(ProblemSpecP& ps);

    //! Destructor of stability check
    ~DruckerBeckerCheck();
	 
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
    DruckerBeckerCheck(const DruckerBeckerCheck &);
    DruckerBeckerCheck& operator=(const DruckerBeckerCheck &);
  };
} // End namespace Uintah
      
#endif  // __DRUCKERBECKER_CHECK_H__

