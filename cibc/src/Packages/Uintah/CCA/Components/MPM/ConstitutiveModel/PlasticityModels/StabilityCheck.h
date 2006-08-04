#ifndef __STABILITY_CHECK_H__
#define __STABILITY_CHECK_H__

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

namespace Uintah {

  /*! \class StabilityCheck
    \brief  A generic wrapper for various methods of checking stability. 
   
    \author  Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering,\n
    University of Utah.\n
    Copyright (C) 2003 Container Dynamics Group\n

    Examples: loss of hyperbolicity/ellipticity, Drucker 
    stability criterion, Hill condition etc.
    Provides an abstract base class for various methods of checking 
    the stability of motion/bifurcation points
  */
  class StabilityCheck {

  public:
	 
    //! Construct an object that can be used to check stability
    StabilityCheck();

    //! Destructor of stability check
    virtual ~StabilityCheck();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

    // Determine if we do the stability.  Instead of checking for the
    // existence of d_stable in ElasticPlastic.cc, instead check 
    // do() which is true except for NoneCheck.cc.
    virtual bool doIt() {
      return true;
    }
	 
    /*! Check the stability and return the direction of instability
      if any */
    virtual bool checkStability(const Matrix3& cauchyStress,
				const Matrix3& deformRate,
				const TangentModulusTensor& tangentModulus,
				Vector& direction) = 0;
  };
} // End namespace Uintah
      
#endif  // __STABILITY_CHECK_H__

