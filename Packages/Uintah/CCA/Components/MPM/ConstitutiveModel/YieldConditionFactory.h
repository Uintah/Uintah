#ifndef _YIELDCONDITIONFACTORY_H_
#define _YIELDCONDITIONFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  // Forward declarations
  class YieldCondition;
  class MPMLabel;

  //! YieldConditionFactory
  /*!
    Creates instances of Yield Conditions
 
    Biswajit Banerjee,
    C-SAFE and Department of Mechanical Engineering,
    University of Utah.
   
    Copyright (C) 2003 Container Dynamics Group
 
    KEYWORDS :
    Yield Conditions, von Mises, Gurson-Tvergaard-Needleman, Rousselier
 
    DESCRIPTION :
    Provides A class to create instances of various yield conditions.
  */

  class YieldConditionFactory {

  public:

    //! Create a yield condition from the input file problem specification.
    static YieldCondition* create(ProblemSpecP& ps);
  };
} // End namespace Uintah
      
#endif /* _YIELDCONDITIONFACTORY_H_ */
