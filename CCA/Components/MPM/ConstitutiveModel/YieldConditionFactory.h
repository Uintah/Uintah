#ifndef _YIELDCONDITIONFACTORY_H_
#define _YIELDCONDITIONFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  // Forward declarations
  class YieldCondition;

  /*! \class YieldConditionFactory
   *  \brief Creates instances of Yield Conditions
   *  \author  Biswajit Banerjee,
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \author  Copyright (C) 2003 Container Dynamics Group
   *  \warning Currently implemented yield conditions:
   *           von Mises, Gurson-Tvergaard-Needleman, Rousselier
  */

  class YieldConditionFactory {

  public:

    //! Create a yield condition from the input file problem specification.
    static YieldCondition* create(ProblemSpecP& ps);
  };
} // End namespace Uintah
      
#endif /* _YIELDCONDITIONFACTORY_H_ */
