#ifndef _STABILITYCHECKFACTORY_H_
#define _STABILITYCHECKFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  // Forward declarations
  class StabilityCheck;

  /*! \class StabilityCheckFactory
   *  \brief Creates instances of stability check methods
   *  \author  Biswajit Banerjee,
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \author  Copyright (C) 2003 Container Dynamics Group
   *  \warning Currently implemented stability checks
   *           Acoustic tensor (loss of ellipticity/hyperbolicity)
  */

  class StabilityCheckFactory {

  public:

    //! Create a yield condition from the input file problem specification.
    static StabilityCheck* create(ProblemSpecP& ps);
  };
} // End namespace Uintah
      
#endif /* _STABILITYCHECKFACTORY_H_ */
