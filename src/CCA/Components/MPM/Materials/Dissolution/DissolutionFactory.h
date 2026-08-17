/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

#ifndef _DISSOLUTIONFACTORY_H_
#define _DISSOLUTIONFACTORY_H_

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/MaterialManagerP.h>

namespace Uintah {

  class Dissolution;
  class MPMLabel;
  class MPMFlags;

  class DissolutionFactory
  {
  public:
        
    // this function has a switch for all known mat_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static Dissolution* create(const ProcessorGroup* myworld,
                               const ProblemSpecP& ps,MaterialManagerP& ss,
                               MPMLabel* lb, MPMFlags* MFlag);
  };
} // End namespace Uintah
  
#endif /* _DISSOLUTIONFACTORY_H_ */
