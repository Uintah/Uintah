#ifndef _TURBULENCEFACTORY_H_
#define _TURBULENCEFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>

namespace Uintah {

  class ICELabel;
  class Turbulence;

  class TurbulenceFactory
  {
  public:
    TurbulenceFactory();
    ~TurbulenceFactory();
    
    // this function has a switch for all known turbulence_models
    // and calls the proper class' readParameters()
    
    static Turbulence* create(ProblemSpecP& ps, SimulationStateP& sharedState);
                              
  };

} // End namespace Uintah

#endif /*_TURBULENCEFACTORY_H_*/
