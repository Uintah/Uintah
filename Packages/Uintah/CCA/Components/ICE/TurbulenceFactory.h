#ifndef _TURBULENCEFACTORY_H_
#define _TURBULENCEFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <string>
namespace Uintah {

  class Turbulence;

  class TurbulenceFactory
  {
  public:
    // this function has a switch for all known turbulence_models
    // and calls the proper class' readParameters()
    
    static Turbulence* create(ProblemSpecP& ps,
                              bool& d_Turb);
  };

} // End namespace Uintah

#endif /*_TURBULENCEFACTORY_H_*/
