#ifndef _EQUATIONOFSTATEFACTORY_H_
#define _EQUATIONOFSTATEFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class MPMEquationOfState;

  class MPMEquationOfStateFactory
  {
  public:
    // this function has a switch for all known mat_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static MPMEquationOfState* create(ProblemSpecP& ps);
  };
} // End namespace Uintah
      
#endif /* _EQUATIONOFSTATEFACTORY_H_ */
