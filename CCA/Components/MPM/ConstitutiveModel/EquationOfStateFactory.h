#ifndef _EQUATIONOFSTATEFACTORY_H_
#define _EQUATIONOFSTATEFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <string>
namespace Uintah {

  class EquationOfState;

  class EquationOfStateFactory
  {
  public:
    // this function has a switch for all known mat_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static EquationOfState* create(ProblemSpecP& ps);
  };
} // End namespace Uintah
      
#endif /* _EQUATIONOFSTATEFACTORY_H_ */
