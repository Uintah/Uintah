#ifndef _EQUATIONOFSTATEFACTORY_H_
#define _EQUATIONOFSTATEFACTORY_H_

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class MPMEquationOfState;

  class MPMEquationOfStateFactory
  {
  public:
    // this function has a switch for all known mat_types
    static MPMEquationOfState* create(ProblemSpecP& ps);
    static MPMEquationOfState* createCopy(const MPMEquationOfState* cm);
  };
} // End namespace Uintah
      
#endif /* _EQUATIONOFSTATEFACTORY_H_ */
