#ifndef Packages_Uintah_CCA_Components_Solvers_SolverFactory_h
#define Packages_Uintah_CCA_Components_Solvers_SolverFactory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>

namespace Uintah {

  class ProcessorGroup;

  class SolverFactory
  {
  public:
    // this function has a switch for all known solvers
    
    static SolverInterface* create(ProblemSpecP& ps, const ProcessorGroup* world, string cmdline);


  };
} // End namespace Uintah


#endif
