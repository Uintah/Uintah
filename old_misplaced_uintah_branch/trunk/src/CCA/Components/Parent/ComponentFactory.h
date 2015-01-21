#ifndef Packages_Uintah_CCA_Components_Parent_ComponentFactory_h
#define Packages_Uintah_CCA_Components_Parent_ComponentFactory_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <string>

#include <CCA/Components/Parent/uintahshare.h>

namespace Uintah {

  class ProcessorGroup;
  class UintahParallelComponent;

  class UINTAHSHARE ComponentFactory  {
  
  public:
    // this function has a switch for all known components
    
    static UintahParallelComponent* create(ProblemSpecP& ps, const ProcessorGroup* world, 
                                           bool doAMR, std::string component, std::string uda);


  };
} // End namespace Uintah


#endif
