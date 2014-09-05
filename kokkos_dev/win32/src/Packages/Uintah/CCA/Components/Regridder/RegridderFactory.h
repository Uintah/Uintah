#ifndef Packages_Uintah_CCA_Components_Regridders_RegridderFactory_h
#define Packages_Uintah_CCA_Components_Regridders_RegridderFactory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>

namespace Uintah {

  class ProcessorGroup;

  class RegridderFactory
  {
  public:
    // this function has a switch for all known regridders
    
    static RegridderCommon* create(ProblemSpecP& ps,
                                   const ProcessorGroup* world);


  };
} // End namespace Uintah


#endif
