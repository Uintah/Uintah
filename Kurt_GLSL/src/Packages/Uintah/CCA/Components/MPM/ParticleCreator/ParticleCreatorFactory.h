#ifndef _PARTICLECREATORFACTORY_H_
#define _PARTICLECREATORFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class ParticleCreator;
  class MPMMaterial;
  class MPMLabel;
  class MPMFlags;

  class ParticleCreatorFactory
  {
  public:
    // this function has a switch for all known mat_types
    
    static ParticleCreator* create(ProblemSpecP& ps, MPMMaterial* mat,
                                   MPMFlags* flags);


  };
} // End namespace Uintah
      
#endif /* _PARTICLECREATORFACTORY_H_ */
