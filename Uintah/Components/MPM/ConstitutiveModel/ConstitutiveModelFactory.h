#ifndef _CONSTITUTIVEMODELFACTORY_H_
#define _CONSTITUTIVEMODELFACTORY_H_

// add #include for each ConstitutiveModel here
#include "ElasticConstitutiveModel.h"
#include "CompMooneyRivlin.h"
#include "CompNeoHook.h"
#include "CompNeoHookPlas.h"
#include "HyperElasticDamage.h"
#include "ViscoElasticDamage.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <string>

using Uintah::Interface::ProblemSpec;
using Uintah::Interface::ProblemSpecP;

namespace Uintah {
namespace Components {
class ConstitutiveModel;

class ConstitutiveModelFactory
{
public:
 

  // this function has a switch for all known mat_types
  // and calls the proper class' readParameters()
  // addMaterial() calls this
  static ConstitutiveModel* create(ProblemSpecP ps);

    
};

} // end namespace Components
} // end namespace Uintah


#endif /* _CONSTITUTIVEMODELFACTORY_H_ */
