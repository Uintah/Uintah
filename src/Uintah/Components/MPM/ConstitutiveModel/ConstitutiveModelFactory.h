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
  enum ConstitutiveModelType { CM_NULL=0,
			       CM_ELASTIC,
			       CM_MOONEY_RIVLIN,
			       CM_NEO_HOOK,
			       CM_NEO_HOOK_PLAS,
                               CM_HYPER_ELASTIC_DAMAGE,
                               CM_VISCOELASTIC_DAMAGE,
			       CM_MAX };

  // this function has a switch for all known mat_types
  // and calls the proper class' readParameters()
  // addMaterial() calls this
  static void readParameters(ProblemSpecP ps, std::string mat_type, 
			     double *p_array);

  
  // this function has a switch for all known mat_types
  // and calls the proper class' readParametersAndCreate()
  static ConstitutiveModel* readParametersAndCreate(ProblemSpecP ps,
						    std::string mat_type);

  // this function has a switch for all known mat_types
  // and calls the proper class' readRestartParametersAndCreate()
  static ConstitutiveModel* readRestartParametersAndCreate(ProblemSpecP ps,
							 std::string mat_type);


  // create the correct kind of model from the mat_type and p_array
  static ConstitutiveModel* create(std::string mat_type, double *p_array);
  
  
};

} // end namespace Components
} // end namespace Uintah


#endif /* _CONSTITUTIVEMODELFACTORY_H_ */
