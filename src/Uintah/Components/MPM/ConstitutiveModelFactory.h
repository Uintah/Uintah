#ifndef _CONSTITUTIVEMODELFACTORY_H_
#define _CONSTITUTIVEMODELFACTORY_H_

// add #include for each ConstitutiveModel here
#include "ElasticConstitutiveModel.h"
#include "CompMooneyRivlin.h"
#include "CompNeoHook.h"
#include "CompNeoHookPlas.h"
#include "HyperElasticDamage.h"
#include "ViscoElasticDamage.h"

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
  static void readParameters(std::ifstream& in, int mat_type, double *p_array);

  // this function has a switch for all known mat_types
  // and calls the proper class' writeParameters()
  static void writeParameters(std::ofstream& out, int mat_type, double *p_array);

  // this function has a switch for all known mat_types
  // and calls the proper class' readParametersAndCreate()
  static ConstitutiveModel* readParametersAndCreate(std::ifstream& in);

  // this function has a switch for all known mat_types
  // and calls the proper class' readRestartParametersAndCreate()
  static ConstitutiveModel* readRestartParametersAndCreate(std::ifstream& in);


  // create the correct kind of model from the mat_type and p_array
  static ConstitutiveModel* create(int mat_type,
							     double *p_array);
  
  // this function has a switch for all known mat_types
  // and calls the proper class' unpackStream()
};


#endif /* _CONSTITUTIVEMODELFACTORY_H_ */
