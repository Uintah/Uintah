#ifndef __GEOMETRY_OBJECT_FACTORY_H__
#define __GEOMETRY_OBJECT_FACTORY_H__

// add #include for each ConstitutiveModel here
#include "GeometryObject.h"
#include "SphereGeometryObject.h"
#include "CylinderGeometryObject.h"
#include "BoxGeometryObject.h"
#include "TriGeometryObject.h"
#include "UnionGeometryObject.h"
#include "DifferenceGeometryObject.h"
#include "IntersectionGeometryObject.h"
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <string>

using Uintah::Interface::ProblemSpec;
using Uintah::Interface::ProblemSpecP;
using namespace Uintah::Components;

namespace Uintah {
namespace Components {
class GeometryObject;

class GeometryObjectFactory
{
public:
  enum GeometryObjectType { GO_NULL=0,
			    GO_BOX,
			    GO_SPHERE,
			    GO_CYLINDER,
			    GO_TRI,
			    GO_UNION,
			    GO_DIFFERENCE,
			    GO_INTERSECTION,
			    GO_MAX };

  // this function has a switch for all known go_types
  // and calls the proper class' readParameters()
  // addMaterial() calls this
  static void readParameters(ProblemSpecP ps, std::string go_type, 
			     double *p_array);

  
  // this function has a switch for all known go_types
  // and calls the proper class' readParametersAndCreate()
  static GeometryObject* readParametersAndCreate(ProblemSpecP ps,
						    std::string go_type);

  // this function has a switch for all known go_types
  // and calls the proper class' readRestartParametersAndCreate()
  static GeometryObject* readRestartParametersAndCreate(ProblemSpecP ps,
							 std::string go_type);


  // create the correct kind of model from the go_type and p_array
  static GeometryObject* create(std::string go_type, double *p_array);
  
  
};

} // end namespace Components
} // end namespace Uintah


#endif /* __GEOMETRY_OBJECT_FACTORY_H__ */
