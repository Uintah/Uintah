/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/DataWarehouse.h>

using Uintah::Components::Arches;
using namespace Uintah::Grid;

namespace Uintah {
namespace Components {

PhysicalConstants::PhysicalConstants()
{
}

PhysicalConstants::~PhysicalConstants()
{
}
#if 0
void PhysicalConstants::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("PhysicalConstants");

  db->require("gravity", d_gravity); //Vector
  db->require("pressure", d_absPressure);
  db->require("viscosity", d_viscosity);

}
#endif


} // end namespace Components
} // end namespace Uintah

