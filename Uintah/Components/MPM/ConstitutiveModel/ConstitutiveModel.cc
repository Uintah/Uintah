
#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Region.h>

using namespace Uintah::Components;
using SCICore::Geometry::Vector;
using namespace Uintah::Components;
using namespace Uintah::Interface;

ConstitutiveModel::ConstitutiveModel()
{
   // Constructor
   px_label = new VarLabel("p.x", ParticleVariable<Point>::getTypeDescription());
   p_deformationMeasure_label = new VarLabel("p.deformationMeasure",
                                             ParticleVariable<Matrix3>::getTypeDescription());
   p_stress_label = new VarLabel("p.stress",
                                 ParticleVariable<Matrix3>::getTypeDescription());
   p_mass_label = new VarLabel("p.mass",
                               ParticleVariable<double>::getTypeDescription());
   p_volume_label = new VarLabel("p.volume",
                                 ParticleVariable<double>::getTypeDescription());
   g_velocity_label = new VarLabel("g.velocity",
                                   NCVariable<Vector>::getTypeDescription());
}

ConstitutiveModel::~ConstitutiveModel()
{
}
