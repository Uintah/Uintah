
#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/VarTypes.h>

using namespace Uintah::MPM;

ConstitutiveModel::ConstitutiveModel()
{
   // Constructor
   pXLabel = new VarLabel("p.x", ParticleVariable<Point>::getTypeDescription(),
				VarLabel::PositionVariable);

   pDeformationMeasureLabel = new VarLabel("p.deformationMeasure",
                               ParticleVariable<Matrix3>::getTypeDescription());

   pStressLabel = new VarLabel("p.stress",
                                 ParticleVariable<Matrix3>::getTypeDescription());

   pMassLabel = new VarLabel("p.mass",
                               ParticleVariable<double>::getTypeDescription());

   pVolumeLabel = new VarLabel("p.volume",
                                 ParticleVariable<double>::getTypeDescription());
   gVelocityLabel = new VarLabel("g.velocity",
                                   NCVariable<Vector>::getTypeDescription());
   deltLabel = new VarLabel("delt", delt_vartype::getTypeDescription());
}

ConstitutiveModel::~ConstitutiveModel()
{
}
