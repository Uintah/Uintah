
#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah::MPM;

ConstitutiveModel::ConstitutiveModel() : d_fudge(1.0)
{
   // Constructor

}

ConstitutiveModel::~ConstitutiveModel()
{
}




