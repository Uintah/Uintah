
#include "LevelField.h"
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>

using namespace Uintah;

#ifdef __sgi
#pragma set woff 1468
#endif

template class LevelField<SCIRun::Vector>;
template class GenericField<LevelMesh, LevelData<SCIRun::Vector> >;
template class LevelField<Matrix3>;
template class GenericField<LevelMesh, LevelData<Matrix3> >;



#ifdef __sgi
#pragma reset woff 1468
#endif
