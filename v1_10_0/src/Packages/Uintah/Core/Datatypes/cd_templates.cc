
#include "LevelField.h"
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Util/NotFinished.h>
#include <utility>
using namespace Uintah;
using std::pair;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif

template class LevelField<SCIRun::Vector>;
template class GenericField<LevelMesh, LevelData<SCIRun::Vector> >;
template class LevelField<Matrix3>;
template class GenericField<LevelMesh, LevelData<Matrix3> >;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
