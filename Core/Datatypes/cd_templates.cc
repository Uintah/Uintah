
#include "LevelField.h"
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>

using namespace Uintah;

#ifdef __sgi
#pragma set woff 1468
#endif

template class LevelField<SCIRun::Vector>;
template class GenericField<LevelMesh, LevelData<SCIRun::Vector> >;
template class LevelField<Matrix3>;
template class GenericField<LevelMesh, LevelData<Matrix3> >;

void Pio(Piostream& stream, LevelData<int>& data);
void Pio(Piostream& stream, LevelData<long>& data);
void Pio(Piostream& stream, LevelData<double>& data);
void Pio(Piostream& stream, LevelData<Vector>& data);
void Pio(Piostream& stream, LevelData<Matrix3>& data);

// template class LevelFieldSFI<SCIRun::Vector>;
// template class LevelFieldSFI<Matrix3>;
// template class LevelFieldSFI<double>;
// template class LevelFieldSFI<int>;
// template class LevelFieldSFI<long>;


#ifdef __sgi
#pragma reset woff 1468
#endif
