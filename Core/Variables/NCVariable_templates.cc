#include <Packages/Uintah/Core/Variables/NCVariable.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <utility>
using namespace Uintah;
using std::pair;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif

template class NCVariable<SCIRun::Vector>;
template class NCVariable<Uintah::Matrix3>;
template class NCVariable<double>;
template class NCVariable<float>;
template class NCVariable<int>;
template class NCVariable<long64>;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
