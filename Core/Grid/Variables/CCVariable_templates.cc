// include before other files so gcc 3.4 can properly instantiate template
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <utility>
using namespace Uintah;
using std::pair;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif

template class CCVariable<SCIRun::Vector>;
template class CCVariable<Uintah::Matrix3>;
template class CCVariable<double>;
template class CCVariable<float>;
template class CCVariable<int>;
template class CCVariable<long64>;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
