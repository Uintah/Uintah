// include before other files so gcc 3.4 can properly instantiate template
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <utility>
using namespace Uintah;
using std::pair;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif

template class SFCYVariable<SCIRun::Vector>;
template class SFCYVariable<Uintah::Matrix3>;
template class SFCYVariable<double>;
template class SFCYVariable<float>;
template class SFCYVariable<int>;
template class SFCYVariable<long64>;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
