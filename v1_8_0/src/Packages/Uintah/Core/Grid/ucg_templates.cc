#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Util/NotFinished.h>
#include <utility>
using namespace Uintah;
using std::pair;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif


template class ParticleVariable<SCIRun::Vector>;
template class ParticleVariable<Uintah::Matrix3>;
template class ParticleVariable<SCIRun::Point>;
template class ParticleVariable<double>;
template class ParticleVariable<int>;
//template class ParticleVariable<long int>;
template class ParticleVariable<long64>;

template class NCVariable<SCIRun::Vector>;
template class NCVariable<Uintah::Matrix3>;
template class NCVariable<double>;
template class NCVariable<int>;
template class NCVariable<long64>;

template class CCVariable<SCIRun::Vector>;
template class CCVariable<Uintah::Matrix3>;
template class CCVariable<double>;
template class CCVariable<int>;
template class CCVariable<long64>;

template class SFCXVariable<SCIRun::Vector>;
template class SFCXVariable<Uintah::Matrix3>;
template class SFCXVariable<double>;
template class SFCXVariable<int>;
template class SFCXVariable<long64>;

template class SFCYVariable<SCIRun::Vector>;
template class SFCYVariable<Uintah::Matrix3>;
template class SFCYVariable<double>;
template class SFCYVariable<int>;
template class SFCYVariable<long64>;

template class SFCZVariable<SCIRun::Vector>;
template class SFCZVariable<Uintah::Matrix3>;
template class SFCZVariable<double>;
template class SFCZVariable<int>;
template class SFCZVariable<long64>;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
