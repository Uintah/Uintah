
#include "ParticleVariable.h"
#include "CCVariable.h"
#include "NCVariable.h"
#include "SFCXVariable.h"
#include "SFCYVariable.h"
#include "SFCZVariable.h"
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Util/NotFinished.h>
#include <utility>
using namespace Uintah;
using std::pair;

#ifdef __sgi
#pragma set woff 1468
#endif


template class ParticleVariable<SCIRun::Vector>;
template class ParticleVariable<Uintah::Matrix3>;
template class ParticleVariable<SCIRun::Point>;
template class ParticleVariable<double>;
template class ParticleVariable<int>;
template class ParticleVariable<long int>;

template class NCVariable<SCIRun::Vector>;
template class NCVariable<Uintah::Matrix3>;
template class NCVariable<double>;
template class NCVariable<int>;
template class NCVariable<long int>;

template class CCVariable<SCIRun::Vector>;
template class CCVariable<Uintah::Matrix3>;
template class CCVariable<double>;
template class CCVariable<int>;
template class CCVariable<long int>;

template class SFCXVariable<SCIRun::Vector>;
template class SFCXVariable<Uintah::Matrix3>;
template class SFCXVariable<double>;
template class SFCXVariable<int>;
template class SFCXVariable<long int>;

template class SFCYVariable<SCIRun::Vector>;
template class SFCYVariable<Uintah::Matrix3>;
template class SFCYVariable<double>;
template class SFCYVariable<int>;
template class SFCYVariable<long int>;

template class SFCZVariable<SCIRun::Vector>;
template class SFCZVariable<Uintah::Matrix3>;
template class SFCZVariable<double>;
template class SFCZVariable<int>;
template class SFCZVariable<long int>;

#ifdef __sgi
#pragma reset woff 1468
#endif
