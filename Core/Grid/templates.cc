
#if 0
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/FCVariable.h>

#ifdef __sgi
#pragma set woff 1468
#endif

template class ParticleVariable<double>;
template class ParticleVariable<Point>;
template class ParticleVariable<Vector>;

#if 0
template class NCVariable<double>;
template class NCVariable<Point>;
#endif
template class NCVariable<Vector>;
template class FCVariable<Vector>;
template class FCVariable<double>;


#endif
