
#if 0
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/NCVariable.h>

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

//
// $Log$
// Revision 1.2  2000/09/25 18:12:20  sparker
// do not use covariant return types due to problems with g++
// other linux/g++ fixes
//
// Revision 1.1  2000/05/20 08:09:32  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
//

#endif
