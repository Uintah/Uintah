
#if 0
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/FCVariable.h>

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

//
// $Log$
// Revision 1.2.2.1  2000/10/26 10:06:12  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.3  2000/10/18 03:46:46  jas
// Added pressure boundary conditions.
//
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
