#ifndef Uintah_RAND48_H
#define Uintah_RAND48_H

#if defined(_WIN32) && !defined(HAVE_DRAND48)
#include <Packages/Uintah/Core/Math/share.h>
SCISHARE double drand48();
#endif

#endif
