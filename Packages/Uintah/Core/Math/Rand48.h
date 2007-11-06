#ifndef Uintah_RAND48_H
#define Uintah_RAND48_H

#if defined(_WIN32) && !defined(HAVE_DRAND48)
#include <Packages/Uintah/Core/Math/uintahshare.h>
UINTAHSHARE double drand48();
#endif

#endif
