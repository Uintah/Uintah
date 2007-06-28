#ifndef UINTAH_HOMEBREW_TypeUtils_H
#define UINTAH_HOMEBREW_TypeUtils_H

#ifndef _WIN32
#include <inttypes.h>
#else
#include <SCIRun/Core/Util/Endian.h> // for long64 and the like
#endif
#include <SCIRun/Core/Geometry/Point.h>
#include <SCIRun/Core/Geometry/Vector.h>
#include <float.h>

using SCIRun::Point;
using SCIRun::Vector;

#include <Core/Disclosure/uintahshare.h>
namespace Uintah {

class Matrix3;
class Stencil7;
class ConnectionList;
class Short27;
class TypeDescription;

typedef int64_t long64;

UINTAHSHARE const TypeDescription* fun_getTypeDescription(bool*);
UINTAHSHARE const TypeDescription* fun_getTypeDescription(unsigned char*);
UINTAHSHARE const TypeDescription* fun_getTypeDescription(int*);
UINTAHSHARE const TypeDescription* fun_getTypeDescription(short int*);
//UINTAHSHARE const TypeDescription* fun_getTypeDescription(long*);
UINTAHSHARE const TypeDescription* fun_getTypeDescription(long64*);
UINTAHSHARE const TypeDescription* fun_getTypeDescription(double*);
UINTAHSHARE const TypeDescription* fun_getTypeDescription(float*);
UINTAHSHARE const TypeDescription* fun_getTypeDescription(Point*);
UINTAHSHARE const TypeDescription* fun_getTypeDescription(Vector*);

// THIS IS A GUESS -> Because of the order of instantiation of
// templates by the IBM xlC compiler, we can not declare the
// fun_getTypeDescription(Stencil7*) in Stencil7.h (where it probably
// should be.)  Instead we have to put it here.  I believe this is
// true for Matrix3 too.  However, both the fun_getTypeDescription of
// Matrix3 and Stencil7 are inplemented in their respective .cc files.

#include <Core/Math/uintahshare.h>
// since these are defined in Math/Grid, and declared here, we need to export them correctly
UINTAHSHARE const TypeDescription* fun_getTypeDescription(Matrix3*);
UINTAHSHARE const TypeDescription* fun_getTypeDescription(Short27*);

#include <Core/Grid/uintahshare.h>
UINTAHSHARE const TypeDescription* fun_getTypeDescription(Stencil7*);

// Added by Oren for implicit ICE AMR pressure solver type that
// appears in ICELabel.cc.
UINTAHSHARE const TypeDescription* fun_getTypeDescription(ConnectionList*);

#include <Core/Disclosure/uintahshare.h>

// these functions are for getting safe values of types
// return back the value in the argument (so we don't have to include
// Vector.h here)
UINTAHSHARE void fun_getLargeValue(double*);
UINTAHSHARE void fun_getSmallValue(double*);
UINTAHSHARE void fun_getZeroValue(double*);
UINTAHSHARE void fun_getZeroValue(bool*);
UINTAHSHARE void fun_getZeroValue(long64*);
UINTAHSHARE void fun_getZeroValue(Vector*);

// these functions should never get called - they just exist for
// template completeness
UINTAHSHARE void fun_getLargeValue(bool*);
UINTAHSHARE void fun_getSmallValue(bool*);
UINTAHSHARE void fun_getLargeValue(long64*);
UINTAHSHARE void fun_getSmallValue(long64*);
UINTAHSHARE void fun_getLargeValue(Vector*);
UINTAHSHARE void fun_getSmallValue(Vector*);


} // End namespace Uintah
   
#include <SCIRun/Core/Datatypes/TypeName.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

// typename.h redefines UINTAHSHARE
#include <Core/Disclosure/uintahshare.h>
namespace SCIRun {
 using std::string;
 using Uintah::long64;
template<> UINTAHSHARE const string find_type_name(long64*);

UINTAHSHARE const TypeDescription* get_type_description(long64*);

} // namespace SCIRun 

#endif


