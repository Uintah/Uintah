#ifndef UINTAH_HOMEBREW_TypeUtils_H
#define UINTAH_HOMEBREW_TypeUtils_H

#include <inttypes.h>
#include <float.h>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace Uintah {

using SCIRun::Point;
using SCIRun::Vector;

class Matrix3;
class Stencil7;
class ConnectionList;
class Short27;
class TypeDescription;

typedef int64_t long64;

const TypeDescription* fun_getTypeDescription(bool*);
const TypeDescription* fun_getTypeDescription(unsigned char*);
const TypeDescription* fun_getTypeDescription(int*);
const TypeDescription* fun_getTypeDescription(short int*);
//const TypeDescription* fun_getTypeDescription(long*);
const TypeDescription* fun_getTypeDescription(long64*);
const TypeDescription* fun_getTypeDescription(double*);
const TypeDescription* fun_getTypeDescription(float*);
const TypeDescription* fun_getTypeDescription(Point*);
const TypeDescription* fun_getTypeDescription(Vector*);
const TypeDescription* fun_getTypeDescription(Matrix3*);

// THIS IS A GUESS -> Because of the order of instantiation of
// templates by the IBM xlC compiler, we can not declare the
// fun_getTypeDescription(Stencil7*) in Stencil7.h (where it probably
// should be.)  Instead we have to put it here.  I believe this is
// true for Matrix3 too.  However, both the fun_getTypeDescription of
// Matrix3 and Stencil7 are inplemented in their respective .cc files.
const TypeDescription* fun_getTypeDescription(Stencil7*);
const TypeDescription* fun_getTypeDescription(Short27*);

// Added by Oren for implicit ICE AMR pressure solver type that
// appears in ICELabel.cc.
const TypeDescription* fun_getTypeDescription(ConnectionList*);

// these functions are for getting safe values of types
// return back the value in the argument (so we don't have to include
// Vector.h here)
void fun_getLargeValue(double*);
void fun_getSmallValue(double*);
void fun_getZeroValue(double*);
void fun_getZeroValue(bool*);
void fun_getZeroValue(long64*);
void fun_getZeroValue(Vector*);

// these functions should never get called - they just exist for
// template completeness
void fun_getLargeValue(bool*);
void fun_getSmallValue(bool*);
void fun_getLargeValue(long64*);
void fun_getSmallValue(long64*);
void fun_getLargeValue(Vector*);
void fun_getSmallValue(Vector*);


} // End namespace Uintah
   
#include <Core/Datatypes/TypeName.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
 using std::string;
 using Uintah::long64;
template<> const string find_type_name(long64*);

const TypeDescription* get_type_description(long64*);

} // namespace SCIRun 

#endif


