#ifndef UINTAH_HOMEBREW_TypeUtils_H
#define UINTAH_HOMEBREW_TypeUtils_H

#include <inttypes.h>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace Uintah {

using SCIRun::Point;
using SCIRun::Vector;

class Matrix3;
class Stencil7;
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

} // End namespace UintahStencil7
   
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


