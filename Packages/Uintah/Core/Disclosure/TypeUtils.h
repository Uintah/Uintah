#ifndef UINTAH_HOMEBREW_TypeUtils_H
#define UINTAH_HOMEBREW_TypeUtils_H

#include <inttypes.h>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace Uintah {

class Matrix3;
class TypeDescription;

using namespace SCIRun;

typedef int64_t long64;

const TypeDescription* fun_getTypeDescription(bool*);
const TypeDescription* fun_getTypeDescription(int*);
const TypeDescription* fun_getTypeDescription(short int*);
//const TypeDescription* fun_getTypeDescription(long*);
const TypeDescription* fun_getTypeDescription(long64*);
const TypeDescription* fun_getTypeDescription(double*);
const TypeDescription* fun_getTypeDescription(Point*);
const TypeDescription* fun_getTypeDescription(Vector*);
const TypeDescription* fun_getTypeDescription(Matrix3*);

} // End namespace Uintah
   
#include <Core/Datatypes/TypeName.h>
#include <string>
namespace SCIRun {
 using std::string;
 using Uintah::long64;
template<> const string find_type_name(long64*);

const TypeDescription* get_type_description(long64*);

} // namespace SCIRun 

#endif


