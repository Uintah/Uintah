#ifndef UINTAH_HOMEBREW_TypeUtils_H
#define UINTAH_HOMEBREW_TypeUtils_H

namespace SCIRun {
  class Point;
  class Vector;
}

namespace Uintah {

class Matrix3;
class TypeDescription;

using namespace SCIRun;

typedef long long long64;

const TypeDescription* fun_getTypeDescription(bool*);
const TypeDescription* fun_getTypeDescription(int*);
const TypeDescription* fun_getTypeDescription(short int*);
const TypeDescription* fun_getTypeDescription(long*);
const TypeDescription* fun_getTypeDescription(long64*);
const TypeDescription* fun_getTypeDescription(double*);
const TypeDescription* fun_getTypeDescription(Point*);
const TypeDescription* fun_getTypeDescription(Vector*);
const TypeDescription* fun_getTypeDescription(Matrix3*);

} // End namespace Uintah
   


#endif


