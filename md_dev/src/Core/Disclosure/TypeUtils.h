/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_TypeUtils_H
#define UINTAH_HOMEBREW_TypeUtils_H

#include <sci_defs/compile_defs.h> // for STATIC_BUILD

#ifndef _WIN32
#  include <inttypes.h>
#else
#  include <Core/Util/Endian.h> // for long64 and the like
#endif

#include <cfloat>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

using SCIRun::Point;
using SCIRun::Vector;

class Matrix3;
class Stencil7;
class Stencil4;
class ConnectionList;
class Short27;
class TypeDescription;

typedef int64_t long64;

 const TypeDescription* fun_getTypeDescription(bool*);
 const TypeDescription* fun_getTypeDescription(unsigned char*);
 const TypeDescription* fun_getTypeDescription(int*);
 const TypeDescription* fun_getTypeDescription(short int*);
// const TypeDescription* fun_getTypeDescription(long*);
 const TypeDescription* fun_getTypeDescription(long64*);
 const TypeDescription* fun_getTypeDescription(double*);
 const TypeDescription* fun_getTypeDescription(float*);
 const TypeDescription* fun_getTypeDescription(Point*);
 const TypeDescription* fun_getTypeDescription(Vector*);

// THIS IS A GUESS -> Because of the order of instantiation of
// templates by the IBM xlC compiler, we can not declare the
// fun_getTypeDescription(Stencil7*) in Stencil7.h (where it probably
// should be.)  Instead we have to put it here.  I believe this is
// true for Matrix3 too.  However, both the fun_getTypeDescription of
// Matrix3 and Stencil7 are inplemented in their respective .cc files.

// since these are defined in Math/Grid, and declared here, we need to export them correctly
 const TypeDescription* fun_getTypeDescription(Matrix3*);
 const TypeDescription* fun_getTypeDescription(Short27*);

 const TypeDescription* fun_getTypeDescription(Stencil7*);
 const TypeDescription* fun_getTypeDescription(Stencil4*);
 
 const TypeDescription* fun_getTypeDescription(FILE**);

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
 void fun_getZeroValue(int64_t*);
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
#include   <string>

#if !defined( STATIC_BUILD )

namespace SCIRun {
  using std::string;
  using Uintah::long64;

  template<>  const string find_type_name(long64*);

   const TypeDescription* get_type_description(long64*);

} // namespace SCIRun 
#endif // STATIC_BUILD

#endif


